"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ZEUS TRADING SYSTEM  v1.0                           ║
║                                                                          ║
║   통합 퀀트 트레이딩 툴킷 (commander + tracker + sentiment 합본)          ║
║                                                                          ║
║   [1] Position Commander  — 포지션 사이징 (Risk Parity / Kelly / EW)     ║
║   [2] Portfolio Tracker   — 실시간 P&L + SL/TP 경보 + 알파 추적          ║
║   [3] Sentiment Scanner   — 뉴스감성 + 어닝서프라이즈 + 내부자거래        ║
║   [4] Full Report         — 3개 모듈 동시 실행 → 통합 HTML               ║
║                                                                          ║
║   의존성: pip install yfinance vaderSentiment                            ║4
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False


# ══════════════════════════════════════════════════════════════════════════
#  ▶ 전역 설정
# ══════════════════════════════════════════════════════════════════════════

# — 포지션 제약 —
MAX_WEIGHT    : float = 0.30
MIN_WEIGHT    : float = 0.02
KELLY_FRAC    : float = 0.50
RISK_FREE     : float = 0.045
VAR_CONF      : float = 0.95

# — ATR 손절/익절 —
ATR_PERIOD    : int   = 14
TARGET_RR     : float = 2.0    # 목표 손익비: TP거리 = SL거리 × RR, EV>0 최소승률 = 1/(1+RR) = 33%
# 변동성 체제별 SL 배수 (ATR% 기준 자동 결정, 클리핑 없음)
# ATR%<1.0 → 1.5x(저변동)  1~2 → 2.0x(중변동)  2~3.5 → 2.5x(고변동)  >3.5 → 3.0x(초고변동)

# — 거래비용 — backtest TransactionCostEngine(zeus_backtest.py)과 동일 수치 사용
# ─────────────────────────────────────────────────────────────────────────────
# [통일 근거]
# backtest와 실전 모두 같은 비용 모델을 써야 성과 비교가 유효.
# 기존: COMM_PER_SHARE=0.005(주당 고정), SLIPPAGE_PCT=0.0005(거래대금 0.05%)
#   문제: 주가 $10짜리 1000주 → 수수료 $5 = 0.05% (맞음)
#          주가 $500짜리 10주  → 수수료 $5 = 0.10% (불일치, 고가주 과대 계상)
# 변경: 거래대금 비례 방식으로 통일 (backtest와 동일)
#
# 수수료 근거: Fidelity/Schwab 리테일 브로커 실제 요율 0.05% (Goldman 2023)
# 슬리피지 근거: Kissell(2013) 유동성 계층별 —
#   일평균거래량 >$5M  → 0.05%(LOW),  >$1M → 0.10%(MEDIUM),
#   >$300K → 0.20%(HIGH), 이하 → 0.50%(VERY_HIGH)
# 시장충격 근거: Almgren & Chriss(2001) impact = 0.5 × √participation × σ_daily
COMM_PCT       : float = 0.0005   # 거래대금 비례 수수료 0.05% (단방향)
# 슬리피지: 거래 실행 시 달러거래량으로 계층 결정 (하단 _calc_trade_cost 함수)
# 아래 기본값은 중형주($1~5M ADV) 기준 단방향 0.10%
SLIPPAGE_PCT   : float = 0.0010   # 기본 슬리피지 0.10% (calc_rebalance fallback용)
# ── 참고: 주당 고정비 제거 (COMM_PER_SHARE, MIN_COMM 삭제) ──────────────
# IBKR Tiered Pro 기준으로 운영 시 COMM_PCT=0.0005가 더 정확한 근사임

# — 세금 —
SHORT_TAX : float = 0.37
LONG_TAX  : float = 0.20

# — 유동성 —
MIN_DOLLAR_VOL : float = 5_000_000

# — 리밸런싱 —
REBAL_THRESHOLD : float = 0.05

# — 데이터 —
DATA_PERIOD : str = "3y"
STALE_DAYS  : int = 3
CLIP_ITER   : int = 10
BENCHMARK   : str = "SPY"

# — 감성 가중치 —
W_NEWS   : float = 0.30
W_EARN   : float = 0.25
W_ANAL   : float = 0.25
W_SHORT  : float = 0.10
W_INSID  : float = 0.10
BUY_THR  : float = 0.30
SELL_THR : float = -0.30

# — 파일 —
POSITIONS_FILE : str = "zeus_positions.json"
OUTPUT_FILE    : str = "zeus_report.html"
FACTORS_FILE   : str = "zeus_sentiment_factors.json"


# ══════════════════════════════════════════════════════════════════════════
#  SECTION A — 공통 유틸리티
# ══════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """OHLCV + 달러 거래대금. MultiIndex 완전 호환."""
    print(f"  📡 OHLCV 수집 ({len(tickers)}종목 / {DATA_PERIOD})...")
    raw = yf.download(tickers, period=DATA_PERIOD, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"];  high   = raw["High"]
        low    = raw["Low"];    volume = raw["Volume"]
    else:
        t = tickers[0]
        close  = raw[["Close"]].rename(columns={"Close":  t})
        high   = raw[["High"]].rename(columns={"High":    t})
        low    = raw[["Low"]].rename(columns={"Low":      t})
        volume = raw[["Volume"]].rename(columns={"Volume": t})

    valid  = [t for t in tickers if t in close.columns and not close[t].dropna().empty]
    failed = set(tickers) - set(valid)
    if failed:
        print(f"  ⚠️  수집 실패: {sorted(failed)}")

    close  = close[valid].dropna(how="all")
    high   = high[valid].dropna(how="all")
    low    = low[valid].dropna(how="all")
    volume = volume[valid].dropna(how="all")
    dollar_vol = (close * volume).tail(20).mean()

    last = close.index[-1]
    try:
        age = (pd.Timestamp.now() - last.tz_localize(None)).days
    except Exception:
        age = 0
    if age > STALE_DAYS:
        print(f"  ⚠️  STALE DATA: {last.date()} ({age}일 전)")

    print(f"  ✅ {len(valid)}종목 | {len(close)}거래일 | 최신: {last.date()}")
    return close, high, low, dollar_vol


def liquidity_filter(tickers: List[str], dollar_vol: pd.Series) -> List[str]:
    passed  = [t for t in tickers if dollar_vol.get(t, 0) >= MIN_DOLLAR_VOL]
    removed = [t for t in tickers if dollar_vol.get(t, 0) < MIN_DOLLAR_VOL]
    for t in removed:
        print(f"  🚫 유동성 미달 제외: {t} (${dollar_vol.get(t,0)/1e6:.1f}M)")
    return passed


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    pc  = close.shift(1)
    tr  = pd.concat([(high-low), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    val = float(tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1])
    return val if (np.isfinite(val) and val > 0) else float((high-low).mean())


def calc_sl_tp(px: float, atr: float) -> Tuple[float, float, float, str]:
    """
    변동성 체제별 ATR 기반 SL/TP.

    원칙:
      SL = ATR × 체제배수  (노이즈 밖 배치, 클리핑 없음)
      TP = SL거리 × TARGET_RR  (손익비 역산)
      EV > 0 조건: 승률 > 1/(1+RR) = 33%

    변동성 체제 (ATR%):
      < 1.0  → SL배수 1.5x  저변동 (TLT, LMT)
      1~2    → SL배수 2.0x  중변동 (MSFT, JPM)
      2~3.5  → SL배수 2.5x  고변동 (AMZN, META)
      > 3.5  → SL배수 3.0x  초고변동 (NVDA, TSLA)
    """
    atr_pct = atr / px * 100
    if atr_pct < 1.0:   sl_mult, regime = 1.5, '저변동'
    elif atr_pct < 2.0: sl_mult, regime = 2.0, '중변동'
    elif atr_pct < 3.5: sl_mult, regime = 2.5, '고변동'
    else:               sl_mult, regime = 3.0, '초고변동'
    sl_dist = sl_mult * atr
    tp_dist = sl_dist * TARGET_RR
    sl = round(px - sl_dist, 2)
    tp = round(px + tp_dist, 2)
    return sl, tp, TARGET_RR, regime


def compute_stats(close: pd.DataFrame) -> pd.DataFrame:
    r  = close.pct_change().dropna()
    mu = r.mean() * 252;  sig = r.std() * np.sqrt(252)
    sr = (mu - RISK_FREE) / sig.replace(0, np.nan)
    mdd = pd.Series([
        ((1+r[c]).cumprod()/(1+r[c]).cumprod().cummax()-1).min()
        for c in close.columns], index=close.columns)
    return pd.DataFrame({"연수익(%)": (mu*100).round(1), "연변동(%)": (sig*100).round(1),
                          "Sharpe": sr.round(2), "MDD(%)": (mdd*100).round(1)})


def portfolio_risk(weights: pd.Series, close: pd.DataFrame) -> Dict:
    r    = close[weights.index].pct_change().dropna()
    w    = weights.values
    cov  = r.cov().values * 252
    pvol = float(np.sqrt(w @ cov @ w)) * 100
    wavg = float(w @ np.sqrt(np.diag(cov)) * 100)
    div  = max(0.0, (wavg - pvol) / max(wavg, 1e-6) * 100)
    daily = (r * weights).sum(axis=1)
    q    = np.percentile(daily, (1-VAR_CONF)*100)
    return {"port_vol": round(pvol,2), "wavg_vol": round(wavg,2),
            "div_benefit": round(div,1),
            "var_95": round(float(q)*100,2),
            "cvar_95": round(float(daily[daily<=q].mean())*100,2)}


# ══════════════════════════════════════════════════════════════════════════
#  SECTION B — 포지션 사이징 (Commander)
# ══════════════════════════════════════════════════════════════════════════

def _clip_norm(w: pd.Series) -> pd.Series:
    """
    MAX_WEIGHT clip 반복 수렴 후 MIN_WEIGHT 1회 제거.
    구버전: 매 반복마다 MIN_WEIGHT 제거 → sum이 줄며 진동 발생
    현재:   수렴 후 MIN_WEIGHT 1회 제거 → 수렴 보장
    """
    w = w.clip(lower=0)
    for _ in range(CLIP_ITER):
        s = w.sum()
        if s <= 0:
            return pd.Series(1 / len(w), index=w.index)
        w = w / s
        w = w.clip(upper=MAX_WEIGHT)
    # MIN_WEIGHT 미달 종목 제거는 수렴 후 1회만
    w[w < MIN_WEIGHT] = 0.0
    s = w.sum()
    return w / s if s > 0 else pd.Series(1 / len(w), index=w.index)


def strategy_risk_parity(close: pd.DataFrame) -> pd.Series:
    """
    Constrained ERC (Equal Risk Contribution).

    수식: RC_i = w_i × (Σw)_i / σ_p  →  목표: 모든 i에서 RC_i/σ_p = 1/n
    Newton-Raphson: w ← w × (σ/n) / (RC + ε)

    MAX_WEIGHT 제약:
      단순 clip → 재정규화 → 다시 30% 초과 발생 (수렴 불완전)
      현재: ERC 업데이트 후 "반복 clip-normalize" → MAX_WEIGHT 완전 준수 후 다음 스텝
      수렴 판별: 연속 비중 변화 < 1e-8
    """
    r   = close.pct_change().dropna()
    cov = r.cov().values * 252
    n   = len(close.columns)
    w   = np.ones(n) / n
    prev_w = w.copy()

    for _ in range(500):
        sigma = float(np.sqrt(w @ cov @ w))
        if sigma < 1e-10:
            break
        g  = cov @ w / sigma
        rc = w * g
        w  = w * ((sigma / n) / (rc + 1e-10))
        w  = np.maximum(w, 0)
        # MAX_WEIGHT 완전 준수: 재정규화 후에도 초과 안 될 때까지 반복
        for _ in range(30):
            if w.max() <= MAX_WEIGHT + 1e-9:
                break
            w = np.clip(w, 0, MAX_WEIGHT)
            s = w.sum()
            if s > 0:
                w /= s
        # 수렴 판별
        if np.max(np.abs(w - prev_w)) < 1e-8:
            break
        prev_w = w.copy()

    return _clip_norm(pd.Series(w, index=close.columns))


def strategy_half_kelly(close: pd.DataFrame) -> pd.Series:
    """
    공분산 고려 Fractional Kelly: f* = Σ⁻¹(μ−rf) × KELLY_FRAC

    수정 이력:
    - v1 (구버전): f*_i = (mu_i - rf) / sigma_i²  (단일 종목 독립 가정, 공분산 무시)
                   → 고상관 종목 동시 편입 시 합산 비중 과배분 위험
    - v2 (현재):   f* = Sigma⁻¹ × (mu - rf)  (공분산 고려, Markowitz-Kelly 연결)
                   → 상관관계 높은 종목은 자동으로 비중 감소
    폴백: Sigma 역행렬 계산 불가(특이행렬) 시 단순 Kelly로 대체
    """
    r      = close.pct_change().dropna()
    mu     = r.mean() * 252
    excess = (mu - RISK_FREE).values
    cov    = r.cov().values * 252

    try:
        cov_inv = np.linalg.inv(cov)
        kw_raw  = cov_inv @ excess * KELLY_FRAC        # Sigma⁻¹(μ-rf) × frac
        kw      = pd.Series(kw_raw, index=close.columns).clip(lower=0)
    except np.linalg.LinAlgError:
        # 특이행렬 폴백: 단순 독립 Kelly
        print("  ⚠️  공분산 역행렬 계산 불가 → 단순 Kelly 대체")
        sigma2 = (r.std() * np.sqrt(252)) ** 2
        kw     = ((mu - RISK_FREE) / sigma2.replace(0, np.nan)).fillna(0).clip(lower=0) * KELLY_FRAC

    if kw.sum() <= 0:
        print("  ⚠️  초과수익 ≤ 0 → Risk Parity 대체")
        return strategy_risk_parity(close)
    return _clip_norm(kw)


def strategy_equal_weight(close: pd.DataFrame) -> pd.Series:
    return _clip_norm(pd.Series(1.0/len(close.columns), index=close.columns))


def calc_shares_lrm(weights: pd.Series, prices: pd.Series, capital: float) -> Tuple[pd.Series, float]:
    """
    Largest Remainder Method — 반올림 오차 최소화.
    단순 int() 절삭 시 자본 13%+ 현금 방치 → LRM으로 1% 미만으로 감소.
    """
    active = weights[weights >= MIN_WEIGHT].index.tolist()
    wa = weights[active] / weights[active].sum(); px = prices[active]
    ideal = capital * wa / px; floors = ideal.apply(np.floor).astype(int); rems = ideal - floors
    cash = capital - float((floors * px).sum())
    for tk in rems.sort_values(ascending=False).index:
        p = float(px[tk])
        if cash >= p: floors[tk] += 1; cash -= p
        else: break
    return floors, round(cash, 4)


def _calc_trade_cost(n_shares: int, price: float,
                     avg_daily_dollar_vol: float = 0.0) -> float:
    """
    단일 거래 왕복 비용 (수수료 + 슬리피지) 계산.
    backtest TransactionCostEngine과 동일 수치 적용.

    수수료: COMM_PCT × 거래대금 (단방향)
    슬리피지: Kissell(2013) ADV 계층 기반 (단방향)
      ADV > $5M  → 0.05%   (대형 유동성)
      ADV > $1M  → 0.10%   (중형)
      ADV > $300K→ 0.20%   (소형)
      이하       → 0.50%   (초소형)
    왕복비용 = (수수료 + 슬리피지) × 2
    """
    if n_shares <= 0 or price <= 0:
        return 0.0
    trade_val = n_shares * price

    # 수수료 (단방향)
    comm_one_way = trade_val * COMM_PCT

    # 슬리피지 계층 (Kissell 2013)
    adv = float(avg_daily_dollar_vol)
    if adv >= 5_000_000:
        slip_pct = 0.0005   # 0.05%
    elif adv >= 1_000_000:
        slip_pct = 0.0010   # 0.10%
    elif adv >= 300_000:
        slip_pct = 0.0020   # 0.20%
    else:
        slip_pct = 0.0050   # 0.50%

    slip_one_way = trade_val * slip_pct

    # 왕복 (매수 + 매도)
    return round((comm_one_way + slip_one_way) * 2.0, 4)


def calc_rebalance(target: pd.Series, existing: Dict[str, int], prices: pd.Series,
                   dollar_vols: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    리밸런싱 주문 계산.
    dollar_vols: 종목별 일평균 달러거래량 (있으면 유동성 계층별 슬리피지 적용)
    """
    all_tk  = sorted(set(target.index) | set(existing.keys()))
    pv      = sum(existing.get(t,0) * float(prices.get(t,0)) for t in all_tk)
    tv      = sum(int(target.get(t,0)) * float(prices.get(t,0)) for t in target.index)
    rows = []
    for tk in all_tk:
        cur = existing.get(tk,0); tgt = int(target.get(tk,0)); px = float(prices.get(tk,0))
        if px <= 0: continue
        delta = tgt - cur; drift = abs((tgt*px)/max(tv,1) - (cur*px)/max(pv,1))
        if delta == 0:             action = "HOLD"
        elif drift<REBAL_THRESHOLD and cur>0: action = "HOLD (괴리 미달)"; delta = 0
        elif delta > 0:            action = "BUY"
        else:                      action = "SELL"
        n = abs(delta)
        # FIX3: 거래량 기반 계층 슬리피지 적용 (backtest와 동일 모델)
        adv_dollar = float(dollar_vols.get(tk, 0)) if dollar_vols is not None else 0.0
        trade_cost = _calc_trade_cost(n, px, adv_dollar)
        rows.append({"ticker":tk, "현재수량":cur, "목표수량":tgt, "매매수량":delta,
                     "action":action, "현재가":round(px,2), "거래금액":round(n*px,2),
                     "수수료+슬리피지":trade_cost, "비중괴리":round(drift*100,1)})
    return pd.DataFrame(rows)


def calc_tax(weights: pd.Series, stats: pd.DataFrame, capital: float, tcost: float) -> Dict:
    gp = float((weights * stats["연수익(%)"]/100).sum()*100)
    ac = tcost*12/capital*100
    return {"gross_pct": round(gp,2), "gross_usd": round(capital*gp/100,0),
            "lt_net": round(gp*(1-LONG_TAX),2), "st_net": round(gp*(1-SHORT_TAX),2),
            "annual_cost": round(ac,3), "real_net": round(gp*(1-LONG_TAX)-ac,2)}


# ══════════════════════════════════════════════════════════════════════════
#  SECTION C — 포트폴리오 트래커 (P&L)
# ══════════════════════════════════════════════════════════════════════════

def load_positions() -> Dict:
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, encoding="utf-8") as f: return json.load(f)
    return {}


def save_positions(positions: Dict) -> None:
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)
    print(f"  💾 저장: {POSITIONS_FILE}")


def add_position(pos: Dict, ticker: str, shares: int, avg_cost: float,
                 buy_date: str, sl: Optional[float]=None, tp: Optional[float]=None, note: str="") -> None:
    if ticker in pos:
        old_sh = pos[ticker]["shares"]; old_c = pos[ticker]["avg_cost"]
        new_sh = old_sh + shares
        pos[ticker].update({"shares": new_sh, "avg_cost": round((old_sh*old_c+shares*avg_cost)/new_sh,4)})
        if sl: pos[ticker]["sl"] = sl
        if tp: pos[ticker]["tp"] = tp
        print(f"  ✅ {ticker} 추가매수 → 평균단가 ${pos[ticker]['avg_cost']:.2f} / {new_sh}주")
    else:
        pos[ticker] = {"shares":shares,"avg_cost":avg_cost,"buy_date":buy_date,"sl":sl,"tp":tp,"note":note}
        print(f"  ✅ {ticker} 신규 등록: {shares}주 @ ${avg_cost:.2f}")


def remove_position(pos: Dict, ticker: str, shares: Optional[int]=None) -> None:
    if ticker not in pos: print(f"  ⚠️  {ticker} 없음"); return
    if shares is None or shares >= pos[ticker]["shares"]:
        del pos[ticker]; print(f"  ✅ {ticker} 전량 청산")
    else:
        pos[ticker]["shares"] -= shares
        print(f"  ✅ {ticker} {shares}주 청산 → 잔량 {pos[ticker]['shares']}주")


def fetch_current_prices(tickers: List[str]) -> pd.Series:
    print(f"  📡 현재가 조회 ({len(tickers)}종목, 15분 지연)...")
    prices = {}
    for tk in tickers:
        try:
            prices[tk] = float(yf.Ticker(tk).fast_info.last_price)
        except Exception:
            try:
                prices[tk] = float(yf.download(tk, period="5d", progress=False, auto_adjust=True)["Close"].iloc[-1])
            except Exception:
                prices[tk] = float("nan")
    return pd.Series(prices)


def calc_pnl(positions: Dict, prices: pd.Series) -> pd.DataFrame:
    rows = []
    today = date.today()
    for tk, p in positions.items():
        px = prices.get(tk, float("nan"))
        sh = p["shares"]; cost = p["avg_cost"]; basis = sh*cost
        mval = sh*px if np.isfinite(px) else float("nan")
        pnl  = mval - basis if np.isfinite(px) else float("nan")
        pnl_pct = pnl/basis*100 if basis>0 and np.isfinite(pnl) else float("nan")
        try:
            hold = (today - datetime.strptime(p["buy_date"],"%Y-%m-%d").date()).days
        except Exception:
            hold = 0
        sl = p.get("sl"); tp = p.get("tp")
        sl_alert = bool(sl and np.isfinite(px) and px <= sl/0.95)
        tp_alert = bool(tp and np.isfinite(px) and px >= tp*0.95)
        rows.append({"ticker":tk,"주식수":sh,"매수단가":round(cost,2),
                     "현재가":round(px,2) if np.isfinite(px) else None,
                     "투자원금":round(basis,2),
                     "평가금액":round(mval,2) if np.isfinite(mval) else None,
                     "미실현손익":round(pnl,2) if np.isfinite(pnl) else None,
                     "수익률(%)":round(pnl_pct,2) if np.isfinite(pnl_pct) else None,
                     "보유일수":hold,"세금구분":"장기(20%)" if hold>=365 else "단기(37%)",
                     "SL":round(sl,2) if sl else None,"TP":round(tp,2) if tp else None,
                     "SL경보":sl_alert,"TP경보":tp_alert,"메모":p.get("note","")})
    return pd.DataFrame(rows)


def fetch_period_returns(tickers: List[str]) -> pd.DataFrame:
    all_tk = list(set(tickers+[BENCHMARK]))
    raw = yf.download(all_tk, period="1y", progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":all_tk[0]})
    periods = {"1일":1,"1주":5,"1개월":21,"3개월":63,"1년":252}
    rows = []
    for tk in tickers:
        if tk not in close.columns: continue
        row = {"ticker":tk}
        for label, days in periods.items():
            if len(close) > days:
                r  = (close[tk].iloc[-1]/close[tk].iloc[-days-1]-1)*100
                row[label] = round(float(r),2)
                if BENCHMARK in close.columns:
                    br = (close[BENCHMARK].iloc[-1]/close[BENCHMARK].iloc[-days-1]-1)*100
                    row[f"α({label})"] = round(float(r-br),2)
            else:
                row[label] = None; row[f"α({label})"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION D — 감성 스캐너 (Sentiment)
# ══════════════════════════════════════════════════════════════════════════

def _news_sentiment(ticker: str) -> Dict:
    if not VADER_OK:
        return {"score":0.0,"count":0,"headlines":[],"error":"vaderSentiment 미설치"}
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        return {"score":0.0,"count":0,"headlines":[],"error":str(e)}
    cutoff = datetime.now() - timedelta(days=7)
    ana = SentimentIntensityAnalyzer(); scores = []; headlines = []
    for n in news[:20]:
        try:
            if datetime.fromtimestamp(n.get("providerPublishTime",0)) < cutoff: continue
        except Exception:
            pass
        title = n.get("title","")
        if not title: continue
        vs = ana.polarity_scores(title); scores.append(vs["compound"])
        headlines.append({"title":title,"score":round(vs["compound"],3),
                          "date":datetime.fromtimestamp(n.get("providerPublishTime",0)).strftime("%m/%d")})
    avg = float(np.mean(scores)) if scores else 0.0
    return {"score":round(avg,3),"count":len(scores),"headlines":headlines[:5],"error":None}


def _earnings_surprise(ticker: str) -> Dict:
    try:
        hist = yf.Ticker(ticker).earnings_history
        if hist is None or len(hist)==0: return {"score":0.0,"avg_surp":0,"count":0}
        surps = []
        for _, row in hist.head(4).iterrows():
            a = float(row.get("epsActual",0) or 0); e = float(row.get("epsEstimate",0) or 0)
            if abs(e) > 0.001: surps.append((a-e)/abs(e))
        if not surps: return {"score":0.0,"avg_surp":0,"count":0}
        wts = np.exp(np.linspace(0,1,len(surps)))[::-1]
        wa  = float(np.average(surps, weights=wts/wts.sum()))
        return {"score":round(float(np.clip(wa/0.5,-1,1)),3),
                "avg_surp":round(float(np.mean(surps))*100,1),"count":len(surps)}
    except Exception:
        return {"score":0.0,"avg_surp":0,"count":0}


def _analyst_revision(ticker: str) -> Dict:
    GMAP = {"strong buy":2,"buy":1,"outperform":1,"overweight":1,"accumulate":1,"add":1,
            "hold":0,"neutral":0,"market perform":0,"equal-weight":0,
            "underperform":-1,"underweight":-1,"sell":-1,"strong sell":-2,"reduce":-1}
    try:
        tk = yf.Ticker(ticker); recs = tk.recommendations
        if recs is None or len(recs)==0:
            rs = tk.recommendations_summary
            if rs is not None and len(rs)>0:
                l = rs.iloc[0]
                buy=float(l.get("strongBuy",0)+l.get("buy",0)); sell=float(l.get("sell",0)+l.get("strongSell",0))
                tot=float(sum(l.get(k,0) for k in ["strongBuy","buy","hold","sell","strongSell"]))
                return {"score":round((buy-sell)/max(tot,1),3),"upgrades":int(buy),"downgrades":int(sell)}
            return {"score":0.0,"upgrades":0,"downgrades":0}
        try:
            recent = recs[recs.index >= datetime.now()-timedelta(days=90)]
        except Exception:
            recent = recs.head(10)
        up=0; dn=0
        for _, row in recent.iterrows():
            act = str(row.get("Action","")).lower()
            f = GMAP.get(str(row.get("From Grade","")).lower(),0)
            t = GMAP.get(str(row.get("To Grade","")).lower(),0)
            if "up" in act or "init" in act: up+=1
            elif "down" in act: dn+=1
            elif t>f: up+=1
            elif t<f: dn+=1
        return {"score":round((up-dn)/max(up+dn,1),3),"upgrades":up,"downgrades":dn}
    except Exception:
        return {"score":0.0,"upgrades":0,"downgrades":0}


def _short_interest(ticker: str) -> Dict:
    try:
        info = yf.Ticker(ticker).info; pct = float(info.get("shortPercentOfFloat",0) or 0)
        score = -1.0 if pct>=0.20 else -0.5 if pct>=0.10 else 0.0 if pct>=0.05 else 0.5
        return {"score":round(score,2),"short_pct":round(pct*100,1)}
    except Exception:
        return {"score":0.0,"short_pct":0}


def _insider_activity(ticker: str) -> Dict:
    try:
        df = yf.Ticker(ticker).insider_transactions
        if df is None or len(df)==0: return {"score":0.0,"buy":0,"sell":0}
        cutoff = datetime.now()-timedelta(days=90)
        date_col = next((c for c in ["Start Date","Date","startDate"] if c in df.columns), None)
        if date_col:
            try: df = df[pd.to_datetime(df[date_col]) >= cutoff]
            except Exception: df = df.head(20)
        text_col = next((c for c in ["Transaction","Text","transaction"] if c in df.columns), None)
        buy=0; sell=0
        if text_col:
            for v in df[text_col].astype(str).str.lower():
                if any(k in v for k in ["purchase","buy","acquisition"]): buy+=1
                elif any(k in v for k in ["sale","sell","disposition"]): sell+=1
        tot = buy+sell
        return {"score":round((buy-sell)/max(tot,1),3),"buy":buy,"sell":sell}
    except Exception:
        return {"score":0.0,"buy":0,"sell":0}


def calc_composite(news:Dict,earn:Dict,anal:Dict,short:Dict,insid:Dict) -> Dict:
    parts = [(news["score"],W_NEWS),(earn["score"],W_EARN),(anal["score"],W_ANAL),
             (short["score"],W_SHORT),(insid["score"],W_INSID)]
    num = sum(s*w for s,w in parts); den = sum(w for _,w in parts)
    comp = num/den if den>0 else 0.0
    sig = ("매수 📈","green") if comp>=BUY_THR else ("매도 📉","red") if comp<=SELL_THR else ("중립 ➡️","yellow")
    return {"composite":round(comp,3),"signal":sig[0],"sig_cls":sig[1]}


def scan_sentiment(tickers: List[str]) -> List[Dict]:
    results = []
    for i, tk in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {tk} 감성 분석...")
        news=_news_sentiment(tk); earn=_earnings_surprise(tk); anal=_analyst_revision(tk)
        short=_short_interest(tk); insid=_insider_activity(tk)
        comp=calc_composite(news,earn,anal,short,insid)
        results.append({"ticker":tk,"news":news,"earnings":earn,"analyst":anal,
                        "short":short,"insider":insid,"composite":comp})
        time.sleep(0.3)
    return sorted(results, key=lambda x: x["composite"]["composite"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION E — HTML 리포트 통합 생성
# ══════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0a0b0e;--bg2:#12141a;--panel:#16181f;--panel2:#1c1f28;
  --border:#1f2333;--border2:#252a3a;
  --accent:#5b6fff;--accent-soft:rgba(91,111,255,.12);
  --green:#22c55e;--red:#ef4444;--yellow:#f59e0b;
  --text:#e8eaf0;--text2:#9aa0b8;--text3:#555d7a;
  --mono:'JetBrains Mono',monospace;--r:12px;--r2:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Pretendard','Segoe UI',system-ui,sans-serif;font-size:14px;padding:28px;line-height:1.6}
.wrap{max-width:1360px;margin:0 auto}
h1{color:var(--text);font-size:1.4rem;font-weight:700;text-align:center;margin-bottom:4px;letter-spacing:-.5px}
.sub{text-align:center;color:var(--text3);font-size:.75rem;margin-bottom:6px}
.nav{display:flex;gap:6px;justify-content:center;margin-bottom:28px;flex-wrap:wrap}
.nav a{padding:6px 16px;border-radius:20px;background:var(--panel);border:1px solid var(--border2);color:var(--accent);text-decoration:none;font-size:.78rem;font-weight:600;transition:all .15s}
.nav a:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.cards{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.card{flex:1;min-width:110px;background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:14px;text-align:center}
.card .cl{font-size:.69rem;color:var(--text3);text-transform:uppercase;margin-bottom:5px;letter-spacing:.5px;font-weight:500}
.card .cv{font-size:1.2rem;font-weight:700}
section{margin-bottom:40px;padding-top:12px}
h2{color:var(--text2);font-size:.78rem;margin:0 0 10px;border-bottom:1px solid var(--border);padding-bottom:6px;text-transform:uppercase;letter-spacing:.7px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.8rem;margin-bottom:6px}
th{color:var(--text3);font-size:.7rem;text-transform:uppercase;padding:6px 8px;text-align:right;border-bottom:2px solid var(--border);white-space:nowrap;font-weight:500}
th:first-child{text-align:left}
td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--border);vertical-align:middle;color:var(--text2)}
td:first-child{text-align:left}
tr:hover td{background:var(--panel2)}
.ticker{font-weight:700;color:var(--accent);font-size:.88rem}
.bold{font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}
.purple{color:#a78bfa}.dim{color:var(--text3);font-size:.75rem}
small{font-size:.75rem}
.note{font-size:.75rem;color:var(--text3);margin:4px 0 14px;line-height:1.6}
.rgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
.rcard{background:var(--panel);border:1px solid var(--border);border-radius:var(--r2);padding:12px;text-align:center}
.rcard .rl{font-size:.69rem;color:var(--text3);margin-bottom:4px;text-transform:uppercase;font-weight:500}
.rcard .rv{font-size:1rem;font-weight:700}
.tax-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
.tax-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--r2);padding:12px}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
"""

def _sb(score: float) -> str:
    pct = (score+1)/2*100; c = "#3fb950" if score>=0 else "#f85149"
    return f'<div class="score-bar"><div class="score-fill" style="width:{pct:.0f}%;background:{c}"></div></div>'

def _fs(v: float) -> str:
    c = "green" if v>0.1 else "red" if v<-0.1 else "dim"
    return f'<span class="{c}">{v:+.2f}</span>'


def build_commander_html(
    active:List[str], weights:pd.Series, shares:pd.Series,
    close:pd.DataFrame, high:pd.DataFrame, low:pd.DataFrame,
    capital:float, cash_left:float, strategy_name:str,
    stats:pd.DataFrame, pf_risk:Dict, corr:pd.DataFrame,
    rebal_df:Optional[pd.DataFrame], tax:Dict,
    dollar_vol:pd.Series, trade_cost:float,
) -> str:
    latest = close.iloc[-1]; allocated = capital - cash_left

    # 포지션 행
    pos_rows = ""
    for tk in active:
        w=float(weights.get(tk,0)); sh=int(shares.get(tk,0))
        if sh<=0: continue
        px=float(latest[tk]); atr=calc_atr(high[tk],low[tk],close[tk])
        sl, tp, rr, regime = calc_sl_tp(px, atr)
        sl_pct=(sl/px-1)*100; tp_pct=(tp/px-1)*100
        s=stats.loc[tk]; liq=dollar_vol.get(tk,0)
        liq_s=f"${liq/1e6:.0f}M" if liq>=1e6 else f"${liq/1e3:.0f}K"
        sc="green" if s["Sharpe"]>0.5 else "yellow" if s["Sharpe"]>0 else "red"
        pos_rows += f"""
        <tr>
          <td class="ticker">{tk}</td><td>${px:,.2f}</td>
          <td class="green bold">{sh}주</td>
          <td><span class="purple bold">{w*100:.1f}%</span><br><small class="dim">${sh*px:,.0f}</small></td>
          <td class="{'green' if s['연수익(%)']>0 else 'red'}">{s['연수익(%)']:+.1f}%</td>
          <td class="dim">{s['연변동(%)']:.1f}%</td>
          <td class="{sc} bold">{s['Sharpe']:.2f}</td><td class="red dim">{s['MDD(%)']:.1f}%</td>
          <td class="red bold">${sl:,.2f}<br><small class="red">{sl_pct:.1f}% | {regime} ATR={atr:.2f}</small></td>
          <td class="green bold">${tp:,.2f}<br><small class="green">+{tp_pct:.1f}%</small></td>
          <td class="dim">{rr:.1f}:1<br><small class="dim">EV>0 승률≥{1/(1+rr)*100:.0f}%</small></td>
          <td class="dim">{liq_s}</td>
        </tr>"""

    # 리밸런싱 행
    rebal_rows = ""
    if rebal_df is not None and len(rebal_df)>0:
        for _, r in rebal_df.iterrows():
            a=r["action"]; c="green" if "BUY" in a else "red" if "SELL" in a else "dim"
            rebal_rows += f"""
            <tr>
              <td class="ticker">{r['ticker']}</td><td>{r['현재수량']}주</td><td>{r['목표수량']}주</td>
              <td class="{c} bold">{r['매매수량']:+d}주</td><td class="{c}">{a}</td>
              <td>${r['현재가']:,.2f}</td><td>${r['거래금액']:,.0f}</td>
              <td class="yellow">${r['수수료+슬리피지']:.2f}</td><td class="dim">{r['비중괴리']:.1f}%p</td>
            </tr>"""
    else:
        rebal_rows = '<tr><td colspan="9" style="text-align:center;color:#8b949e;padding:12px">기존 포지션 미입력 — 신규 매수 기준</td></tr>'

    # 상관관계
    corr_hdr  = "".join(f"<th>{t}</th>" for t in corr.columns)
    corr_rows = ""
    for idx, row in corr.iterrows():
        cells = ""
        for col, v in row.items():
            if idx==col: cells += "<td class='dim'>—</td>"
            else:
                c = "red" if v>0.7 else "yellow" if v>0.4 else "green"
                cells += f'<td class="{c}">{v:.2f}</td>'
        corr_rows += f"<tr><td class='ticker'>{idx}</td>{cells}</tr>"

    # 체크리스트
    cl = [("Bid-Ask 스프레드","매수 직전 0.10% 이하 확인"),
          ("어닝 일정","2주 내 실적 발표 없는지 확인 (finviz)"),
          ("시장 국면","Zeus Dashboard Fear & Greed 지수 확인"),
          ("주문 분할","20분 평균 거래량 1% 초과 시 TWAP 분할"),
          ("SL 주문 방식","Stop-Limit으로 슬리피지 제어"),
          ("리밸런싱 임계",f"±{REBAL_THRESHOLD*100:.0f}%p 이내 괴리 → HOLD"),
          ("세금 보유기간",f"1년↑ {LONG_TAX*100:.0f}% vs 1년↓ {SHORT_TAX*100:.0f}%"),
          ("집중도 체크","단일 종목 30% 초과 시 분산 검토")]
    cl_html = "".join(f'<div class="check-item"><span class="chk">☐</span><b>{t}:</b> {d}</div>' for t,d in cl)

    vrc = "red" if pf_risk["var_95"]<-3 else "yellow" if pf_risk["var_95"]<-1.5 else "green"
    crc = "red" if pf_risk["cvar_95"]<-4 else "yellow" if pf_risk["cvar_95"]<-2 else "green"

    return f"""
<section id="commander">
<h2>🚀 포지션 사이징 — {strategy_name}</h2>
<div class="cards">
  <div class="card"><div class="cl">총 예산</div><div class="cv">${capital:,.0f}</div></div>
  <div class="card"><div class="cl">실투자</div><div class="cv green">${allocated:,.0f}</div></div>
  <div class="card"><div class="cl">잔여현금</div><div class="cv">${cash_left:,.0f} ({cash_left/capital*100:.1f}%)</div></div>
  <div class="card"><div class="cl">투자종목</div><div class="cv">{len(active)}개</div></div>
  <div class="card"><div class="cl">거래비용</div><div class="cv yellow">${trade_cost:.2f}</div></div>
  <div class="card"><div class="cl">포트 변동성</div><div class="cv yellow">{pf_risk['port_vol']:.1f}%</div></div>
</div>

<h2>📊 포트폴리오 리스크</h2>
<div class="rgrid">
  <div class="rcard"><div class="rl">포트폴리오 변동성</div><div class="rv yellow">{pf_risk['port_vol']:.1f}%</div></div>
  <div class="rcard"><div class="rl">가중평균 개별변동성</div><div class="rv">{pf_risk['wavg_vol']:.1f}%</div></div>
  <div class="rcard"><div class="rl">분산 효과</div><div class="rv green">−{pf_risk['div_benefit']:.1f}%p</div></div>
  <div class="rcard"><div class="rl">1일 VaR 95%</div><div class="rv {vrc}">{pf_risk['var_95']:.2f}%</div></div>
  <div class="rcard"><div class="rl">1일 CVaR 95%</div><div class="rv {crc}">{pf_risk['cvar_95']:.2f}%</div></div>
</div>

<h2>💰 세후 기대수익</h2>
<div class="tax-grid">
  <div class="tax-card"><div class="tl">세전 기대수익</div><div class="tv">{tax['gross_pct']:+.1f}%<br><small class="dim">${tax['gross_usd']:,.0f}</small></div></div>
  <div class="tax-card"><div class="tl">장기세후 (1년↑, {LONG_TAX*100:.0f}%)</div><div class="tv green">{tax['lt_net']:+.1f}%</div></div>
  <div class="tax-card"><div class="tl">단기세후 (1년↓, {SHORT_TAX*100:.0f}%)</div><div class="tv red">{tax['st_net']:+.1f}%</div></div>
  <div class="tax-card"><div class="tl">연간 거래비용</div><div class="tv yellow">−{tax['annual_cost']:.3f}%</div></div>
  <div class="tax-card"><div class="tl">실질 기대수익 (장기+비용)</div><div class="tv {'green' if tax['real_net']>0 else 'red'}">{tax['real_net']:+.1f}%</div></div>
</div>

<h2>💼 포지션 테이블 (LRM 반올림)</h2>
<table>
  <thead><tr><th>종목</th><th>현재가</th><th>수량</th><th>비중(금액)</th>
  <th>연수익</th><th>연변동</th><th>Sharpe</th><th>MDD</th>
  <th>🛑 손절가</th><th>🎯 익절가</th><th>손익비</th><th>거래대금</th></tr></thead>
  <tbody>{pos_rows}</tbody>
</table>

<h2>🔄 증분 리밸런싱</h2>
<table>
  <thead><tr><th>종목</th><th>현재수량</th><th>목표수량</th><th>매매수량</th>
  <th>액션</th><th>현재가</th><th>거래금액</th><th>수수료+슬리피지</th><th>비중괴리</th></tr></thead>
  <tbody>{rebal_rows}</tbody>
</table>
<p class="note">리밸런싱 임계: ±{REBAL_THRESHOLD*100:.0f}%p 미만 → HOLD | 수수료 {COMM_PCT*100:.3f}% (거래대금 비례) | 슬리피지 유동성 계층별 0.05~0.50% (Kissell 2013)</p>

<h2>🔗 상관관계</h2>
<table><thead><tr><th>종목</th>{corr_hdr}</tr></thead><tbody>{corr_rows}</tbody></table>
<p class="note">🟢 r &lt;0.40 | 🟡 0.40~0.70 | 🔴 ≥0.70 (분산효과 소멸)</p>

<h2>✅ 매매 전 체크리스트</h2>
<div class="check-box">{cl_html}</div>
</section>"""


def build_tracker_html(pnl_df: pd.DataFrame, returns_df: pd.DataFrame) -> str:
    if pnl_df.empty:
        return '<section id="tracker"><h2>📈 P&L 트래커</h2><p class="note">포지션 없음 — 포지션 추가 후 실행하세요</p></section>'

    total_basis = pnl_df["투자원금"].sum()
    total_mval  = pnl_df["평가금액"].dropna().sum()
    total_pnl   = total_mval - total_basis
    total_pct   = total_pnl/total_basis*100 if total_basis>0 else 0
    alerts      = pnl_df[(pnl_df["SL경보"]==True)|(pnl_df["TP경보"]==True)]
    n_win       = (pnl_df["미실현손익"].fillna(0)>0).sum()
    n_loss      = (pnl_df["미실현손익"].fillna(0)<0).sum()
    pc          = "green" if total_pnl>=0 else "red"

    pnl_rows = ""
    for _, r in pnl_df.sort_values("수익률(%)", ascending=False, na_position="last").iterrows():
        pv=r["미실현손익"] or 0; pct=r["수익률(%)"] or 0; c="green" if pv>0 else "red" if pv<0 else "dim"
        rc = "alert-sl" if r["SL경보"] else ("alert-tp" if r["TP경보"] else "")
        sl_b = '<span class="badge badge-sl">🚨SL</span>' if r["SL경보"] else ""
        tp_b = '<span class="badge badge-tp">🎯TP</span>' if r["TP경보"] else ""
        tx_b = f'<span class="badge badge-{"lt" if "장기" in str(r["세금구분"]) else "st"}">{r["세금구분"]}</span>'
        pnl_rows += f"""
        <tr class="{rc}">
          <td class="ticker">{r['ticker']} {sl_b}{tp_b}</td>
          <td>{r['주식수']}주</td><td>${r['매수단가']:,.2f}</td><td>${r['현재가']:,.2f}</td>
          <td class="dim">${r['투자원금']:,.0f}</td><td class="dim">${r['평가금액']:,.0f}</td>
          <td class="{c} bold">${pv:+,.0f}</td><td class="{c} bold">{pct:+.2f}%</td>
          <td class="dim">{r['보유일수']}일</td><td>{tx_b}</td>
          <td class="red dim">{"$"+f"{r['SL']:,.2f}" if r['SL'] else "—"}</td>
          <td class="green dim">{"$"+f"{r['TP']:,.2f}" if r['TP'] else "—"}</td>
        </tr>"""

    ret_rows = ""
    for _, r in returns_df.iterrows():
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return '<td class="dim">—</td>'
            c = "green" if v>0 else "red" if v<0 else "dim"; return f'<td class="{c}">{v:+.2f}%</td>'
        ret_rows += f"""<tr><td class="ticker">{r['ticker']}</td>
          {fmt(r.get('1일'))}{fmt(r.get('α(1일)'))}{fmt(r.get('1주'))}{fmt(r.get('α(1주)'))}
          {fmt(r.get('1개월'))}{fmt(r.get('α(1개월)'))}{fmt(r.get('3개월'))}{fmt(r.get('α(3개월)'))}
          {fmt(r.get('1년'))}{fmt(r.get('α(1년)'))}</tr>"""

    return f"""
<section id="tracker">
<h2>📈 포트폴리오 P&L 트래커</h2>
<div class="cards">
  <div class="card"><div class="cl">총 투자원금</div><div class="cv">${total_basis:,.0f}</div></div>
  <div class="card"><div class="cl">총 평가금액</div><div class="cv {pc}">${total_mval:,.0f}</div></div>
  <div class="card"><div class="cl">미실현 총손익</div><div class="cv {pc}">${total_pnl:+,.0f}</div></div>
  <div class="card"><div class="cl">총 수익률</div><div class="cv {pc}">{total_pct:+.2f}%</div></div>
  <div class="card"><div class="cl">수익/손실 종목</div><div class="cv">{n_win}✅/{n_loss}❌</div></div>
  <div class="card"><div class="cl">경보</div><div class="cv {'red' if len(alerts)>0 else 'green'}">{len(alerts)}개</div></div>
</div>

<table>
  <thead><tr><th>종목</th><th>수량</th><th>매수단가</th><th>현재가</th>
  <th>투자원금</th><th>평가금액</th><th>미실현손익</th><th>수익률</th>
  <th>보유일수</th><th>세금구분</th><th>손절가</th><th>익절가</th></tr></thead>
  <tbody>{pnl_rows}</tbody>
</table>
<p class="note">🚨 SL근접: 손절가 95% 이내 | 🎯 TP근접: 익절가 95% 달성 | 세금: 보유 365일 기준</p>

<h2>📊 기간별 수익률 & SPY 알파</h2>
<table>
  <thead><tr><th>종목</th><th>1일</th><th>α(1일)</th><th>1주</th><th>α(1주)</th>
  <th>1개월</th><th>α(1개월)</th><th>3개월</th><th>α(3개월)</th><th>1년</th><th>α(1년)</th></tr></thead>
  <tbody>{ret_rows}</tbody>
</table>
<p class="note">α = 해당 종목 수익률 − SPY 수익률 (양수 = SPY 아웃퍼폼)</p>
</section>"""


def build_sentiment_html(results: List[Dict]) -> str:
    if not results:
        return '<section id="sentiment"><h2>🔍 감성 스캐너</h2><p class="note">데이터 없음</p></section>'

    buy_c  = sum(1 for r in results if "매수" in r["composite"]["signal"])
    sell_c = sum(1 for r in results if "매도" in r["composite"]["signal"])
    hold_c = sum(1 for r in results if "중립" in r["composite"]["signal"])

    rows = ""
    for r in results:
        tk=r["ticker"]; comp=r["composite"]; sc=comp["composite"]; sig=comp["signal"]
        sig_cls = "sig-buy" if "매수" in sig else "sig-sell" if "매도" in sig else "sig-hold"
        hdl = r["news"].get("headlines",[])
        news_html = "".join(f'<div class="news-line">{"🟢" if h["score"]>0.05 else "🔴" if h["score"]<-0.05 else "⚪"} {h["date"]} {h["title"][:50]}...</div>' for h in hdl[:3]) or '<div class="news-line dim">뉴스 없음</div>'
        surp = r["earnings"].get("avg_surp",0)
        rows += f"""
        <tr>
          <td class="ticker">{tk}</td>
          <td style="text-align:center"><span class="signal {sig_cls}">{sig}</span><br><small>{sc:+.3f}</small>{_sb(sc)}</td>
          <td>{_fs(r['news']['score'])}<br><small class="dim">{r['news']['count']}건</small></td>
          <td>{_fs(r['earnings']['score'])}<br><small class="{'green' if surp>0 else 'red'}">EPS {surp:+.1f}%</small></td>
          <td>{_fs(r['analyst']['score'])}<br><small class="dim">↑{r['analyst'].get('upgrades',0)} ↓{r['analyst'].get('downgrades',0)}</small></td>
          <td>{_fs(r['short']['score'])}<br><small class="dim">공매도 {r['short'].get('short_pct',0):.1f}%</small></td>
          <td>{_fs(r['insider']['score'])}<br><small class="dim">매수{r['insider'].get('buy',0)}/매도{r['insider'].get('sell',0)}</small></td>
          <td style="text-align:left">{news_html}</td>
        </tr>"""

    vader_str = "✅ VADER 활성화" if VADER_OK else "⚠️ vaderSentiment 미설치 (pip install vaderSentiment)"

    return f"""
<section id="sentiment">
<h2>🔍 감성 스캐너 &nbsp;<small class="dim">📈 매수 {buy_c} | ➡️ 중립 {hold_c} | 📉 매도 {sell_c} | {vader_str}</small></h2>
<table>
  <thead><tr><th>종목</th><th>종합신호</th>
  <th>뉴스감성<br>({W_NEWS*100:.0f}%)</th><th>어닝서프라이즈<br>({W_EARN*100:.0f}%)</th>
  <th>애널리스트<br>({W_ANAL*100:.0f}%)</th><th>공매도<br>({W_SHORT*100:.0f}%)</th>
  <th>내부자거래<br>({W_INSID*100:.0f}%)</th><th>최신 뉴스</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="note">점수: -1.0 (매우 부정) ~ +1.0 (매우 긍정) | 매수≥{BUY_THR} | 매도≤{SELL_THR} | ⚠️ 보조 참고 지표 — 단독 매매 결정 금지</p>
</section>"""


def generate_full_html(sections: List[str], now_str: str) -> str:
    nav = """<div class="nav">
      <a href="#commander">🚀 포지션 사이징</a>
      <a href="#tracker">📈 P&L 트래커</a>
      <a href="#sentiment">🔍 감성 스캐너</a>
    </div>"""
    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ZEUS Trading System</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>⚡ ZEUS Trading System</h1>
<div class="sub">{now_str} &nbsp;|&nbsp; ⚠️ 15분 지연 데이터</div>
{nav}
{body}
</div>
</body>
</html>"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f: f.write(html)
    return OUTPUT_FILE


# ══════════════════════════════════════════════════════════════════════════
#  SECTION F — 메뉴 인터페이스
# ══════════════════════════════════════════════════════════════════════════

def run_commander() -> Optional[str]:
    """포지션 사이징 모듈 단독 실행. HTML 섹션 반환."""
    print("\n  ── POSITION COMMANDER ──────────────────")
    raw = input("  티커 (예: NVDA, IONQ, RTX): ").strip()
    if not raw: return None
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    raw_cap = input("  투자 금액 USD (예: 10000): ").strip()
    capital = float(raw_cap) if raw_cap else 10_000.0
    print("  전략: [1]Risk Parity  [2]Half-Kelly  [3]Equal Weight")
    choice = input("  > ").strip()
    print("  기존 보유수량 (예: NVDA=3, TSLA=5 | 없으면 Enter): ", end="")
    raw_pos = input().strip()
    existing: Dict[str, int] = {}
    if raw_pos:
        for item in raw_pos.split(","):
            if "=" in item:
                tk, sh = item.split("=", 1)
                try: existing[tk.strip().upper()] = int(sh.strip())
                except ValueError: pass

    close, high, low, dollar_vol = fetch_ohlcv(tickers)
    passed = liquidity_filter(list(close.columns), dollar_vol)
    if not passed: print("  ❌ 유동성 통과 종목 없음"); return None

    close=close[passed]; high=high[passed]; low=low[passed]
    print("\n  ⚙️  포지션 계산 중...")

    if choice == "2":   weights, sname = strategy_half_kelly(close), f"Half-Kelly (frac={KELLY_FRAC})"
    elif choice == "3": weights, sname = strategy_equal_weight(close), "Equal Weight"
    else:               weights, sname = strategy_risk_parity(close), "Risk Parity (ERC)"

    active = weights[weights >= MIN_WEIGHT].index.tolist()
    prices_now = close[active].iloc[-1]
    shares, cash_left = calc_shares_lrm(weights[active], prices_now, capital)
    stats   = compute_stats(close[active])
    pf_risk = portfolio_risk(weights[active], close[active])
    corr    = close[active].pct_change().dropna().corr().round(2)
    # FIX3: dollar_vol을 calc_rebalance에 전달 → 유동성 계층 슬리피지 적용
    rebal_df = calc_rebalance(shares, existing, prices_now,
                               dollar_vols=dollar_vol) if existing else None

    # Commander → Tracker 자동 연동: SL/TP를 positions.json에 저장
    _pos = load_positions()
    for tk in active:
        sh = int(shares.get(tk, 0))
        if sh <= 0: continue
        px_tk = float(prices_now[tk])
        atr_tk = calc_atr(high[tk], low[tk], close[tk])
        sl_tk, tp_tk, _, _ = calc_sl_tp(px_tk, atr_tk)
        if tk not in _pos:
            _pos[tk] = {'shares': sh, 'avg_cost': round(px_tk, 4),
                        'buy_date': date.today().isoformat(),
                        'sl': sl_tk, 'tp': tp_tk, 'note': f'Commander {sname}'}
        else:
            _pos[tk]['sl'] = sl_tk; _pos[tk]['tp'] = tp_tk
    save_positions(_pos)
    print(f'  💾 SL/TP → {POSITIONS_FILE} 자동 저장 (Tracker 연동)')

    if rebal_df is not None:
        tcost = float(rebal_df[rebal_df["action"].isin(["BUY","SELL"])]["수수료+슬리피지"].sum())
    else:
        # 신규 포지션 초기 매수 비용 추정 (FIX3: dollar_vol 계층 기반)
        tcost = sum(
            _calc_trade_cost(
                int(shares.get(t, 0)),
                float(prices_now.get(t, 0)),
                float(dollar_vol.get(t, 0)) if hasattr(dollar_vol, "get") else 0.0
            )
            for t in active
        )

    tax = calc_tax(weights[active], stats, capital, tcost)

    print(f"\n  {'종목':<8} {'비중':>7} {'수량':>5} {'투자금':>9} {'연수익':>8} {'Sharpe':>7}")
    print(f"  {'─'*50}")
    for tk in active:
        sh=int(shares.get(tk,0)); px=float(prices_now[tk]); s=stats.loc[tk]
        print(f"  {tk:<8} {weights[tk]*100:>6.1f}%  {sh:>4}주  ${sh*px:>8,.0f}  {s['연수익(%)']:>+7.1f}%  {s['Sharpe']:>6.2f}")
    print(f"  실투자: ${capital-cash_left:,.0f} | 잔여: ${cash_left:.0f} | 포트변동성: {pf_risk['port_vol']:.1f}% | VaR: {pf_risk['var_95']:.2f}%")

    return build_commander_html(active, weights, shares, close, high, low,
                                capital, cash_left, sname, stats, pf_risk, corr,
                                rebal_df, tax, dollar_vol, tcost)


def run_tracker() -> Optional[str]:
    """P&L 트래커 모듈 단독 실행."""
    print("\n  ── PORTFOLIO TRACKER ───────────────────")
    positions = load_positions()
    print(f"  포지션 파일: {POSITIONS_FILE} ({len(positions)}개 종목)")

    while True:
        print("\n  [1]리포트  [2]포지션추가  [3]청산  [4]목록  [0]이전메뉴")
        c = input("  > ").strip()
        if c == "0": break
        elif c == "1":
            if not positions: print("  포지션 없음"); continue
            tickers = list(positions.keys())
            prices  = fetch_current_prices(tickers)
            pnl_df  = calc_pnl(positions, prices)
            ret_df  = fetch_period_returns(tickers)

            alerts = pnl_df[(pnl_df["SL경보"]==True)|(pnl_df["TP경보"]==True)]
            for _, a in alerts.iterrows():
                if a["SL경보"]: print(f"  🚨 {a['ticker']}: SL ${a['SL']:,.2f} 근접 (현재 ${a['현재가']:,.2f})")
                if a["TP경보"]: print(f"  🎯 {a['ticker']}: TP ${a['TP']:,.2f} 근접 (현재 ${a['현재가']:,.2f})")

            total_b = pnl_df["투자원금"].sum(); total_m = pnl_df["평가금액"].dropna().sum()
            total_p = total_m - total_b
            print(f"\n  총 P&L: ${total_p:+,.0f} ({total_p/total_b*100:+.2f}%) | 평가금액: ${total_m:,.0f}")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            html = build_tracker_html(pnl_df, ret_df)
            sections = [html]
            out = generate_full_html(sections, now_str)
            print(f"  💾 리포트: {out}")
            try:
                subprocess.Popen(f'start "" "{os.path.abspath(out)}"', shell=True) if sys.platform=="win32" else subprocess.Popen(["xdg-open", os.path.abspath(out)])
            except Exception: pass

        elif c == "2":
            tk  = input("  종목: ").strip().upper()
            sh  = int(input("  주식수: ").strip())
            avg = float(input("  평균 매수단가 ($): ").strip())
            dt  = input("  매수일 (YYYY-MM-DD, Enter=오늘): ").strip() or date.today().isoformat()
            sl_r = input("  손절가 (없으면 Enter): ").strip()
            tp_r = input("  익절가 (없으면 Enter): ").strip()
            note = input("  메모: ").strip()
            add_position(positions, tk, sh, avg, dt, float(sl_r) if sl_r else None, float(tp_r) if tp_r else None, note)
            save_positions(positions)

        elif c == "3":
            print("  현재:", list(positions.keys()))
            tk  = input("  청산 종목: ").strip().upper()
            sh_r = input("  청산 주식수 (전량=Enter): ").strip()
            remove_position(positions, tk, int(sh_r) if sh_r else None)
            save_positions(positions)

        elif c == "4":
            if not positions: print("  포지션 없음"); continue
            print(f"\n  {'종목':<8} {'수량':>5} {'매수단가':>10} {'매수일':>12} {'SL':>9} {'TP':>9}")
            print(f"  {'─'*58}")
            for tk, p in positions.items():
                sl = f"${p['sl']:,.2f}" if p.get("sl") else "—"
                tp = f"${p['tp']:,.2f}" if p.get("tp") else "—"
                print(f"  {tk:<8} {p['shares']:>4}주  ${p['avg_cost']:>9,.2f}  {p['buy_date']:>12}  {sl:>9}  {tp:>9}")

    if positions:
        tickers = list(positions.keys())
        prices  = fetch_current_prices(tickers)
        pnl_df  = calc_pnl(positions, prices)
        ret_df  = fetch_period_returns(tickers)
        return build_tracker_html(pnl_df, ret_df)
    return build_tracker_html(pd.DataFrame(), pd.DataFrame())


def run_sentiment() -> Optional[str]:
    """감성 스캐너 모듈 단독 실행."""
    print("\n  ── SENTIMENT SCANNER ───────────────────")
    if not VADER_OK:
        print("  ⚠️  pip install vaderSentiment (없어도 실행 가능)")
    raw = input("  스캔할 티커 (예: NVDA, TSLA, AMZN): ").strip()
    if not raw: return None
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    print(f"\n  {len(tickers)}개 종목 스캔 중...\n")
    results = scan_sentiment(tickers)

    print(f"\n  {'종목':<8} {'신호':>12} {'종합점수':>9} {'뉴스':>6} {'어닝':>6} {'애널리':>6} {'공매도':>6}")
    print(f"  {'─'*60}")
    for r in results:
        comp = r["composite"]
        print(f"  {r['ticker']:<8} {comp['signal']:>12}  {comp['composite']:>+8.3f}"
              f"  {r['news']['score']:>+5.2f}  {r['earnings']['score']:>+5.2f}"
              f"  {r['analyst']['score']:>+5.2f}  {r['short']['score']:>+5.2f}")

    # Zeus backtest TIER1 팩터 JSON 저장
    factor_data = {r["ticker"]: {"sentiment_score": r["composite"]["composite"],
                                  "signal": r["composite"]["signal"],
                                  "news": r["news"]["score"],
                                  "earnings": r["earnings"]["score"],
                                  "analyst": r["analyst"]["score"]} for r in results}
    with open(FACTORS_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "factors": factor_data}, f, indent=2)
    print(f"\n  📊 팩터 JSON: {FACTORS_FILE} (Zeus backtest v13 TIER1 연동용)")

    return build_sentiment_html(results)


def run_full_report() -> None:
    """3개 모듈 순차 실행 → 통합 HTML."""
    print("\n  ── FULL REPORT (전체 실행) ──────────────")
    sections = []

    cmd_html = run_commander()
    if cmd_html: sections.append(cmd_html)

    trk_html = run_tracker()
    if trk_html: sections.append(trk_html)

    sent_html = run_sentiment()
    if sent_html: sections.append(sent_html)

    if not sections:
        print("  ❌ 실행된 모듈이 없습니다.")
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = generate_full_html(sections, now_str)
    print(f"\n  💾 통합 리포트: {out}")
    try:
        subprocess.Popen(f'start "" "{os.path.abspath(out)}"', shell=True) if sys.platform=="win32" else subprocess.Popen(["xdg-open", os.path.abspath(out)])
    except Exception:
        print(f"  📂 직접 열기: {os.path.abspath(out)}")


# ══════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═"*55)
    print("  ⚡ ZEUS TRADING SYSTEM  v1.0")
    print("═"*55)

    while True:
        print("\n  [1] Position Commander  — 포지션 사이징")
        print("  [2] Portfolio Tracker   — P&L 추적 + 경보")
        print("  [3] Sentiment Scanner   — 뉴스 감성 + 대안데이터")
        print("  [4] Full Report         — 3개 모듈 통합 실행")
        print("  [0] 종료")
        c = input("\n  > ").strip()

        if c == "0":
            break
        elif c == "1":
            html = run_commander()
            if html:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                out = generate_full_html([html], now_str)
                print(f"  💾 리포트: {out}")
                try:
                    subprocess.Popen(f'start "" "{os.path.abspath(out)}"', shell=True) if sys.platform=="win32" else subprocess.Popen(["xdg-open", os.path.abspath(out)])
                except Exception: pass
        elif c == "2":
            run_tracker()
        elif c == "3":
            html = run_sentiment()
            if html:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                out = generate_full_html([html], now_str)
                print(f"  💾 리포트: {out}")
                try:
                    subprocess.Popen(f'start "" "{os.path.abspath(out)}"', shell=True) if sys.platform=="win32" else subprocess.Popen(["xdg-open", os.path.abspath(out)])
                except Exception: pass
        elif c == "4":
            run_full_report()
        else:
            print("  올바른 번호를 입력하세요.")

    print("  👋 종료")


if __name__ == "__main__":
    main()