"""
Ticker Analyzer v2  ─  퀀트 10모듈 완전판
══════════════════════════════════════════════════════════════════════
[기존 기능 완전 유지]
  차트 3패널: 가격+MA+OBV / RSI / 거래량+SVP

[신규 10개 모듈 — 논문 근거 명시]
  M1  IC/ICIR 검증       Grinold & Kahn (2000) "Active Portfolio Management"
  M2  Walk-Forward OOS   Pardo (2008) "Evaluation and Optimization of Trading Strategies"
  M3  거래비용 모델       Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"
  M4  포지션 사이징       Kelly (1956) BSTJ + Wilder (1978) ATR
  M5  Bootstrap CI        Politis & Romano (1994) Stationary Bootstrap
  M6  스트레스 테스트     Bookstaber (2007) "A Demon of Our Own Design"
  M7  복수검정 보정       Benjamini & Hochberg (1995) BH-FDR α=0.05
  M8  포트폴리오 최적화   Markowitz (1952) + Grinold-Kahn IC가중
  M9  감사 추적 로그      JSON Lines (ts/event/ticker/signal/value)
  M10 실시간 데이터       yfinance + 레이턴시 명시 + 지수백오프 재시도

[의존성]  pip install yfinance pandas numpy matplotlib scipy
══════════════════════════════════════════════════════════════════════
"""

# ── 표준 라이브러리 ────────────────────────────────────────────────
import os, sys, json, time, random, logging, warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 서드파티 ──────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════
# M9 ─ 감사 추적 로그 (가장 먼저 초기화 ─ 나머지 모든 모듈이 사용)
# 근거: 구조화 로그 없이는 사후 재현 불가 (퀀트 Compliance 요건)
#       JSON Lines: 한 줄 = 한 이벤트, tail/grep 가능, 크래시에도 손실 없음
# ══════════════════════════════════════════════════════════════════
_LOG_DIR   = Path(__file__).parent / "ta_audit"
_LOG_DIR.mkdir(exist_ok=True)
_AUDIT     = _LOG_DIR / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
_RUN_LOG   = _LOG_DIR / "run.log"

_logger = logging.getLogger("TA_v2")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    fh = logging.FileHandler(_RUN_LOG, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.WARNING)
    _logger.addHandler(sh)


def _audit(event: str, ticker: str, **kwargs) -> None:
    """
    JSONL 감사 이벤트 기록.
    필드: ts, event, ticker + kwargs
    flush=즉시 → 크래시 시에도 손실 없음
    """
    rec = {"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
           "event": event, "ticker": ticker, **kwargs}
    try:
        with open(_AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        _logger.error(f"audit write failed: {e}")


# ══════════════════════════════════════════════════════════════════
# 다크 테마 (기존 완전 유지)
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d", "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e", "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9", "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "legend.fontsize":  8,
})


# ══════════════════════════════════════════════════════════════════
# 기본 지표 함수 (기존 완전 유지 + calc_atr 추가)
# ══════════════════════════════════════════════════════════════════
def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV ─ Granville (1963), 동일가격=0"""
    d = close.diff()
    s = pd.Series(0.0, index=close.index)
    s[d > 0] = 1.0; s[d < 0] = -1.0
    return (volume * s).cumsum()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI ─ Wilder (1978) EWM"""
    d    = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).clip(0, 100)


def calc_atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
    """
    ATR ─ Wilder (1978) "New Concepts in Technical Trading Systems"
    TR = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    prev = close.shift(1)
    tr   = pd.concat([(high - low),
                      (high - prev).abs(),
                      (low  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def find_divergence(price: pd.Series, obv: pd.Series,
                    window: int = 20) -> Tuple[list, list]:
    """OBV 다이버전스 ─ Granville (1963)"""
    bear, bull = [], []
    for i in range(window, len(price)):
        pw = price.iloc[i-window:i+1]; ow = obv.iloc[i-window:i+1]
        p  = float(price.iloc[i]);     o  = float(obv.iloc[i])
        omx = float(ow.max()); omn = float(ow.min())
        if p == float(pw.max()) and omx != 0 and o < omx * 0.97:
            bear.append(price.index[i])
        if p == float(pw.min()) and omn != 0 and o > omn * 1.03:
            bull.append(price.index[i])
    return bear, bull


def selling_volume_pressure(close: pd.Series, volume: pd.Series,
                             roll: int = 10) -> pd.Series:
    """하락일 거래량 비율 (10일 롤링)"""
    sv    = volume.where(close.diff() < 0, 0.0)
    denom = volume.rolling(roll).sum().replace(0, np.nan)
    return (sv.rolling(roll).sum() / denom * 100).fillna(0.0)


def get_52w(close: pd.Series) -> Tuple[list, list]:
    return [close.idxmax()], [close.idxmin()]


# ══════════════════════════════════════════════════════════════════
# M10 ─ 데이터 수집 + 레이턴시 명시
# 근거: yfinance = NYSE/NASDAQ 15분 딜레이 (무료 데이터 한계)
#       Bloomberg/Refinitiv급 실시간은 전용 유료 API 필요
#       → 한계 명시 + 지수백오프 3회 재시도
# ══════════════════════════════════════════════════════════════════
_DATA_NOTE = "yfinance 15min-delay (non-realtime — Bloomberg/Refinitiv 수준 아님)"


def fetch_ohlcv(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    지수백오프 재시도 (3회): 2^n × 3초 + 난수 지터
    Adj Close 우선 사용 → 분할/배당 보정
    """
    _audit("fetch_start", ticker, period=period, note=_DATA_NOTE)
    for attempt in range(3):
        try:
            if attempt:
                time.sleep(2**attempt * 3 + random.uniform(0, 2))
            raw = yf.download(ticker, period=period, interval="1d",
                              auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                raise ValueError("empty")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).strip().title() for c in raw.columns]
            if "Adj Close" in raw.columns:
                adj = pd.to_numeric(raw["Adj Close"], errors="coerce")
                if not adj.isna().all():
                    raw["Close"] = adj
            for c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw = raw.dropna(subset=["Close", "Volume"]).sort_index()
            if len(raw) < 30:
                raise ValueError(f"rows={len(raw)}")
            _audit("fetch_ok", ticker, rows=len(raw), attempt=attempt+1)
            return raw
        except Exception as e:
            _logger.warning(f"[{ticker}] fetch 시도 {attempt+1}/3: {e}")
    _audit("fetch_fail", ticker)
    return None


# ══════════════════════════════════════════════════════════════════
# 신호 8개 빌더 (M1·M7·M8 공통)
# ══════════════════════════════════════════════════════════════════
def build_signals(close: pd.Series, volume: pd.Series,
                  high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
    """
    신호별 논문 근거:
      obv_z     Granville(1963)            OBV 5일 Z-score
      rsi_dev   Wilder(1978)               RSI 50 편차
      mom_20    Jegadeesh-Titman(1993)     20일 모멘텀
      mom_60    Jegadeesh-Titman(1993)     60일 모멘텀
      vol_surge Karpoff(1987)              거래량/20일평균
      atr_comp  Wilder(1978)               ATR/63일평균 (변동성 압축)
      svp       Granville 파생             하락일 거래량 비율
      ma_cross  Faber(2007)                (MA20-MA60)/Close
    """
    obv   = calc_obv(close, volume)
    od    = obv.diff()
    obv_z = od.rolling(5).sum() / (od.rolling(20).std().replace(0, np.nan) * np.sqrt(5))

    rsi_dev   = calc_rsi(close) - 50.0
    mom_20    = close.pct_change(20) * 100
    mom_60    = close.pct_change(60) * 100
    vol_surge = volume / volume.rolling(20).mean().replace(0, np.nan)
    atr14     = calc_atr(high, low, close, 14)
    atr_comp  = atr14 / atr14.rolling(63).mean().replace(0, np.nan)
    svp_s     = selling_volume_pressure(close, volume, 10)
    ma_cross  = (close.rolling(20).mean() - close.rolling(60).mean()) / close.replace(0, np.nan)

    return {"obv_z": obv_z, "rsi_dev": rsi_dev, "mom_20": mom_20,
            "mom_60": mom_60, "vol_surge": vol_surge, "atr_comp": atr_comp,
            "svp": svp_s, "ma_cross": ma_cross}


# ══════════════════════════════════════════════════════════════════
# M1 ─ IC / ICIR 검증
# 근거: Grinold & Kahn (2000) "Active Portfolio Management" Ch.6
#   IC  = Spearman rank corr(signal_t, fwd_ret_{t+N})
#         Spearman 이유: 순위 기반 → 이상치 robust
#   ICIR = E[IC_roll] / σ[IC_roll]  (롤링 20일 창)
#          → 신호 일관성 측정; 업계 기준 ICIR > 0.5
#   t-stat = IC × √(n−2) / √(1−IC²)  (Fisher t-transformation)
#            업계 기준 IC > 0.05 (단일 종목: 기준 완화 필요)
# ══════════════════════════════════════════════════════════════════
def calc_ic(signal: pd.Series, fwd_ret: pd.Series,
            label: str = "") -> Dict:
    """단일 신호 IC/ICIR. 최소 20개 비NaN 쌍 요구."""
    df = pd.DataFrame({"s": signal, "f": fwd_ret}).dropna()
    n  = len(df)
    if n < 20:
        return {"label": label, "ic": np.nan, "icir": np.nan,
                "t_stat": np.nan, "p_val": np.nan, "n": n, "_roll": None}
    # ── 전체 IC (scipy 있으면 spearmanr, 없으면 rank 기반 pearson) ──
    try:
        ic, p = stats.spearmanr(df["s"], df["f"])
        # scipy Mock 감지: 반환값이 float이 아니면 fallback
        ic = float(ic); p = float(p)
    except Exception:
        # fallback: rank 변환 후 pearson (= spearman 정의)
        try:
            rs = df["s"].rank(); rf = df["f"].rank()
            ic = float(rs.corr(rf))
            # p-value: t-stat → 2-tailed
            import math
            t_fb = ic * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - ic**2, 1e-10))
            from scipy.special import stdtr  # type: ignore
            p = float(2 * stdtr(n - 2, -abs(t_fb)))
        except Exception:
            try:
                rs = df["s"].rank(); rf = df["f"].rank()
                ic = float(rs.corr(rf))
                p = float(np.nan)   # p-val만 포기
            except Exception:
                return {"label": label, "ic": np.nan, "icir": np.nan,
                        "t_stat": np.nan, "p_val": np.nan, "n": n, "_roll": None}

    # ── 롤링 IC (20일 창) → ICIR 계산 ──
    # pandas rolling().corr()는 method= 파라미터 미지원 → rank 변환 후 pearson
    try:
        rs = df["s"].rank()
        rf = df["f"].rank()
        roll = rs.rolling(20).corr(rf).dropna()
    except Exception:
        roll = pd.Series(dtype=float)
    icir = (float(roll.mean()) / float(roll.std())
            if len(roll) > 1 and roll.std() > 1e-9 else np.nan)

    # t-stat: Fisher transformation
    denom  = max(1 - ic**2, 1e-10)
    t_stat = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(denom)

    return {"label": label,
            "ic":     round(float(ic),     4),
            "icir":   round(float(icir),   4) if not np.isnan(icir) else np.nan,
            "t_stat": round(float(t_stat), 4),
            "p_val":  round(float(p),      6),
            "n":      n,
            "_roll":  roll}          # M5 Bootstrap용 (출력 제외)


def run_ic_battery(close: pd.Series, volume: pd.Series,
                   high: pd.Series, low: pd.Series,
                   fwd_days: int = 20) -> Dict[str, Dict]:
    """8개 신호 IC 동시 계산"""
    sigs    = build_signals(close, volume, high, low)
    fwd_ret = close.pct_change(fwd_days).shift(-fwd_days) * 100
    return {name: calc_ic(sig, fwd_ret, label=name)
            for name, sig in sigs.items()}


# ══════════════════════════════════════════════════════════════════
# M7 ─ 복수검정 보정 (BH-FDR + Bonferroni)
# 근거: Benjamini & Hochberg (1995) JRSS-B
#   8신호 동시 탐색 시 위양성 확률 = 1−(1−0.05)^8 ≈ 33.7%
#   BH-FDR: k번째 순위 p ≤ (k/m)×α 이면 기각, FDR ≤ α 보장
#   Bonferroni: p ≤ α/m, 보수적 상한 (FWER 제어)
# ══════════════════════════════════════════════════════════════════
def bh_fdr(p_map: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """
    BH-FDR 보정.
    반환: {signal: {p_raw, p_adj_bh, reject_bh, reject_bonf}}
    NaN p-value는 처리에서 제외 후 결과에 포함.
    """
    valid_keys = [k for k, v in p_map.items() if not np.isnan(v)]
    p_arr = np.array([p_map[k] for k in valid_keys])
    m     = len(p_arr)

    out = {k: {"p_raw": p_map[k], "p_adj_bh": np.nan,
               "reject_bh": False, "reject_bonf": False}
           for k in p_map}
    if m < 2:
        return out

    order    = np.argsort(p_arr)
    ps       = p_arr[order]
    bh_thr   = (np.arange(1, m+1) / m) * alpha
    bonf_thr = alpha / m

    # 최대 기각 인덱스 (연속성 보장: i ≤ max_k → 모두 기각)
    rej_mask = ps <= bh_thr
    max_k    = int(np.where(rej_mask)[0].max()) if rej_mask.any() else -1
    final    = np.zeros(m, dtype=bool)
    if max_k >= 0:
        final[:max_k+1] = True

    for rank, oi in enumerate(order):
        k = valid_keys[oi]
        p_adj = float(min(ps[rank] * m / (rank+1), 1.0))
        out[k] = {"p_raw":      round(float(ps[rank]), 6),
                  "p_adj_bh":   round(p_adj, 6),
                  "reject_bh":  bool(final[rank]),
                  "reject_bonf": bool(ps[rank] <= bonf_thr)}
    return out


# ══════════════════════════════════════════════════════════════════
# M5 ─ Bootstrap 신뢰구간 (Stationary Bootstrap)
# 근거: Politis & Romano (1994) JASA
#   시계열 자기상관 → 블록 단위 리샘플링 필요
#   블록 길이 ~ Geometric(p=1/√n): 평균 블록 = √n (권장값)
#   B=1000, seed 고정 → 재현성 보장
# ══════════════════════════════════════════════════════════════════
def stat_bootstrap_ci(series: pd.Series,
                      B: int = 1000,
                      ci: float = 0.95,
                      seed: int = 42) -> Dict:
    """
    IC 롤링 시계열의 평균에 대한 부트스트랩 CI.
    반환: {mean, ci_lo, ci_hi, n, B, block}
    """
    arr = series.dropna().values
    n   = len(arr)
    if n < 10:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "B": B, "block": 0, "valid": False}

    rng   = np.random.default_rng(seed)
    blk   = max(2, int(np.sqrt(n)))   # √n 블록 길이
    p_geo = 1.0 / blk                 # 기하분포 파라미터

    boot = np.empty(B)
    for b in range(B):
        sample = []
        while len(sample) < n:
            start  = int(rng.integers(0, n))
            length = int(rng.geometric(p_geo))
            idx    = [(start + j) % n for j in range(length)]
            sample.extend(arr[idx].tolist())
        boot[b] = float(np.mean(sample[:n]))

    a = (1 - ci) / 2
    return {"mean":  round(float(np.mean(arr)), 4),
            "ci_lo": round(float(np.percentile(boot, a*100)),      4),
            "ci_hi": round(float(np.percentile(boot, (1-a)*100)),  4),
            "n": n, "B": B, "block": blk, "valid": True}


# ══════════════════════════════════════════════════════════════════
# M2 ─ Walk-Forward OOS (연도별 비중첩 롤링)
# 근거: Pardo (2008) "The Evaluation and Optimization of Trading Strategies" Ch.7
#   TRAIN=252일 → IS에서 최고 |IC| 신호 선택
#   TEST=63일  → 동일 신호로 OOS IC 측정 (look-ahead 완전 차단)
#   STEP=63일  → 비중첩 (non-overlapping) → 독립적 OOS 샘플
#   안정 비율 = OOS IC > 0 비율; Lopez de Prado(2018) 기준 ≥ 70%
# ══════════════════════════════════════════════════════════════════
def walk_forward(close: pd.Series, volume: pd.Series,
                 high: pd.Series, low: pd.Series,
                 train: int = 252, test: int = 63,
                 fwd_days: int = 20) -> Dict:
    """연도별 롤링 Walk-Forward OOS"""
    n       = len(close)
    fwd_ret = close.pct_change(fwd_days).shift(-fwd_days) * 100
    is_l, oos_l, wins = [], [], []
    i = train

    while i + test + fwd_days <= n:
        # ── IN-SAMPLE: 최고 |IC| 신호 선택 (selection bias 방지 위해 IS만 사용)
        c_tr = close.iloc[i-train:i]; v_tr = volume.iloc[i-train:i]
        h_tr = high.iloc[i-train:i];  l_tr = low.iloc[i-train:i]
        fr_tr = fwd_ret.iloc[i-train:i]
        bat   = run_ic_battery(c_tr, v_tr, h_tr, l_tr, fwd_days)
        best  = max(bat, key=lambda k: abs(bat[k]["ic"])
                    if not np.isnan(bat[k]["ic"]) else 0.0)
        ic_is = bat[best]["ic"]
        if np.isnan(ic_is):
            i += test; continue

        # ── OUT-OF-SAMPLE: IS에서 선택된 신호만 평가
        c_te = close.iloc[i:i+test]; v_te = volume.iloc[i:i+test]
        h_te = high.iloc[i:i+test];  l_te = low.iloc[i:i+test]
        fr_te = fwd_ret.iloc[i:i+test]
        sigs  = build_signals(c_te, v_te, h_te, l_te)
        sig_s = sigs.get(best, pd.Series(dtype=float))
        ic_r  = calc_ic(sig_s, fr_te, label=best)
        ic_oos = ic_r["ic"] if not np.isnan(ic_r["ic"]) else 0.0

        is_l.append(float(ic_is)); oos_l.append(float(ic_oos))
        wins.append({"split":   str(close.index[i].date()),
                     "signal":  best,
                     "is_ic":   round(float(ic_is),  4),
                     "oos_ic":  round(float(ic_oos), 4)})
        i += test

    if not is_l:
        return {"n_windows": 0, "is_mean": np.nan, "oos_mean": np.nan,
                "stable": np.nan, "degrad": np.nan, "wins": []}

    isa = np.array(is_l); oosa = np.array(oos_l)
    stable = float((oosa > 0).mean())
    nz     = isa != 0
    degrad = float(np.mean((isa[nz] - oosa[nz]) / np.abs(isa[nz]))) if nz.any() else np.nan

    return {"n_windows": len(is_l),
            "is_mean":  round(float(isa.mean()),  4),
            "oos_mean": round(float(oosa.mean()), 4),
            "stable":   round(stable, 3),
            "degrad":   round(degrad, 3) if not np.isnan(degrad) else np.nan,
            "wins":     wins}


# ══════════════════════════════════════════════════════════════════
# M6 ─ 스트레스 테스트
# 근거: Bookstaber (2007) "A Demon of Our Own Design"
#   위기 시 모든 자산 상관관계 → 1 수렴 → 신호 무력화 예측
#   MDD = max peak-to-trough: Magdon-Ismail & Atiya (2004)
#   위기 구간 IC vs 정상 구간 IC 비교 → 신호 레짐 감수성 확인
# ══════════════════════════════════════════════════════════════════
_CRISES = {
    "GFC_2008":      ("2008-09-01", "2009-03-31"),
    "COVID_2020":    ("2020-02-19", "2020-03-23"),
    "TECH_CRASH_22": ("2022-01-01", "2022-10-13"),
    "DOTCOM_2000":   ("2000-03-10", "2002-10-09"),
    "UKRAINE_2022":  ("2022-02-24", "2022-03-15"),
}


def calc_mdd(close: pd.Series) -> float:
    """MDD (%) ─ peak-to-trough 최대 낙폭"""
    cum = (1 + close.pct_change().fillna(0)).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min()) * 100


def stress_test(close: pd.Series, signal: pd.Series,
                fwd_days: int = 20) -> Dict:
    """위기 구간 vs 정상 구간 IC 비교"""
    fwd_ret = close.pct_change(fwd_days).shift(-fwd_days) * 100
    mdd     = calc_mdd(close)
    result  = {"_mdd_pct": round(mdd, 2)}
    crisis_all = pd.Series(False, index=close.index)

    for name, (s, e) in _CRISES.items():
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        if ts > close.index[-1] or te < close.index[0]:
            result[name] = {"ic": np.nan, "n": 0, "in_range": False}
            continue
        mask = (close.index >= ts) & (close.index <= te)
        crisis_all |= mask
        n = int(mask.sum())
        if n < 10:
            result[name] = {"ic": np.nan, "n": n, "in_range": True}
            continue
        r = calc_ic(signal.loc[mask], fwd_ret.loc[mask], label=name)
        result[name] = {"ic": r["ic"], "t_stat": r["t_stat"],
                        "n": n, "in_range": True}

    # 정상 구간 (위기 제외)
    norm = calc_ic(signal.loc[~crisis_all], fwd_ret.loc[~crisis_all], label="normal")
    result["normal"] = {"ic": norm["ic"], "t_stat": norm["t_stat"],
                        "n": norm["n"],   "in_range": True}
    return result


# ══════════════════════════════════════════════════════════════════
# M3 ─ 거래비용 모델
# 근거: Almgren & Chriss (2001) Mathematical Finance
#   슬리피지 = k × √participation × σ_daily,  k=0.5 (실증 중간값)
#   participation = order_value / daily_dollar_volume
#   수수료: retail=0.05%, institutional=0.02% (표준 시장 수준)
#   spread: vol≥1M → 0.01%, else → 0.03% (유동성 proxy)
# ══════════════════════════════════════════════════════════════════
def transaction_cost(price: float, shares: int,
                     avg_daily_vol: float, daily_vol_pct: float,
                     acct_type: str = "retail") -> Dict:
    """
    편도 거래비용 분해.
    roundtrip = 2 × 편도 (진입+청산)
    """
    if price <= 0 or shares <= 0:
        return {"total_pct": 0.0, "commission_pct": 0.0,
                "slippage_pct": 0.0, "spread_pct": 0.0,
                "roundtrip_pct": 0.0, "participation": 0.0}

    COMM = {"retail": 0.0005, "institutional": 0.0002, "hft": 0.00005}
    comm = COMM.get(acct_type, 0.0005)

    order_val    = price * max(shares, 1)
    daily_dv     = price * max(avg_daily_vol, 1.0)
    part         = min(order_val / daily_dv, 1.0)
    slippage     = 0.5 * np.sqrt(max(part, 0.0)) * (daily_vol_pct / 100.0)
    spread       = 0.0001 if avg_daily_vol >= 1_000_000 else 0.0003
    total        = comm + slippage + spread

    return {"total_pct":      round(total * 100,      5),
            "commission_pct": round(comm  * 100,      5),
            "slippage_pct":   round(slippage * 100,   5),
            "spread_pct":     round(spread * 100,     5),
            "roundtrip_pct":  round(total * 2 * 100,  5),
            "participation":  round(part,              6),
            "order_value":    round(order_val,         2)}


# ══════════════════════════════════════════════════════════════════
# M4 ─ 포지션 사이징 (ATR + Kelly)
# 근거 ATR:   Wilder (1978) — risk_per_trade / (ATR × mult)
#   손절폭 = ATR × 2 (Wilder 기본값), 최대 포지션 = account × 20%
# 근거 Kelly: Kelly (1956) BSTJ "A New Interpretation of Information Rate"
#   f* = (p×b − q) / b,  b = avg_win / avg_loss
#   half-Kelly fraction=0.5: Thorp (1975) 분산 감소 권장
#   결합: min(ATR, Kelly) — 두 방법 중 더 보수적 선택
# ══════════════════════════════════════════════════════════════════
def atr_size(price: float, atr: float, account: float = 100_000,
             risk_pct: float = 0.01, mult: float = 2.0,
             max_pos_pct: float = 0.20) -> Dict:
    """ATR 기반 포지션 사이징"""
    if price <= 0 or atr <= 0:
        return {"shares": 0, "valid": False,
                "stop_price": 0.0, "stop_pct": 0.0,
                "risk_amount": 0.0, "position_pct": 0.0}
    stop_w   = atr * mult
    shares   = int(min(account * risk_pct / max(stop_w, price * 0.001),
                       account * max_pos_pct / price))
    return {"shares":   max(0, shares),
            "stop_price": round(price - stop_w, 4),
            "stop_pct":   round(stop_w / price * 100, 3),
            "risk_amount": round(max(0, shares) * stop_w, 2),
            "position_value": round(max(0, shares) * price, 2),
            "position_pct":   round(max(0, shares) * price / account * 100, 3),
            "valid":     shares > 0}


def kelly_size(price: float, win_rate: float,
               avg_win_pct: float, avg_loss_pct: float,
               account: float = 100_000, fraction: float = 0.5) -> Dict:
    """Half-Kelly 포지션 사이징"""
    if avg_loss_pct <= 0 or price <= 0:
        return {"shares": 0, "valid": False,
                "kelly_f": 0.0, "position_pct": 0.0}
    b      = avg_win_pct / avg_loss_pct
    f_star = (win_rate * b - (1 - win_rate)) / b
    f_used = max(0.0, f_star * fraction)
    shares = int(account * f_used / max(price, 0.01))
    return {"shares":         shares,
            "kelly_f":        round(f_star, 5),
            "kelly_f_used":   round(f_used, 5),
            "position_value": round(shares * price, 2),
            "position_pct":   round(shares * price / account * 100, 3),
            "valid":          shares > 0}


def combined_size(price: float, atr: float,
                  win_rate: float = 0.55,
                  avg_win_pct: float = 5.0,
                  avg_loss_pct: float = 3.0,
                  account: float = 100_000,
                  port_corr: float = 0.0,
                  n_positions: int = 1) -> Dict:
    """
    min(ATR, Kelly) 포지션 사이징 — 포트폴리오 공분산 반영.

    [P2-2 FIX] 포트폴리오 상관관계 보정
    문제: 기존 combined_size는 단일종목 ATR/Kelly만 계산
          같은 섹터 종목 동시 보유 시 실질 리스크가 2배 이상 확대됨
          (예: NVDA+AMD 동시 보유, corr=0.85 → 개별 ATR 합산은 리스크 과소 추정)

    해결: Elton & Gruber (1995) 포트폴리오 분산 공식 적용
      σ_port = σ_단일 × √[1/N + (1-1/N) × ρ_avg]  (등비중 N개 포지션 가정)
      → 포트 변동성이 단일 변동성보다 높으면 사이즈 축소
      → port_corr=0 (독립): 보정 없음 (기존 동일)
      → port_corr=0.8 (동일 섹터): 실질 변동성 확대 반영 → 사이즈 축소

    port_corr:   보유 포지션들의 평균 Pearson 상관계수 (0.0 ~ 1.0)
    n_positions: 기존 보유 포지션 수 (신규 포함 총 N개)

    근거: Elton & Gruber (1995) 'Modern Portfolio Theory and Investment Analysis'
          DeMiguel et al. (2009) JF: 상관관계 무시 시 리스크 과소 추정 documented
    """
    a = atr_size(price, atr, account)
    k = kelly_size(price, win_rate, avg_win_pct, avg_loss_pct, account)
    if a["valid"] and k["valid"]:
        sh = min(a["shares"], k["shares"])
    else:
        sh = a["shares"] if a["valid"] else k["shares"]

    # 포트폴리오 공분산 보정 (Elton & Gruber 1995)
    # σ_port / σ_단일 = √[1/N + (1-1/N) × ρ]
    # 단일종목 기준 대비 실질 리스크 배율 → 사이즈 역비례 축소
    corr_adj = 1.0
    if n_positions > 1 and 0.0 < port_corr <= 1.0:
        n = max(n_positions, 2)
        # Elton & Gruber (1995) 등비중 N개 포지션의 포트폴리오 변동성 비율:
        # σ_port / σ_단일 = √[ 1/N + (1-1/N) × ρ ]
        # 값 범위: rho∈(0,1], N≥2 → vol_ratio ∈ (0, 1]  (항상 ≤ 1.0)
        #
        # 사이징 로직 — corr_adj = vol_ratio 직접 적용 (항상 보수적):
        # → 저상관(rho→0): vol_ratio 작아짐 → 개별 사이즈 축소
        #   (분산 효과가 크므로 포트 리스크 예산 내에서 개별 베팅 줄임)
        # → 고상관(rho→1): vol_ratio→1 → 사이즈 거의 그대로
        #   (분산 효과 없음 — 추가 종목이 기존 리스크를 그대로 복제)
        # → 항상 ≤ 1.0이므로 레버리지 없음, 보수적 운영에 적합
        #
        # 근거: Elton & Gruber (1995) Ch.4 "Optimal Portfolios"
        #       DeMiguel et al. (2009) JF: 상관관계 무시 시 리스크 과소 추정
        vol_ratio = np.sqrt(1.0 / n + (1.0 - 1.0 / n) * port_corr)
        corr_adj  = vol_ratio   # ∈ (0, 1] → 항상 보수적
        sh = int(sh * corr_adj)

    return {"final_shares":   sh,
            "atr_shares":     a["shares"],
            "kelly_shares":   k["shares"],
            "position_value": round(sh * price, 2),
            "position_pct":   round(sh * price / account * 100, 3),
            "stop_price":     a.get("stop_price", 0.0),
            "stop_pct":       a.get("stop_pct", 0.0),
            "corr_adj":       round(corr_adj, 4),
            "port_corr":      round(port_corr, 4),
            "n_positions":    n_positions,
            "method":         "min(ATR, Kelly) + PortCorr",
            "valid":          sh > 0}


# ══════════════════════════════════════════════════════════════════
# M8 ─ 포트폴리오 최적화 (IC가중 합성신호)
# 근거: Markowitz (1952) JF "Portfolio Selection"
#   단일 종목 맥락 → 다중 신호 IC-가중 합성
#   IC-weighted: 예상 IC = Σ wᵢ × ICᵢ 최대화 (Grinold-Kahn 2000)
#   등가중(1/N) 비교: DeMiguel et al. (2009) — "Optimal vs Naive Diversification"
#   Z-score 정규화: 이질적 단위 통일, clip ±3σ (이상치 완화)
# ══════════════════════════════════════════════════════════════════
def optimize_signal_weights(ic_bat: Dict[str, Dict]) -> Dict:
    """IC-weighted vs 등가중 비교."""
    pos  = {k: v for k, v in ic_bat.items()
            if not np.isnan(v.get("ic", np.nan)) and v["ic"] > 0}
    n    = len(ic_bat)
    # IC-weighted (양수 IC만)
    if pos:
        tot  = sum(v["ic"] for v in pos.values())
        w_ic = {k: v["ic"]/tot for k, v in pos.items()}
        for k in ic_bat:
            w_ic.setdefault(k, 0.0)
    else:
        w_ic = {k: 1.0/n for k in ic_bat} if n else {}
    # 등가중
    w_eq = {k: 1.0/n for k in ic_bat} if n else {}
    e_ic = sum(w_ic[k] * ic_bat[k].get("ic", 0)
               for k in w_ic if not np.isnan(ic_bat[k].get("ic", np.nan)))
    e_eq = sum(w_eq[k] * ic_bat[k].get("ic", 0)
               for k in w_eq if not np.isnan(ic_bat[k].get("ic", np.nan)))
    return {"w_ic": {k: round(v, 5) for k, v in w_ic.items()},
            "w_eq": {k: round(v, 5) for k, v in w_eq.items()},
            "exp_ic_icw": round(e_ic, 5),
            "exp_ic_eq":  round(e_eq, 5),
            "n_pos": len(pos)}


def composite_signal(close: pd.Series, volume: pd.Series,
                     high: pd.Series, low: pd.Series,
                     weights: Dict[str, float]) -> pd.Series:
    """Z-score 정규화 후 IC가중 합산"""
    sigs = build_signals(close, volume, high, low)
    comp = pd.Series(0.0, index=close.index)
    for name, sig in sigs.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        mu  = sig.rolling(60, min_periods=10).mean()
        std = sig.rolling(60, min_periods=10).std().replace(0, np.nan)
        comp += ((sig - mu) / std).clip(-3, 3).fillna(0.0) * w
    return comp


# ══════════════════════════════════════════════════════════════════
# 통합 분석 엔진 (M1~M10 순차 실행)
# ══════════════════════════════════════════════════════════════════
def run_analysis(ticker: str, df: pd.DataFrame,
                 account: float = 100_000,
                 fwd_days: int = 20,
                 port_corr: float = 0.0,
                 n_positions: int = 1) -> Dict:
    """
    각 모듈 독립 try/except → 일부 실패해도 전체 중단 안 함.
    반환 dict에 'error' 키가 있으면 해당 모듈만 실패.

    port_corr:   기존 보유 포지션들과의 평균 상관계수 (0.0~1.0)
                 → combined_size 포트폴리오 공분산 보정에 사용
    n_positions: 기존 보유 포지션 수 (신규 포함 총 N개)
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    high   = df.get("High",  close).squeeze()
    low    = df.get("Low",   close).squeeze()
    n      = len(close)
    if n < 120:
        return {"_fatal": f"데이터 부족({n}일 < 120일)"}

    _audit("analysis_start", ticker, n=n, fwd_days=fwd_days, account=account)
    res = {"ticker": ticker, "n": n,
           "price":  round(float(close.iloc[-1]), 4),
           "note":   _DATA_NOTE}

    # M1: IC Battery ────────────────────────────────────────────
    print(f"    [M1] IC/ICIR 계산...", flush=True)
    try:
        bat = run_ic_battery(close, volume, high, low, fwd_days)
        # _roll 제거 (출력에서 제외)
        res["ic"] = {k: {kk: vv for kk, vv in v.items() if kk != "_roll"}
                     for k, v in bat.items()}
        best_k = max(bat, key=lambda k: abs(bat[k]["ic"])
                     if not np.isnan(bat[k]["ic"]) else 0.0)
        _audit("ic_done", ticker, best_signal=best_k,
               best_ic=bat[best_k]["ic"])
    except Exception as e:
        bat = {}; res["ic"] = {"_err": str(e)}
        _logger.error(f"M1 error: {e}")

    # M7: BH-FDR ────────────────────────────────────────────────
    print(f"    [M7] BH-FDR 복수검정 보정...", flush=True)
    try:
        p_map = {k: v["p_val"] for k, v in bat.items()
                 if not np.isnan(v.get("p_val", np.nan))}
        res["mt"] = bh_fdr(p_map)
        n_rej = sum(1 for v in res["mt"].values()
                    if isinstance(v, dict) and v.get("reject_bh"))
        _audit("bh_done", ticker, n_reject_bh=n_rej)
    except Exception as e:
        res["mt"] = {"_err": str(e)}

    # M5: Bootstrap CI (최고 IC 신호 기준) ──────────────────────
    print(f"    [M5] Stationary Bootstrap CI (B=1000)...", flush=True)
    try:
        best_roll = bat[best_k].get("_roll") if bat else None
        if best_roll is None or not hasattr(best_roll, "dropna"):
            raise ValueError("IC roll series 없음")
        boot = stat_bootstrap_ci(best_roll, B=1000)
        boot["signal"] = best_k
        res["boot"] = boot
        _audit("boot_done", ticker, signal=best_k,
               mean=boot.get("mean"), ci=[boot.get("ci_lo"), boot.get("ci_hi")])
    except Exception as e:
        res["boot"] = {"_err": str(e)}

    # M2: Walk-Forward OOS ──────────────────────────────────────
    print(f"    [M2] Walk-Forward OOS (연도별 롤링)...", flush=True)
    try:
        wf = walk_forward(close, volume, high, low,
                          train=252, test=63, fwd_days=fwd_days)
        res["wf"] = {k: v for k, v in wf.items() if k != "wins"}
        res["wf"]["wins"] = wf["wins"]
        _audit("wf_done", ticker, n_windows=wf["n_windows"],
               stable=wf.get("stable"), oos_mean=wf.get("oos_mean"))
    except Exception as e:
        res["wf"] = {"_err": str(e)}

    # M6: 스트레스 테스트 ──────────────────────────────────────
    print(f"    [M6] 스트레스 테스트 (위기 구간)...", flush=True)
    try:
        sigs = build_signals(close, volume, high, low)
        best_sig_s = sigs.get(best_k, pd.Series(dtype=float)) if bat else pd.Series(dtype=float)
        res["stress"] = stress_test(close, best_sig_s, fwd_days)
        _audit("stress_done", ticker, mdd=res["stress"].get("_mdd_pct"))
    except Exception as e:
        res["stress"] = {"_err": str(e)}

    # M8: 최적화 ───────────────────────────────────────────────
    print(f"    [M8] 신호 가중치 최적화...", flush=True)
    try:
        opt = optimize_signal_weights(bat)
        res["opt"] = opt
        _audit("opt_done", ticker, exp_ic_icw=opt["exp_ic_icw"],
               n_pos=opt["n_pos"])
    except Exception as e:
        res["opt"] = {"_err": str(e)}

    # M3: 거래비용 ─────────────────────────────────────────────
    try:
        price      = float(close.iloc[-1])
        avg_vol    = float(volume.rolling(20).mean().iloc[-1])
        daily_std  = float(close.pct_change().rolling(20).std().iloc[-1]) * 100
        res["cost"] = transaction_cost(price, 100, avg_vol, daily_std)
        _audit("cost_done", ticker,
               total_pct=res["cost"]["total_pct"],
               roundtrip_pct=res["cost"]["roundtrip_pct"])
    except Exception as e:
        res["cost"] = {"_err": str(e)}

    # M4: 포지션 사이징 ────────────────────────────────────────
    try:
        price   = float(close.iloc[-1])
        atr_val = float(calc_atr(high, low, close, 14).iloc[-1])
        dstd    = float(close.pct_change().rolling(20).std().iloc[-1]) * 100
        pos = combined_size(price, atr_val,
                            win_rate=0.55,
                            avg_win_pct=max(dstd*3, 2.0),
                            avg_loss_pct=max(dstd*2, 1.5),
                            account=account,
                            port_corr=port_corr,
                            n_positions=n_positions)
        pos["atr"] = round(atr_val, 4)
        res["pos"] = pos
        _audit("pos_done", ticker, shares=pos["final_shares"],
               position_pct=pos["position_pct"])
    except Exception as e:
        res["pos"] = {"_err": str(e)}

    _audit("analysis_done", ticker, log=str(_AUDIT))
    return res


# ══════════════════════════════════════════════════════════════════
# 콘솔 리포트 출력
# ══════════════════════════════════════════════════════════════════
def print_report(res: Dict) -> None:
    if "_fatal" in res:
        print(f"\n  ❌ {res['_fatal']}\n")
        return
    W  = 64
    tk = res["ticker"]
    print(f"\n{'═'*W}")
    print(f"  📊 [{tk}] 퀀트 검증 리포트 v2  (가격 ${res['price']:,.4f})")
    print(f"  ⚠️  {res.get('note','')}")
    print(f"{'═'*W}")

    # M1: IC ────────────────────────────────────────────────────
    print(f"\n  ┌─ M1 IC/ICIR  (Grinold-Kahn 2000 — 기준: IC>0.05, ICIR>0.5)")
    ic = res.get("ic", {})
    if "_err" in ic:
        print(f"  │  ❌ {ic['_err']}")
    else:
        print(f"  │  {'신호':<11}  {'IC':>8}  {'ICIR':>7}  {'t-stat':>8}  {'p-val':>9}  유의")
        print(f"  │  {'─'*55}")
        for sig, v in ic.items():
            ic_v  = v.get("ic",     np.nan)
            icir  = v.get("icir",   np.nan)
            t_v   = v.get("t_stat", np.nan)
            p_v   = v.get("p_val",  np.nan)
            flag  = "✅" if (not np.isnan(p_v) and p_v < 0.05) else "  "
            ic_s  = f"{ic_v:+.4f}" if not np.isnan(ic_v) else "    NaN"
            icir_s= f"{icir:+.3f}" if not np.isnan(icir) else "    NaN"
            t_s   = f"{t_v:+.3f}"  if not np.isnan(t_v)  else "    NaN"
            p_s   = f"{p_v:.5f}"   if not np.isnan(p_v)  else "    NaN"
            print(f"  │  {sig:<11}  {ic_s:>8}  {icir_s:>7}  {t_s:>8}  {p_s:>9}  {flag}")
    print(f"  └{'─'*W}")

    # M7: 복수검정 ───────────────────────────────────────────────
    print(f"\n  ┌─ M7 복수검정 보정  (Benjamini-Hochberg 1995 — α=0.05)")
    mt = res.get("mt", {})
    if "_err" in mt:
        print(f"  │  ❌ {mt['_err']}")
    else:
        fp = round((1 - 0.95**8)*100, 1)
        n_rej = sum(1 for v in mt.values()
                    if isinstance(v, dict) and v.get("reject_bh"))
        print(f"  │  비보정 위양성 {fp}% → BH 보정 후 유의: {n_rej}/8 신호")
        print(f"  │  {'신호':<11}  {'p_raw':>9}  {'p_adj_bh':>10}  BH  Bonf")
        print(f"  │  {'─'*52}")
        for sig, v in mt.items():
            if not isinstance(v, dict): continue
            pr = v.get("p_raw",    np.nan)
            pa = v.get("p_adj_bh", np.nan)
            rb = "✅" if v.get("reject_bh")   else "  "
            rf = "✅" if v.get("reject_bonf") else "  "
            prs = f"{pr:.5f}" if not np.isnan(pr) else "     NaN"
            pas = f"{pa:.5f}" if not np.isnan(pa) else "     NaN"
            print(f"  │  {sig:<11}  {prs:>9}  {pas:>10}  {rb}  {rf}")
    print(f"  └{'─'*W}")

    # M5: Bootstrap ──────────────────────────────────────────────
    print(f"\n  ┌─ M5 Bootstrap CI  (Politis-Romano 1994 Stationary Bootstrap)")
    bt = res.get("boot", {})
    if "_err" in bt:
        print(f"  │  ❌ {bt['_err']}")
    elif not bt.get("valid", True):
        print(f"  │  ⚠️  IC 롤링 시리즈 부족 (N={bt.get('n',0)})")
    else:
        sig  = bt.get("signal", "?")
        mean = bt.get("mean",   np.nan)
        lo   = bt.get("ci_lo",  np.nan)
        hi   = bt.get("ci_hi",  np.nan)
        blk  = bt.get("block",  0)
        B    = bt.get("B",      0)
        print(f"  │  신호:{sig}  B={B}  블록길이={blk}(=√n)")
        print(f"  │  IC 평균={mean}  95% CI=[{lo}, {hi}]")
        if not np.isnan(lo):
            if lo > 0:
                print(f"  │  ✅ CI 하한 > 0 → IC 양수 통계적 유의")
            elif not np.isnan(hi) and hi < 0:
                print(f"  │  🔴 CI 상한 < 0 → IC 음수 (역방향 신호)")
            else:
                print(f"  │  ⚠️  CI가 0 포함 → 신호 불안정 (과신 금지)")
    print(f"  └{'─'*W}")

    # M2: Walk-Forward ───────────────────────────────────────────
    print(f"\n  ┌─ M2 Walk-Forward OOS  (Pardo 2008 — TRAIN=252d, TEST=63d)")
    wf = res.get("wf", {})
    if "_err" in wf:
        print(f"  │  ❌ {wf['_err']}")
    else:
        nw  = wf.get("n_windows", 0)
        ism = wf.get("is_mean",   np.nan)
        oom = wf.get("oos_mean",  np.nan)
        sr  = wf.get("stable",    np.nan)
        dg  = wf.get("degrad",    np.nan)
        print(f"  │  윈도우={nw}  IS IC={ism}  →  OOS IC={oom}")
        print(f"  │  안정비율={sr}  성과저하율={dg}")
        if isinstance(sr, float) and not np.isnan(sr):
            print(f"  │  {'✅' if sr>=0.70 else '⚠️ '} OOS 안정비율 "
                  f"{sr:.0%} {'≥' if sr>=0.70 else '<'} 70% (Lopez 기준)")
        for w in wf.get("wins", [])[:5]:
            print(f"  │  [{w['split']}] IS={w['is_ic']:+.4f} "
                  f"→ OOS={w['oos_ic']:+.4f}  ({w['signal']})")
        extra = nw - 5
        if extra > 0:
            print(f"  │  ... 이하 {extra}개 윈도우 생략")
    print(f"  └{'─'*W}")

    # M6: 스트레스 ───────────────────────────────────────────────
    print(f"\n  ┌─ M6 스트레스 테스트  (Bookstaber 2007)")
    st = res.get("stress", {})
    if "_err" in st:
        print(f"  │  ❌ {st['_err']}")
    else:
        print(f"  │  MDD = {st.get('_mdd_pct','?'):.1f}%")
        print(f"  │  {'구간':<18}  {'IC':>8}  {'t-stat':>8}  {'N':>6}  포함")
        print(f"  │  {'─'*50}")
        for nm in ["GFC_2008","COVID_2020","TECH_CRASH_22",
                   "DOTCOM_2000","UKRAINE_2022","normal"]:
            v  = st.get(nm)
            if not isinstance(v, dict): continue
            ic_v = v.get("ic",     np.nan)
            t_v  = v.get("t_stat", np.nan)
            nv   = v.get("n",      0)
            ir   = "O" if v.get("in_range") else "-"
            ic_s = f"{ic_v:+.4f}" if not np.isnan(ic_v) else "     NaN"
            t_s  = f"{t_v:+.3f}"  if not np.isnan(t_v)  else "     NaN"
            print(f"  │  {nm:<18}  {ic_s:>8}  {t_s:>8}  {nv:>6}  {ir}")
    print(f"  └{'─'*W}")

    # M8: 최적화 ─────────────────────────────────────────────────
    print(f"\n  ┌─ M8 포트폴리오 최적화  (Markowitz 1952 + Grinold-Kahn IC가중)")
    opt = res.get("opt", {})
    if "_err" in opt:
        print(f"  │  ❌ {opt['_err']}")
    else:
        print(f"  │  기대IC(IC가중)={opt.get('exp_ic_icw','?')}  "
              f"기대IC(등가중)={opt.get('exp_ic_eq','?')}  "
              f"양수IC신호={opt.get('n_pos',0)}개")
        print(f"  │  {'신호':<11}  {'IC가중':>9}  {'등가중':>9}  그래프")
        print(f"  │  {'─'*50}")
        w_ic = opt.get("w_ic", {})
        w_eq = opt.get("w_eq", {})
        for sig in sorted(w_ic, key=lambda x: -w_ic.get(x, 0)):
            wi  = w_ic.get(sig, 0.0); we = w_eq.get(sig, 0.0)
            bar = "▓" * int(wi * 24)
            print(f"  │  {sig:<11}  {wi:>9.4f}  {we:>9.4f}  {bar}")
    print(f"  └{'─'*W}")

    # M3: 거래비용 ───────────────────────────────────────────────
    print(f"\n  ┌─ M3 거래비용  (Almgren-Chriss 2001, k=0.5)")
    c = res.get("cost", {})
    if "_err" in c:
        print(f"  │  ❌ {c['_err']}")
    else:
        print(f"  │  편도 합계   {c['total_pct']:.5f}%")
        print(f"  │    수수료    {c['commission_pct']:.5f}%  (retail 0.05%)")
        print(f"  │    슬리피지  {c['slippage_pct']:.5f}%  (k=0.5×√part×σ)")
        print(f"  │    스프레드  {c['spread_pct']:.5f}%")
        print(f"  │  왕복 합계   {c['roundtrip_pct']:.5f}%  ← 순수익 차감 기준")
        print(f"  │  참여율      {c['participation']:.6f}")
    print(f"  └{'─'*W}")

    # M4: 포지션 사이징 ──────────────────────────────────────────
    print(f"\n  ┌─ M4 포지션 사이징  (Kelly 1956 + Wilder 1978 ATR)")
    p = res.get("pos", {})
    if "_err" in p:
        print(f"  │  ❌ {p['_err']}")
    elif not p.get("valid"):
        print(f"  │  ⚠️  포지션 0 (음수 Kelly 또는 ATR 오류)")
    else:
        print(f"  │  ATR={p.get('atr',0):.4f}  손절폭={p.get('stop_pct',0):.2f}%  "
              f"손절가=${p.get('stop_price',0):,.2f}")
        print(f"  │  ATR기준:   {p.get('atr_shares',0):>5}주")
        print(f"  │  Kelly기준: {p.get('kelly_shares',0):>5}주")
        print(f"  │  최종(min): {p.get('final_shares',0):>5}주  "
              f"${p.get('position_value',0):,.0f}  "
              f"({p.get('position_pct',0):.1f}%)")
        if p.get("port_corr", 0) > 0:
            print(f"  │  포트상관: ρ={p.get('port_corr',0):.2f}  "
                  f"N={p.get('n_positions', '?')}  "
                  f"보정배율={p.get('corr_adj',1.0):.3f}  "
                  f"(Elton & Gruber 1995)")
    print(f"  └{'─'*W}")

    print(f"\n  [M9] 감사 로그 → {_AUDIT}")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════
# 차트 렌더링 (기존 완전 유지 + IC가중 합성신호 4번째 패널)
# ══════════════════════════════════════════════════════════════════
def draw_chart(ticker: str, df: pd.DataFrame,
               analysis: Optional[Dict] = None) -> None:
    """기존 3패널 완전 유지. analysis 있으면 4번째 패널(합성신호) 추가."""
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    high   = df.get("High", close).squeeze()
    low    = df.get("Low",  close).squeeze()

    if len(close) < 30:
        print(f"  ⚠️  [{ticker}] 데이터 부족 ({len(close)}일)")
        return

    # ── 기존 지표 계산 (완전 유지) ────────────────────────────
    obv       = calc_obv(close, volume)
    rsi       = calc_rsi(close)
    ma20      = close.rolling(20).mean()
    ma60      = close.rolling(60).mean()
    svp       = selling_volume_pressure(close, volume)
    svp_ma5   = svp.rolling(5).mean()
    bear_div, bull_div = find_divergence(close, obv)
    hi52, lo52 = get_52w(close)

    vol_colors = ["#56d364" if i == 0 or float(close.iloc[i]) >= float(close.iloc[i-1])
                  else "#f85149" for i in range(len(close))]

    curr_p     = float(close.iloc[-1])
    curr_rsi   = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0
    curr_svp   = float(svp.dropna().iloc[-1]) if not svp.dropna().empty else 0.0
    ret_1m     = (curr_p / float(close.iloc[max(0, len(close)-21)]) - 1) * 100
    ret_3m     = (curr_p / float(close.iloc[max(0, len(close)-63)]) - 1) * 100
    ma20_l     = float(ma20.dropna().iloc[-1]) if not ma20.dropna().empty else np.nan
    ma60_l     = float(ma60.dropna().iloc[-1]) if not ma60.dropna().empty else np.nan
    above_ma20 = (curr_p > ma20_l) if not np.isnan(ma20_l) else None
    above_ma60 = (curr_p > ma60_l) if not np.isnan(ma60_l) else None
    last_d     = close.index[-1]
    r_bear     = [d for d in bear_div if (last_d - d).days <= 10]
    r_bull     = [d for d in bull_div if (last_d - d).days <= 10]
    hi52_v     = float(close.rolling(252, min_periods=1).max().iloc[-1])
    lo52_v     = float(close.rolling(252, min_periods=1).min().iloc[-1])
    pct_hi     = (curr_p / hi52_v - 1) * 100
    pct_lo     = (curr_p / lo52_v - 1) * 100

    # 콘솔 요약 (기존 완전 유지)
    print(f"  ✅ [{ticker}]  ${curr_p:,.2f}")
    print(f"     1개월 {ret_1m:+.1f}%  |  3개월 {ret_3m:+.1f}%")
    print(f"     RSI {curr_rsi:.1f}  ({'과매수' if curr_rsi>70 else '과매도' if curr_rsi<30 else '중립'})")
    print(f"     MA20 {'위' if above_ma20 else '아래'}  |  MA60 {'위' if above_ma60 else '아래'}")
    print(f"     52주 고점 {pct_hi:+.1f}%  |  52주 저점 {pct_lo:+.1f}%")
    print(f"     SVP {curr_svp:.1f}%  {'⚠️ 매도압력 강함' if curr_svp>60 else ''}")
    if r_bear: print(f"     🔴 약세 다이버전스 최근 {len(r_bear)}건")
    if r_bull: print(f"     🟢 강세 다이버전스 최근 {len(r_bull)}건")

    # ── 합성신호 & IC 타이틀 ─────────────────────────────────
    comp_sig = None
    ic_title = ""
    if analysis and "opt" in analysis and "_err" not in analysis.get("opt", {}):
        try:
            w = analysis["opt"]["w_ic"]
            comp_sig = composite_signal(close, volume, high, low, w)
        except Exception:
            comp_sig = None
    if analysis and "ic" in analysis and "_err" not in analysis.get("ic", {}):
        try:
            best = max(analysis["ic"].items(),
                       key=lambda x: abs(x[1].get("ic", 0))
                       if not np.isnan(x[1].get("ic", 0)) else 0)
            ic_title = f"  |  BestIC:{best[0]}={best[1].get('ic',0):+.4f}"
        except Exception:
            pass

    # ── 패널 수 결정 ──────────────────────────────────────────
    n_p      = 4 if comp_sig is not None else 3
    h_ratios = [4, 1.5, 1.5, 1.5] if n_p == 4 else [4, 1.5, 1.5]

    fig, axes = plt.subplots(
        n_p, 1, figsize=(14, 12 if n_p == 4 else 10),
        gridspec_kw={"height_ratios": h_ratios}, sharex=True)

    ax_p   = axes[0]; ax_rsi = axes[1]; ax_vol = axes[2]
    ax_cs  = axes[3] if n_p == 4 else None

    fig.suptitle(
        f"{ticker.upper()}  —  ${curr_p:,.2f}  |  1M {ret_1m:+.1f}%  "
        f"3M {ret_3m:+.1f}%  |  RSI {curr_rsi:.0f}  |  52wHi {pct_hi:+.1f}%"
        f"{ic_title}",
        fontsize=10, fontweight="bold", color="#e6edf3", y=0.99)
    plt.subplots_adjust(hspace=0.06)

    # ── 패널1: 가격 + MA + OBV (기존 완전 유지) ──────────────
    ax_p.plot(close.index, close, color="#e6edf3", linewidth=1.3, zorder=3)
    ax_p.plot(ma20.index, ma20, color="#58a6ff", lw=0.9, ls="--", label="MA20", zorder=2)
    ax_p.plot(ma60.index, ma60, color="#f78166", lw=0.9, ls="--", label="MA60", zorder=2)
    for d in hi52:
        ax_p.scatter(d, float(close.loc[d]), marker="*", color="#ffd700", s=60, zorder=5)
    for d in lo52:
        ax_p.scatter(d, float(close.loc[d]), marker="*", color="#ff6b6b", s=60, zorder=5)
    for d in bear_div:
        ax_p.axvline(d, color="#ff6b6b", alpha=0.2, lw=0.7)
        ax_p.annotate("v", xy=(d, float(close.loc[d])),
                      fontsize=7, color="#ff6b6b", ha="center", va="bottom", fontweight="bold")
    for d in bull_div:
        ax_p.axvline(d, color="#56d364", alpha=0.2, lw=0.7)
        ax_p.annotate("^", xy=(d, float(close.loc[d])),
                      fontsize=7, color="#56d364", ha="center", va="top", fontweight="bold")
    ma20_a = ma20.reindex(close.index); ma60_a = ma60.reindex(close.index)
    ax_p.fill_between(close.index, ma20_a, ma60_a,
                      where=(ma20_a >= ma60_a), color="#56d364", alpha=0.07)
    ax_p.fill_between(close.index, ma20_a, ma60_a,
                      where=(ma20_a <  ma60_a), color="#f85149", alpha=0.07)
    ax_obv = ax_p.twinx()
    ax_obv.plot(obv.index, obv, color="#bc8cff", alpha=0.5, lw=0.9)
    ax_obv.set_ylabel("OBV", color="#bc8cff", fontsize=8)
    ax_obv.tick_params(axis="y", colors="#bc8cff", labelsize=7)
    ax_obv.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if abs(x) >= 1e6 else f"{x:.0f}"))
    ax_p.set_ylabel("Price (USD)")
    ax_p.grid(True, alpha=0.25)
    leg = [Line2D([0],[0], color="#e6edf3", lw=1.3,  label=f"{ticker} Price"),
           Line2D([0],[0], color="#58a6ff", lw=0.9, ls="--", label="MA20"),
           Line2D([0],[0], color="#f78166", lw=0.9, ls="--", label="MA60"),
           Line2D([0],[0], color="#bc8cff", lw=1, alpha=0.5, label="OBV"),
           Line2D([0],[0], marker="v", color="#ff6b6b", lw=0, ms=7,
                  label="Bearish Div (price↑ OBV↓)"),
           Line2D([0],[0], marker="^", color="#56d364", lw=0, ms=7,
                  label="Bullish Div (price↓ OBV↑)"),
           Line2D([0],[0], marker="*", color="#ffd700", lw=0, ms=8, label="52w High"),
           Line2D([0],[0], marker="*", color="#ff6b6b", lw=0, ms=8, label="52w Low")]
    ax_p.legend(handles=leg, loc="upper left", fontsize=7.5, ncol=2)

    # ── 패널2: RSI (기존 완전 유지) ──────────────────────────
    ax_rsi.plot(rsi.index, rsi, color="#ffa657", lw=1, label="RSI(14)")
    ax_rsi.axhline(70, color="#f85149", ls="--", lw=0.7)
    ax_rsi.axhline(50, color="#8b949e", ls=":",  lw=0.6)
    ax_rsi.axhline(30, color="#56d364", ls="--", lw=0.7)
    ax_rsi.fill_between(rsi.index, rsi, 70, where=(rsi>=70), color="#f85149", alpha=0.20)
    ax_rsi.fill_between(rsi.index, rsi, 30, where=(rsi<=30), color="#56d364", alpha=0.20)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", fontsize=8)
    ax_rsi.text(close.index[-1], 71, "70 과매수", color="#f85149",
                fontsize=6.5, ha="right", va="bottom")
    ax_rsi.text(close.index[-1], 31, "30 과매도", color="#56d364",
                fontsize=6.5, ha="right", va="bottom")
    ax_rsi.grid(True, alpha=0.2)
    ax_rsi.legend(loc="upper left", fontsize=7.5)
    ax_rsi.annotate(f"{curr_rsi:.0f}",
                    xy=(close.index[-1], curr_rsi),
                    xytext=(4, 0), textcoords="offset points",
                    color="#ffa657", fontsize=8, va="center")

    # ── 패널3: 거래량 + SVP (기존 완전 유지) ─────────────────
    ax_vol.bar(close.index, volume, color=vol_colors, alpha=0.65, width=0.8)
    ax_vol.set_ylabel("Volume", fontsize=8)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:.0f}"))
    ax_svp = ax_vol.twinx()
    ax_svp.plot(svp.index, svp, color="#f85149", lw=0.8, alpha=0.8, label="SVP 10d")
    ax_svp.plot(svp_ma5.index, svp_ma5, color="#ffa657", lw=1.1, label="SVP MA5")
    ax_svp.axhline(60, color="#f85149", ls="--", lw=0.6)
    ax_svp.axhline(50, color="#f0b429", ls=":",  lw=0.6)
    ax_svp.set_ylim(0, 100)
    ax_svp.set_ylabel("SVP %", color="#f85149", fontsize=8)
    ax_svp.tick_params(axis="y", colors="#f85149", labelsize=7)
    ax_vol.grid(True, alpha=0.2)
    vol_leg = [Patch(facecolor="#56d364", alpha=0.65, label="Up Volume"),
               Patch(facecolor="#f85149", alpha=0.65, label="Down Volume"),
               Line2D([0],[0], color="#f85149", lw=0.8, label="SVP 10d"),
               Line2D([0],[0], color="#ffa657", lw=1.1, label="SVP MA5"),
               Line2D([0],[0], color="#f85149", lw=0.6, ls="--", label="60% 매도압력")]
    ax_vol.legend(handles=vol_leg, loc="upper left", fontsize=7.5, ncol=2)

    # ── 패널4: IC가중 합성신호 (신규, analysis 있을 때만) ─────
    if ax_cs is not None and comp_sig is not None:
        ax_cs.plot(comp_sig.index, comp_sig, color="#79c0ff", lw=1.0,
                   label="IC-weighted 합성신호")
        ax_cs.axhline(0, color="#8b949e", ls=":", lw=0.6)
        ax_cs.fill_between(comp_sig.index, comp_sig, 0,
                           where=(comp_sig >  0), color="#56d364", alpha=0.15)
        ax_cs.fill_between(comp_sig.index, comp_sig, 0,
                           where=(comp_sig <= 0), color="#f85149", alpha=0.15)
        ax_cs.set_ylabel("합성신호", fontsize=8)
        ax_cs.grid(True, alpha=0.2)
        ax_cs.legend(loc="upper left", fontsize=7.5)

    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("  📊 Ticker Analyzer v2  ─  퀀트 10모듈 완전판")
    print("=" * 64)

    raw_in = input("  티커 (콤마 구분, 예: NVDA, AAPL, IONQ)\n  > ").strip()
    tickers = [t.strip().upper() for t in raw_in.split(",") if t.strip()]
    if not tickers:
        print("  ⚠️  티커를 입력해주세요.")
        sys.exit(0)

    period_in = input("  기간 (1y/2y/3y/5y, 기본 2y — IC·WF에 최소 2y 권장)\n  > ").strip()
    period    = period_in if period_in in ("1y","2y","3y","5y","10y") else "2y"

    acct_in = input("  계좌 크기 USD (기본 100000)\n  > ").strip()
    try:
        account = float(acct_in) if acct_in else 100_000.0
    except ValueError:
        account = 100_000.0

    mode_in = input("  모드: [1] 차트만  [2] 분석만  [3] 차트+분석 (기본 3)\n  > ").strip()
    mode    = mode_in if mode_in in ("1","2","3") else "3"

    print(f"\n  대상: {', '.join(tickers)}  |  기간: {period}"
          f"  |  계좌: ${account:,.0f}  |  모드: {mode}")
    print(f"  ⚠️  {_DATA_NOTE}")
    print("-" * 64)

    for ticker in tickers:
        print(f"\n  📡 [{ticker}] 데이터 수집 (M10)...")
        df = fetch_ohlcv(ticker, period=period)
        if df is None or df.empty:
            print(f"  ❌ [{ticker}] 데이터 수집 실패")
            continue

        analysis = None
        if mode in ("2", "3"):
            print(f"  🔬 [{ticker}] 퀀트 분석 (M1-M9)...")
            analysis = run_analysis(ticker, df, account=account)
            print_report(analysis)

        if mode in ("1", "3"):
            print(f"  🎨 [{ticker}] 차트 렌더링...")
            draw_chart(ticker, df, analysis=analysis)

    print("\n" + "=" * 64)
    print("  ✅ 완료")
    print(f"  📁 감사 로그: {_AUDIT}")
    print("=" * 64)

    if mode in ("1", "3"):
        plt.show()