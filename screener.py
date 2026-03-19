"""
Stock Screener — 급락 후 반등 응축 종목 탐색
==============================================
목표 패턴: 급락 → 저점 → 낙폭의 10~30% 반등 → 이평선 수렴(응축) 중

조건 1: 급락 확인 + 반등 위치 (낙폭의 10~30% 구간)
조건 2: 이평선(20MA / 60MA) 수렴 중 — 간격이 좁아지는 방향
조건 3: SVP — 현재가가 POC(최대 거래량 구간) 근처 (지지 확인)
조건 4: OBV 상승 + Volume Dry-up (거래량 줄면서 매집)

수치 근거:
  낙폭 최소 15%:    O'Neil(1988) CANSLIM — 의미있는 조정 기준
  반등 10~30%:      Wyckoff(1910) Spring/Retest — 2차 상승 직전 응축 구간
                    10% 미만 = 바닥 미확인, 30% 초과 = 이미 반등 소화
  MA 20/60:         Murphy(1999) — 단기/중기 수렴 = 에너지 압축 신호
  수렴 판단 10일:   Elder(1993) Triple Screen — 단기 MA 방향 기준
  VP bins 50/10%:   Steidlmayer(1984) Market Profile POC 정의
  POC band ±5%:     Kroll(1993) — 지지/저항 허용 오차
  OBV raw slope:    Granville(1963) — 누적값 부호반전 방지
  Volume Dry-up 80%: Wyckoff(1910) — 20% 감소 = 매도 물량 소진

실행: python screener.py
의존: pip install yfinance pandas numpy requests
"""

# ── Windows cp949 인코딩 문제 방지 ─────────────────────────────────────
import sys as _sys, io as _io, os as _os
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if _sys.stdout and hasattr(_sys.stdout, 'buffer') and not isinstance(_sys.stdout, _io.TextIOWrapper):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    elif _sys.stdout and hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
try:
    if _sys.stderr and hasattr(_sys.stderr, 'buffer') and not isinstance(_sys.stderr, _io.TextIOWrapper):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')
    elif _sys.stderr and hasattr(_sys.stderr, 'reconfigure'):
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────
import io
import sys
import time
import warnings
from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance 없음: pip install yfinance")


# ──────────────────────────────────────────────
#  파라미터 상수
# ──────────────────────────────────────────────
DOWNLOAD_PERIOD  = "1y"    # ~252 거래일 (52주 고저 계산용)
LOOKBACK_MIN     = 120     # 최소 6개월 데이터 필요

# 조건 1 — 급락 + 반등 위치
DRAWDOWN_MIN     = 0.15    # 최소 낙폭 15% (O'Neil 1988 CANSLIM)
RECOVERY_LO      = 0.05    # 반등 하한: 낙폭의 10% (Wyckoff Spring)
RECOVERY_HI      = 0.30    # 반등 상한: 낙폭의 30% (Wyckoff Retest)

# 조건 2 — 이평선 수렴
MA_SHORT         = 20
MA_LONG          = 60
MA_CONV_LOOKBACK = 10      # 10일 전 gap과 비교 (Elder 1993)

# 조건 3 — SVP
VP_BINS          = 50      # Steidlmayer(1984)
VP_TOP_PCT       = 0.10    # 상위 10% = POC
POC_BAND_PCT     = 0.05    # ±5% 허용 (Kroll 1993)

# 조건 4 — OBV + Dry-up
OBV_WINDOW       = 20      # Granville(1963)
VOLUME_DRY_RATIO = 0.80    # 5일/20일 < 80% (Wyckoff 1910)


# ──────────────────────────────────────────────
#  티커 수집
# ──────────────────────────────────────────────
def get_tickers() -> list:
    """
    S&P 500 + S&P 400 + S&P 600 (위키피디아)
    + Russell 2000 (iShares IWM ETF CSV)
    → 중복 제거 후 약 3000종목
    """
    print("\n  [티커 수집] S&P500 + S&P400 + S&P600 + Russell2000 수집 중...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    tickers = []

    # ── 1. 위키피디아 S&P 500 / 400 / 600
    wiki_urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]
    for url in wiki_urls:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            df  = pd.read_html(io.StringIO(res.text))[0]
            col = next(
                (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
                df.columns[0]
            )
            batch = df[col].tolist()
            tickers += batch
            label = url.split("List_of_")[1].split("_companies")[0].replace("%26P_", "S&P ")
            print(f"  [v] {label}: {len(batch)}종목")
        except Exception as e:
            print(f"  [경고] 위키 크롤링 실패: {e}")

    # ── 2. Russell 2000 (iShares IWM ETF 보유종목 CSV, ~2000종목)
    iwm_url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        res = requests.get(iwm_url, headers=headers, timeout=20)
        # iShares CSV는 앞에 펀드 메타데이터가 가변 행수로 존재
        # → "Ticker"가 포함된 행을 헤더로 동적 탐지
        # BOM·공백·따옴표 등 무관하게 "ticker" 포함 줄을 헤더로 탐지
        lines = res.text.splitlines()
        header_row = next(
            (i for i, l in enumerate(lines)
             if "ticker" in l.strip().lstrip("﻿").lower()[:10]),
            None
        )
        if header_row is None:
            print(f"  [경고] IWM CSV 헤더 탐지 실패. 앞 5줄: {lines[:5]}")
        else:
            iwm_df = pd.read_csv(
                io.StringIO(res.text),
                skiprows=header_row,
                on_bad_lines="skip",   # 컬럼 수 불일치 행 무시
                encoding_errors="ignore",
            )
            # BOM 등으로 오염된 컬럼명도 처리
            iwm_df.columns = [c.strip().lstrip("﻿") for c in iwm_df.columns]
            col = next(
                (c for c in iwm_df.columns if c.lower() == "ticker"),
                None
            )
            if col:
                raw   = iwm_df[col].dropna().tolist()
                # 현금·채권·파생 제거: 알파벳+하이픈만, 1~5자
                valid = [
                    str(t).strip() for t in raw
                    if str(t).strip().replace("-", "").isalpha()
                    and 1 <= len(str(t).strip()) <= 5
                ]
                tickers += valid
                print(f"  [v] Russell 2000 (IWM): {len(valid)}종목")
            else:
                print(f"  [경고] Ticker 컬럼 없음. 컬럼: {list(iwm_df.columns[:6])}")
    except Exception as e:
        print(f"  [경고] Russell 2000 수집 실패: {e}")

    cleaned = sorted(set(str(t).replace(".", "-").strip() for t in tickers if t and str(t).strip()))
    print(f"\n  [완료] 총 {len(cleaned)}종목 (중복 제거 후)\n")
    return cleaned if cleaned else ["AAPL", "MSFT", "NVDA"]


# ──────────────────────────────────────────────
#  OBV
# ──────────────────────────────────────────────
def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Granville(1963) — 상승일 +거래량, 하락일 -거래량 누적"""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# ──────────────────────────────────────────────
#  조건 1: 급락 확인 + 반등 10~30% 위치
# ──────────────────────────────────────────────
def check_drawdown_recovery(df: pd.DataFrame):
    """
    반환: (bool, drawdown_pct, recovery_ratio, 설명)

    Wyckoff(1910) Spring/Retest:
      급락 후 낙폭의 10~30% 구간 = 2차 상승 직전 응축 타이밍.
      - 10% 미만: 아직 저점 탐색 중
      - 10~30%: 반등 초입, 매집 응축 구간 ← 목표
      - 30% 초과: 반등 어느 정도 소화됨

    O'Neil(1988) CANSLIM:
      의미있는 조정 = 15% 이상 낙폭.
      15% 미만 노이즈는 제외.

    peak→trough 탐색: 전체 기간 최고가 이후 최저가.
    """
    close = df["Close"].values.astype(float)
    if len(close) < 60:
        return False, 0.0, 0.0, "데이터부족"

    # 전체 기간 peak → trough (peak 이후 저점)
    peak_idx = int(np.argmax(close))
    trough_idx = peak_idx + int(np.argmin(close[peak_idx:]))

    peak   = close[peak_idx]
    trough = close[trough_idx]
    cur    = close[-1]

    # 낙폭이 peak보다 이후에 발생했는지, 현재가가 trough 이후인지 확인
    if trough_idx <= peak_idx or trough_idx >= len(close) - 1:
        return False, 0.0, 0.0, "패턴없음"

    drawdown = (peak - trough) / peak          # 낙폭 비율
    if drawdown < DRAWDOWN_MIN:
        return False, drawdown, 0.0, f"낙폭부족({drawdown*100:.1f}%)"

    # 현재가가 trough 이후여야 함 (반등 중)
    if cur < trough:
        return False, drawdown, 0.0, "저점미형성"

    recovery_ratio = (cur - trough) / (peak - trough)  # 낙폭 대비 반등 비율

    if RECOVERY_LO <= recovery_ratio <= RECOVERY_HI:
        desc = f"낙폭{drawdown*100:.0f}% 반등{recovery_ratio*100:.0f}%"
        return True, drawdown, recovery_ratio, desc
    elif recovery_ratio < RECOVERY_LO:
        return False, drawdown, recovery_ratio, f"반등부족({recovery_ratio*100:.0f}%)"
    else:
        return False, drawdown, recovery_ratio, f"반등과다({recovery_ratio*100:.0f}%)"


# ──────────────────────────────────────────────
#  조건 2: 이평선 수렴 확인
# ──────────────────────────────────────────────
def check_ma_convergence(df: pd.DataFrame):
    """
    반환: (bool, gap_now_pct, gap_prev_pct, 설명)

    Elder(1993) Triple Screen 수렴 원칙:
      20MA와 60MA의 간격(절댓값)이 10일 전보다 좁아지는 중
      = 두 이평선이 압착되며 에너지 응축.

    방향 조건 없음 (20MA>60MA 불필요):
      반등 10~30% 구간은 아직 데드크로스 상태일 수 있음.
      방향보다 "수렴 중인가"가 핵심.
    """
    close = df["Close"]
    if len(close) < MA_LONG + MA_CONV_LOOKBACK:
        return False, 0.0, 0.0, "데이터부족"

    ma20_ser = close.rolling(MA_SHORT).mean()
    ma60_ser = close.rolling(MA_LONG).mean()

    # 현재 gap (절댓값, 60MA 대비 %)
    ma20_now  = float(ma20_ser.iloc[-1])
    ma60_now  = float(ma60_ser.iloc[-1])
    ma20_prev = float(ma20_ser.iloc[-1 - MA_CONV_LOOKBACK])
    ma60_prev = float(ma60_ser.iloc[-1 - MA_CONV_LOOKBACK])

    if ma60_now <= 0 or ma60_prev <= 0:
        return False, 0.0, 0.0, "MA이상"

    gap_now  = abs(ma20_now  - ma60_now)  / ma60_now
    gap_prev = abs(ma20_prev - ma60_prev) / ma60_prev

    converging = gap_now < gap_prev

    if converging:
        desc = f"수렴중({gap_prev*100:.1f}%→{gap_now*100:.1f}%)"
        return True, gap_now, gap_prev, desc
    else:
        desc = f"발산중({gap_prev*100:.1f}%→{gap_now*100:.1f}%)"
        return False, gap_now, gap_prev, desc


# ──────────────────────────────────────────────
#  조건 3: SVP 두꺼운 매물대 지지
# ──────────────────────────────────────────────
def check_svp_support(df: pd.DataFrame):
    """
    반환: (bool, poc_low, poc_high, 설명)

    Steidlmayer(1984): 50 bins, 거래량 상위 10% = POC.
    OHLC 가중 분배 (O=0.2, H=0.25, L=0.25, C=0.3) —
    Close 단일 포인트 할당 시 HVN 과소평가 방지.

    통과: ① POC 내부  ② POC 상단 ±5%  ③ POC 상단 돌파 직후 (Kroll 1993)
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    opens  = df["Open"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    cur    = close[-1]

    p_lo, p_hi = low.min(), high.max()
    if p_hi - p_lo < 1e-9:
        return False, 0.0, 0.0, "범위없음"

    bins     = np.linspace(p_lo, p_hi, VP_BINS + 1)
    vol_dist = np.zeros(VP_BINS)

    for o, h, l, c, v in zip(opens, high, low, close, volume):
        for px, w in ((o, 0.20), (h, 0.25), (l, 0.25), (c, 0.30)):
            idx = min(max(int(np.searchsorted(bins, px, side="right")) - 1, 0), VP_BINS - 1)
            vol_dist[idx] += v * w

    threshold = np.percentile(vol_dist, (1 - VP_TOP_PCT) * 100)
    poc_bins  = np.where(vol_dist >= threshold)[0]
    if len(poc_bins) == 0:
        return False, 0.0, 0.0, "POC없음"

    poc_low  = float(bins[poc_bins.min()])
    poc_high = float(bins[poc_bins.max() + 1])
    band_lo  = poc_high * (1 - POC_BAND_PCT)
    band_hi  = poc_high * (1 + POC_BAND_PCT)

    if poc_low <= cur <= poc_high:
        return True,  poc_low, poc_high, "POC 내부 지지"
    elif band_lo <= cur < poc_high:
        return True,  poc_low, poc_high, "POC 상단 지지 (±5%내)"
    elif poc_high <= cur <= band_hi:
        return True,  poc_low, poc_high, "POC 상향돌파 직후"
    else:
        dist = (cur - poc_high) / poc_high * 100
        return False, poc_low, poc_high, f"POC 대비 {dist:+.1f}%"


# ──────────────────────────────────────────────
#  조건 4: OBV 상승 + Volume Dry-up
# ──────────────────────────────────────────────
def check_obv_accumulation(df: pd.DataFrame):
    """
    반환: (bool, vol_ratio_pct, 설명)

    반등 10~30% 구간에서는 가격이 오르는 중이 맞으므로
    기존 "가격 횡보" 조건 제거. OBV 방향만 확인.

    OBV raw slope > 0 (Granville 1963):
      반등 중 OBV가 함께 올라가야 진짜 매집.
      정규화 없이 raw slope — 누적값 음수 시 부호반전 방지.

    Volume Dry-up (Wyckoff 1910):
      최근 5일 평균거래량 / 20일 평균거래량 < 80%.
      거래량 줄면서 가격 유지 = 매도 물량 소진, 조용한 매집.
    """
    if len(df) < OBV_WINDOW + 5:
        return False, 0.0, "데이터부족"

    close  = df["Close"]
    volume = df["Volume"]
    obv    = calc_obv(close, volume)

    # OBV raw slope
    obv_w = obv.iloc[-OBV_WINDOW:].values.astype(float)
    x     = np.arange(len(obv_w), dtype=float)
    denom = ((x - x.mean()) ** 2).sum()
    obv_slope = ((x - x.mean()) * (obv_w - obv_w.mean())).sum() / max(denom, 1e-12)

    # Volume Dry-up
    vol_5  = float(volume.iloc[-5:].mean())
    vol_20 = float(volume.iloc[-20:].mean())
    dry_up = (vol_20 > 0) and ((vol_5 / vol_20) < VOLUME_DRY_RATIO)
    vol_ratio = (vol_5 / vol_20 * 100) if vol_20 > 0 else 999.0

    passed = (obv_slope > 0) and dry_up

    if passed:
        desc = f"OBV상승+Dry-up(거래량{vol_ratio:.0f}%)"
    else:
        reasons = []
        if obv_slope <= 0: reasons.append("OBV하락")
        if not dry_up:     reasons.append(f"거래량과다({vol_ratio:.0f}%)")
        desc = "/".join(reasons)

    return passed, vol_ratio, desc


# ──────────────────────────────────────────────
#  스크리닝 루프
# ──────────────────────────────────────────────
def run_screener(tickers: list) -> list:
    total   = len(tickers)
    results = []

    print(f"  총 {total}종목 스크리닝")
    print(f"  조건1: 급락{DRAWDOWN_MIN*100:.0f}%↑ + 반등 {RECOVERY_LO*100:.0f}~{RECOVERY_HI*100:.0f}% 구간")
    print(f"  조건2: 이평선({MA_SHORT}/{MA_LONG}MA) 수렴 중 ({MA_CONV_LOOKBACK}일 비교)")
    print(f"  조건3: SVP POC ±{POC_BAND_PCT*100:.0f}% 이내 or 돌파")
    print(f"  조건4: OBV 상승 + Dry-up <{VOLUME_DRY_RATIO*100:.0f}%")
    print(f"  {'─'*60}")

    # ── 배치 다운로드 (500종목씩 분할)
    # 3000종목 일괄 요청 → 타임아웃/실패 폭증 → 단건 재시도 수백 개 발생
    # 500종목 배치로 나누면: 요청 안정성 ↑, 실패 재시도 ↓, 전체 속도 ↑
    BATCH_SIZE = 500
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_data = {}   # {ticker: df}

    for bi, batch in enumerate(batches):
        print(f"  [배치 {bi+1}/{len(batches)}] {len(batch)}종목 다운로드 중...", end="", flush=True)
        try:
            raw = yf.download(
                batch, period=DOWNLOAD_PERIOD,
                group_by="ticker", threads=True,
                auto_adjust=True, progress=False
            )
        except Exception as e:
            print(f" 실패({e}), 스킵")
            continue

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        ok_cnt = 0
        for tk in batch:
            try:
                if is_multi:
                    if tk not in raw.columns.get_level_values(0):
                        continue
                    df_tk = raw[tk].copy()
                else:
                    df_tk = raw.copy()
                df_tk = df_tk.dropna(subset=["Close", "Volume"])
                if len(df_tk) >= LOOKBACK_MIN:
                    all_data[tk] = df_tk
                    ok_cnt += 1
            except Exception:
                pass
        print(f" {ok_cnt}종목 OK")

    # 배치에서 누락된 종목 단건 재시도 (누락이 적어서 빠름)
    missing = [tk for tk in tickers if tk not in all_data]
    if missing:
        print(f"  [재시도] {len(missing)}종목 단건 재다운로드...", end="", flush=True)
        ok_retry = 0
        for tk in missing:
            try:
                df_r = yf.download(tk, period=DOWNLOAD_PERIOD,
                                   auto_adjust=True, progress=False)
                if isinstance(df_r.columns, pd.MultiIndex):
                    df_r.columns = df_r.columns.get_level_values(0)
                df_r = df_r.dropna(subset=["Close","Volume"])
                if len(df_r) >= LOOKBACK_MIN:
                    all_data[tk] = df_r
                    ok_retry += 1
            except Exception:
                pass
        print(f" {ok_retry}종목 복구")

    retry_cache = all_data   # 이후 코드와 호환
    print(f"  총 {len(all_data)}/{len(tickers)}종목 데이터 확보\n")

    # 기존 코드 호환: data / is_multi / all_levels 변수 유지
    data = None
    is_multi = False
    all_levels = set()

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:3d}/{total}] {ticker:<8}", end="\r", flush=True)

        try:
            # all_data(배치 결과) 에서 직접 가져옴
            if ticker in retry_cache:
                df = retry_cache[ticker].copy()
            else:
                continue

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
            df = df[df["Volume"] > 0].dropna()

            if len(df) < LOOKBACK_MIN:
                continue

            # ── 조건 1: 급락 + 반등 10~30%
            c1, drawdown, recovery, c1_desc = check_drawdown_recovery(df)
            if not c1:
                continue

            # ── 조건 2: 이평선 수렴
            c2, gap_now, gap_prev, c2_desc = check_ma_convergence(df)
            if not c2:
                continue

            # ── 조건 3: SVP
            c3, poc_low, poc_high, svp_desc = check_svp_support(df)
            if not c3:
                continue

            # ── 조건 4: OBV 상승 + Dry-up
            c4, vol_ratio, obv_desc = check_obv_accumulation(df)
            if not c4:
                continue

            # 통과
            cur  = float(df["Close"].iloc[-1])
            ma20 = float(df["Close"].rolling(MA_SHORT).mean().iloc[-1])
            ma60 = float(df["Close"].rolling(MA_LONG).mean().iloc[-1])

            results.append({
                "ticker":    ticker,
                "price":     round(cur, 2),
                "drawdown":  round(drawdown * 100, 1),
                "recovery":  round(recovery * 100, 1),
                "ma20":      round(ma20, 2),
                "ma60":      round(ma60, 2),
                "conv_desc": c2_desc,
                "poc_low":   round(poc_low, 2),
                "poc_high":  round(poc_high, 2),
                "svp_desc":  svp_desc,
                "vol_ratio": round(vol_ratio, 1),
                "obv_desc":  obv_desc,
            })
            print(f"  [OK] {ticker:<8}  ${cur:.2f}  {c1_desc}  {c2_desc}  {svp_desc}")

        except Exception:
            continue

    return results


# ──────────────────────────────────────────────
#  결과 출력
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
#  결과 저장 (zeus_unified 파이프라인 연동)
# ──────────────────────────────────────────────
SCREENER_JSON = "screener_results.json"

def save_results(results: list) -> None:
    """
    스크리닝 결과를 JSON으로 저장.
    zeus_unified.py가 읽어서 Trading Commander에 자동 전달.
    """
    tickers = [r["ticker"] for r in results]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "count": len(results),
        "tickers": tickers,
        "details": results,
    }
    try:
        with open(SCREENER_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  [저장] 결과 저장: {SCREENER_JSON}  ({len(tickers)}종목 → zeus_unified 자동 연동)")
    except Exception as e:
        print(f"\n  [!]  JSON 저장 실패: {e}")


def print_results(results: list) -> None:
    W = 78
    print(f"\n{'='*W}")
    print(f"  [목표] 스크리닝 완료 — {len(results)}종목 통과  "
          f"({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*W}")

    if not results:
        print("  조건을 모두 만족하는 종목이 없습니다.\n")
        return

    # 반등 비율 오름차순 (응축 초입일수록 상단)
    results.sort(key=lambda x: x["recovery"])

    # ── 표1: 급락/반등/수렴
    print(f"\n  {'티커':8}  {'현재가':>9}  {'낙폭':>7}  {'반등률':>7}  {'수렴 상태'}")
    print(f"  {'─'*72}")
    for r in results:
        print(f"  {r['ticker']:<8}  ${r['price']:>8.2f}  "
              f"{r['drawdown']:>6.1f}%  {r['recovery']:>6.1f}%  "
              f"{r['conv_desc']}")

    # ── 표2: SVP + OBV
    print(f"\n  {'티커':8}  {'POC 구간':>22}  {'SVP':16}  OBV/거래량")
    print(f"  {'─'*72}")
    for r in results:
        poc = f"${r['poc_low']:.2f}~${r['poc_high']:.2f}"
        print(f"  {r['ticker']:<8}  {poc:>22}  {r['svp_desc']:<16}  {r['obv_desc']}")

    print(f"\n  ※ 반등률 = (현재가-저점)/(고점-저점)×100")
    print(f"     Dry-up = 최근5일/20일평균 < {VOLUME_DRY_RATIO*100:.0f}%")
    print(f"{'='*W}\n")


# ──────────────────────────────────────────────
#  진입점
# ──────────────────────────────────────────────
def main():
    print("  +======================================+")
    print("  |   Stock Screener  4조건 동시 충족    |")
    print("  +======================================+")
    print(f"  분석기간: {DOWNLOAD_PERIOD}  최소데이터: {LOOKBACK_MIN}거래일 | 목표: 급락 후 응축 종목\n")
    print("  1) S&P500 + S&P400 + S&P600 + Russell2000 (~3000종목)")
    print("  2) 직접 입력\n")

    try:
        choice = input("  선택 (1/2, 기본=1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  종료"); return

    if choice == "2":
        try:
            raw = input("  티커 입력 (쉼표 구분, 예: AAPL,NVDA,MSFT): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  종료"); return
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if not tickers:
            tickers = get_tickers()
    else:
        tickers = get_tickers()

    results = run_screener(tickers)
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()