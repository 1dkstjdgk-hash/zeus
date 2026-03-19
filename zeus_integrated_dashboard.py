#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZEUS INTEGRATED DASHBOARD  v2.1

v2 변경사항:
  1. 파일명 자동 탐색  (zeus_analyzer.py 없으면 Ticker_analyzer_v2.py 자동 폴백)
  2. 백그라운드 Job 큐  (Backtest 90분도 브라우저 타임아웃 없음)
  3. 실시간 로그 폴링  (2초마다 진행상황 업데이트, 단계별 진행바)
  4. 모든 무거운 작업을 별도 스레드로 실행

실행:
    py zeus_integrated_dashboard.py --server        HTTP 서버 (포트 8876) ★권장
    py zeus_integrated_dashboard.py --server --live  + 5분 자동갱신
    py zeus_integrated_dashboard.py                  HTML 파일만 생성
"""
from __future__ import annotations

import builtins, importlib.util, io, json, math, os, re, subprocess
import sys, threading, time, traceback, uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream_name in ("stdout", "stderr"):
    try:
        _stream = getattr(sys, _stream_name, None)
        if _stream and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True   # 메인 종료 시 요청 스레드도 즉시 종료
    allow_reuse_address = True
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).parent.resolve()

OUT_DASH    = HERE / "zeus_integrated_out_dashboard.html"
OUT_AN      = HERE / "zeus_integrated_out_analyzer.html"
OUT_BT      = HERE / "zeus_integrated_out_backtest.html"
OUT_TRD     = HERE / "zeus_integrated_out_trading.html"
UNIFIED     = HERE / "zeus_integrated_dashboard.html"
GH_PAGES_DIR = HERE / "docs"

SIGNALS_FILE    = HERE / "zeus_signals.json"
REGIME_FILE     = HERE / "zeus_regime.json"
POSITIONS_FILE  = HERE / "zeus_positions.json"
SCREENER_JSON   = HERE / "screener_results.json"  # screener.py 결과 파이프라인

PORT        = 8876
REFRESH_SEC = 300
PUBLIC_SITE = True
PUBLIC_BIND_HOST = "0.0.0.0"

# ── 파일명 후보 ──────────────────────────────────────────────────────
MODULE_CANDIDATES: Dict[str, List[str]] = {
    "dashboard": ["zeus_dashboard.py","market_dashboard_v8.py","market_dashboard_v7.py","market_dashboard.py"],
    "analyzer":  ["zeus_analyzer.py","Ticker_analyzer_v2.py","Ticker_analyzer_v3.py","Ticker_analyzer.py","ticker_analyzer.py"],
    "backtest":  ["zeus_backtest.py","zeus_backtest_v12.py","zeus_backtest_v11.py","zeus_backtest_v10.py"],
    "trading":   ["zeus_trading.py","zeus_trading_system.py","zeus_trading_v2.py"],
    "screener":  ["screener.py"],
}

def _find_mod_path(key: str) -> Optional[Path]:
    for name in MODULE_CANDIDATES[key]:
        p = HERE / name
        if p.exists():
            return p
    return None

_MODS: Dict[str, Any]      = {}
_MOD_NAMES: Dict[str, str] = {}

class _MockModule:
    """없는 패키지를 Mock으로 대체 — 임포트 에러 방지"""
    def __init__(self, name=""):
        self.__name__ = name
    def __getattr__(self, k):
        return _MockModule(k)
    def __call__(self, *a, **kw):
        return _MockModule()
    def __iter__(self): return iter([])
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def update(self, *a, **kw): pass
    def rcParams(self): return {}
    rcParams = {}

def _inject_mocks(missing_pkgs: list):
    """누락 패키지를 sys.modules에 Mock으로 등록"""
    for pkg in missing_pkgs:
        root = pkg.split(".")[0]
        if root not in sys.modules:
            sys.modules[root] = _MockModule(root)
        if pkg not in sys.modules:
            sys.modules[pkg] = _MockModule(pkg)

def _load_mod(key: str, path: Path) -> Optional[Any]:
    buf = io.StringIO()
    # 1차 시도: 정상 로드
    try:
        spec = importlib.util.spec_from_file_location(f"_zeus_{key}", str(path))
        mod  = importlib.util.module_from_spec(spec)
        with redirect_stdout(buf), redirect_stderr(buf):
            spec.loader.exec_module(mod)
        _MOD_NAMES[key] = path.name
        return mod
    except ModuleNotFoundError as e:
        missing = e.name or str(e).replace("No module named ", "").strip("'")
        # 2차 시도: 누락 패키지 Mock 주입 후 재시도
        # matplotlib, scipy 등 시각화 라이브러리는 Mock으로 대체 가능
        MOCKABLE = {
            "matplotlib", "matplotlib.pyplot", "matplotlib.dates",
            "matplotlib.lines", "matplotlib.patches", "matplotlib.ticker",
            "matplotlib.gridspec", "matplotlib.colors", "matplotlib.cm",
            "scipy", "scipy.stats", "scipy.optimize", "scipy.signal",
            "seaborn", "plotly", "plotly.graph_objects", "plotly.express",
            "sklearn", "sklearn.preprocessing", "sklearn.linear_model",
            "ta", "ta.momentum", "ta.trend", "ta.volatility", "ta.volume",
        }
        root = missing.split(".")[0] if missing else ""
        if root in {m.split(".")[0] for m in MOCKABLE}:
            # mock 주입할 서브모듈 목록 (파일 헤더 파싱)
            to_mock = [pkg for pkg in MOCKABLE if pkg.startswith(root)]
            _inject_mocks(to_mock)
            print(f"  ⚠️  [{key}] '{missing}' 없음 → Mock 주입 후 재시도…")
            try:
                spec2 = importlib.util.spec_from_file_location(f"_zeus_{key}", str(path))
                mod2  = importlib.util.module_from_spec(spec2)
                # ── 핵심: __file__을 HERE 기준으로 변조 ──────────────────
                # Path(__file__).parent 기반 경로 (ta_audit 등)가
                # 읽기전용 디렉토리가 아닌 HERE를 가리키게 함
                mod2.__file__ = str(HERE / path.name)
                # ta_audit 폴더 미리 생성 (logging.FileHandler 대비)
                (HERE / "ta_audit").mkdir(exist_ok=True)
                buf2  = io.StringIO()
                with redirect_stdout(buf2), redirect_stderr(buf2):
                    spec2.loader.exec_module(mod2)
                _MOD_NAMES[key] = path.name
                print(f"  ✅ [{key}] Mock 주입 성공 (차트 기능 비활성)")
                return mod2
            except Exception as e2:
                print(f"  ❌ [{key}] {path.name}  →  재시도 실패: {e2}")
                return None
        print(f"  ❌ [{key}] {path.name}  →  ModuleNotFoundError: {missing}")
        print(f"       설치: pip install {missing}")
        return None
    except Exception as e:
        print(f"  ❌ [{key}] {path.name}  →  {type(e).__name__}: {e}")
        return None

def load_all() -> None:
    for key in MODULE_CANDIDATES:
        p = _find_mod_path(key)
        if p is None:
            tried = ", ".join(MODULE_CANDIDATES[key][:3])
            print(f"  ❌ [{key}] 파일 없음  (탐색: {tried} …)")
            _MODS[key] = None
        else:
            mod = _load_mod(key, p)
            _MODS[key] = mod
            if mod: print(f"  ✅ [{key}] {p.name}")

# ── 백그라운드 Job ───────────────────────────────────────────────────
class _Job:
    def __init__(self, jid: str):
        self.id = jid
        self.lines: List[str] = []
        self.done   = False
        self.result = {}
        self._lock  = threading.Lock()
        self.started = datetime.now().isoformat()

    def append(self, text: str):
        with self._lock:
            for line in text.splitlines():
                stripped = line.rstrip()
                if stripped:
                    self.lines.append(stripped)

    def finish(self, result: dict):
        with self._lock:
            self.result = result
            self.done   = True

    def snapshot(self) -> dict:
        with self._lock:
            return {"id": self.id, "done": self.done,
                    "lines": list(self.lines[-300:]),
                    "result": self.result if self.done else {},
                    "started": self.started}

_JOBS: Dict[str, _Job] = {}
_JOB_LOCK = threading.Lock()
_ANALYZER_RATE_LIMIT: Dict[str, float] = {}
_ANALYZER_WINDOW_SEC = 300

def _new_job() -> _Job:
    jid = uuid.uuid4().hex[:8]
    job = _Job(jid)
    with _JOB_LOCK:
        _JOBS[jid] = job
        if len(_JOBS) > 30:
            oldest = sorted(_JOBS)[0]
            del _JOBS[oldest]
    return job

class _TeeStream(io.StringIO):
    """redirect_stdout 안에서도 sys.__stdout__(실제 터미널)에 동시 출력."""
    def write(self, s):
        try:
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
        except Exception:
            pass
        return super().write(s)

def _run(fn, *a, **kw) -> Tuple[Any, str]:
    buf_o, buf_e = _TeeStream(), _TeeStream()
    res = None
    try:
        with redirect_stdout(buf_o), redirect_stderr(buf_e):
            res = fn(*a, **kw)
    except SystemExit: pass
    except Exception as e:
        buf_o.write(f"\n❌ {e}\n{traceback.format_exc()}")
    return res, buf_o.getvalue() + buf_e.getvalue()

def _run_bg(job: _Job, fn, *a, **kw) -> Any:
    """백그라운드 스레드 전용 실행.
    stdout/stderr 를 캡처하면서 sys.__stdout__ 으로 동시 tee 출력."""
    buf_o, buf_e = _TeeStream(), _TeeStream()
    res = None
    try:
        with redirect_stdout(buf_o), redirect_stderr(buf_e):
            res = fn(*a, **kw)
    except SystemExit:
        pass
    except Exception as e:
        buf_o.write(f"\n❌ 오류: {e}\n{traceback.format_exc()}")
    combined = buf_o.getvalue() + buf_e.getvalue()
    if combined.strip():
        job.append(combined)
    return res

class _FakeInput:
    def __init__(self, answers): self._it = iter(answers); self._orig = builtins.input
    def __enter__(self): builtins.input = lambda *_: next(self._it, ""); return self
    def __exit__(self, *_): builtins.input = self._orig

# ── 파이프라인 파일 ──────────────────────────────────────────────────
def _save_signals(tickers: List[str], bucket_stats: dict = None) -> None:
    SIGNALS_FILE.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "top_tickers":  tickers,
        "bucket_stats": bucket_stats or {},
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def _load_signals() -> dict:
    return json.loads(SIGNALS_FILE.read_text(encoding="utf-8")) \
        if SIGNALS_FILE.exists() else {}

def _load_screener() -> dict:
    """screener.py가 저장한 screener_results.json 로드."""
    return json.loads(SCREENER_JSON.read_text(encoding="utf-8")) \
        if SCREENER_JSON.exists() else {}

def _do_screener(job: Optional["_Job"], tickers_input: str = "") -> dict:
    """
    screener 모듈을 직접 호출 (subprocess 제거 → input() 블로킹 없음).

    tickers_input:
      ""        → 전체 S&P500+400+600+Russell2000 (~3000종목)
      "AAPL,..."→ 직접 입력 종목만 스크리닝
    """
    sc_path = _find_mod_path("screener")
    if not sc_path:
        msg = "screener.py 없음"
        if job: job.append(f"❌ {msg}")
        return {"ok": False, "log": msg}

    # ── screener 모듈 동적 로드 ──────────────────────────────
    if job: job.append("📦 screener 모듈 로딩…")
    sc_mod = _load_mod("screener", sc_path)
    if sc_mod is None:
        msg = "screener 모듈 로드 실패"
        if job: job.append(f"❌ {msg}")
        return {"ok": False, "log": msg}

    # ── 티커 결정 ──────────────────────────────────────────
    raw = tickers_input.strip()
    if raw:
        tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        if job: job.append(f"🔍 직접 입력 {len(tickers)}종목 스크리닝 시작…")
    else:
        if job: job.append("🔍 전체 종목 수집 중… (S&P500+400+600+Russell2000, 약 3~5분)")
        tickers_res, log_out = _run(sc_mod.get_tickers)
        if log_out.strip() and job:
            for ln in log_out.splitlines():
                if ln.strip(): job.append(ln.rstrip())
        tickers = tickers_res or []
        if not tickers:
            msg = "티커 수집 실패 (네트워크 확인)"
            if job: job.append(f"❌ {msg}")
            return {"ok": False, "log": msg}
        if job: job.append(f"✅ {len(tickers)}종목 수집 완료 — 스크리닝 시작…")

    # ── 스크리닝 실행 ───────────────────────────────────────
    results_res, log_out2 = _run(sc_mod.run_screener, tickers)
    if log_out2.strip() and job:
        for ln in log_out2.splitlines():
            if ln.strip(): job.append(ln.rstrip())
    results = results_res or []

    # ── 저장 ────────────────────────────────────────────────
    try:
        sc_mod.save_results(results)
    except Exception as e:
        if job: job.append(f"⚠️ JSON 저장 오류: {e}")

    sc_data = _load_screener()
    out_tickers = sc_data.get("tickers", [r.get("ticker","") for r in results])
    count = len(out_tickers)

    if not out_tickers:
        msg = "통과 종목 없음 (조건 강화 또는 데이터 부족)"
        if job: job.append(f"⚠️  {msg}")
        return {"ok": True, "tickers": [], "count": 0, "log": msg}

    # signals.json 저장 → Trading Commander 자동 연동
    _save_signals(out_tickers, bucket_stats={"source": "screener",
                                              "generated_at": sc_data.get("generated_at","")})
    if job:
        job.append(f"✅ 스크리너 완료 — {count}종목 통과")
        job.append(f"   {', '.join(out_tickers[:20])}{'...' if count>20 else ''}")
        job.append(f"💾 zeus_signals.json 저장 → Trading 탭 자동 연동")

    return {"ok": True, "tickers": out_tickers, "count": count,
            "details": sc_data.get("details", results)}

def api_screener_bg(tickers_input: str = "") -> str:
    job = _new_job()
    def _w(): job.finish(_do_screener(job, tickers_input))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def _save_regime(regime: str, fg: float) -> None:
    max_w = 0.15 if fg >= 80 else 0.30 if fg >= 40 else 0.20 if fg >= 20 else 0.15
    REGIME_FILE.write_text(json.dumps({
        "regime": regime, "fg_score": fg,
        "max_weight_override": max_w,
        "updated_at": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_regime() -> dict:
    return json.loads(REGIME_FILE.read_text(encoding="utf-8")) \
        if REGIME_FILE.exists() else {}

def _load_positions() -> dict:
    mod = _MODS.get("trading")
    try:
        if mod and hasattr(mod, "load_positions"): return mod.load_positions()
    except Exception: pass
    return json.loads(POSITIONS_FILE.read_text(encoding="utf-8")) \
        if POSITIONS_FILE.exists() else {}

def _client_ip(handler: BaseHTTPRequestHandler) -> str:
    forwarded = handler.headers.get("X-Forwarded-For", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = handler.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip
    return handler.client_address[0] if handler.client_address else "unknown"

def _check_public_analyzer_access(handler: BaseHTTPRequestHandler, tickers: List[str]) -> Optional[str]:
    cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
    if len(cleaned) != 1:
        return "공개 Analyzer는 한 번에 티커 1개만 분석할 수 있습니다."
    now = time.time()
    ip = _client_ip(handler)
    last = _ANALYZER_RATE_LIMIT.get(ip, 0.0)
    remain = int(max(0, _ANALYZER_WINDOW_SEC - (now - last)))
    if remain > 0:
        mins, secs = divmod(remain, 60)
        return f"Analyzer는 5분마다 1회만 사용할 수 있습니다. {mins:02d}:{secs:02d} 후 다시 시도해주세요."
    _ANALYZER_RATE_LIMIT[ip] = now
    if len(_ANALYZER_RATE_LIMIT) > 4096:
        cutoff = now - (_ANALYZER_WINDOW_SEC * 4)
        for key, ts in list(_ANALYZER_RATE_LIMIT.items()):
            if ts < cutoff:
                _ANALYZER_RATE_LIMIT.pop(key, None)
    return None

# ── API ──────────────────────────────────────────────────────────────
def _do_dashboard(job: Optional[_Job] = None) -> dict:
    mod = _MODS.get("dashboard")
    if not mod: return {"ok": False, "log": "dashboard 모듈 없음"}
    if job: job.append("🌊 Dashboard 데이터 수집 중…")

    if job: res = _run_bg(job, mod.run_once, False)
    else:   res, _ = _run(mod.run_once, False)

    html = res if isinstance(res, str) else None
    if not html:
        out_name = getattr(mod, "OUTPUT_FILE", "market_dashboard.html")
        for cand in [HERE / out_name,
                     Path(str(getattr(mod,"__file__","."))).parent / out_name,
                     HERE / "market_dashboard.html"]:
            if cand.exists(): html = cand.read_text(encoding="utf-8"); break

    if html:
        OUT_DASH.write_text(html, encoding="utf-8")
        fg_score, regime = 50.0, "sideways"
        try:
            mana = mod.MarketAnalyzer()
            fg_score = float((mana.fear_greed() or {}).get("score", 50))
        except Exception: pass
        try:
            import yfinance as yf, pandas as pd
            vr = yf.download("^VIX", period="5d", interval="1d", auto_adjust=False, progress=False)
            if isinstance(vr.columns, pd.MultiIndex): vr.columns = vr.columns.get_level_values(0)
            vv = float(pd.to_numeric(vr.get("Close", pd.Series(dtype=float)), errors="coerce").dropna().iloc[-1])
            sr = yf.download("SPY", period="6mo", interval="1d", auto_adjust=False, progress=False)
            if isinstance(sr.columns, pd.MultiIndex): sr.columns = sr.columns.get_level_values(0)
            sc = pd.to_numeric(sr.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
            regime = mod.SmartScoreEngine.detect_regime(sc, vv)
        except Exception: pass
        _save_regime(regime, fg_score)
        if job: job.append(f"✅ Dashboard 완료  국면:{regime}  F&G:{fg_score:.0f}")
        return {"ok": True, "html": OUT_DASH.name, "regime": regime, "fg_score": fg_score}

    msg = "HTML 생성 실패"
    if job: job.append(f"❌ {msg}")
    return {"ok": False, "log": msg}

def api_dashboard() -> dict: return _do_dashboard()

def api_dashboard_bg() -> str:
    job = _new_job()
    def _w(): job.finish(_do_dashboard(job))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def _do_analyzer(job: _Job, tickers: List[str], period: str, account: float) -> dict:
    mod = _MODS.get("analyzer")
    if not mod:
        job.append("❌ analyzer 모듈 없음")
        job.append(f"   탐색: {', '.join(MODULE_CANDIDATES['analyzer'])}")
        return {"ok": False}

    sections, total = [], len(tickers)
    for idx, tk in enumerate(tickers, 1):
        job.append(f"[{idx}/{total}] 📡 {tk} 데이터 수집 ({period})…")
        df = _run_bg(job, mod.fetch_ohlcv, tk, period)
        if df is None or (hasattr(df, "empty") and df.empty):
            job.append(f"  ⚠️  {tk}: 데이터 없음"); continue

        job.append(f"[{idx}/{total}] 🔬 {tk} 분석 중 ({len(df)}일)…")
        res = _run_bg(job, mod.run_analysis, tk, df, account)
        if not res: job.append(f"  ❌ {tk}: 분석 실패"); continue
        if "_fatal" in res: job.append(f"  ❌ {tk}: {res['_fatal']}"); continue

        job.append(f"  ✅ {tk}: 분석 완료  차트 렌더링…")
        sections.append(_an_to_html(res, mod=mod, df=df))
        job.append(f"  ✅ {tk}: 완료")

        # SL → positions.json 자동 저장
        try:
            tmod = _MODS.get("trading")
            if tmod:
                pos = tmod.load_positions()
                pos_info = res.get("pos", {})
                if pos_info and not pos_info.get("_err") and tk in pos:
                    sl = pos_info.get("stop_price") or pos_info.get("sl")
                    if sl: pos[tk]["sl"] = round(float(sl), 4); tmod.save_positions(pos)
                    job.append(f"  💾 {tk} SL → positions.json 저장")
        except Exception: pass

    if not sections: return {"ok": False}
    html = _wrap_dark(f"📊 Analyzer ({', '.join(tickers)})", "\n".join(sections))
    OUT_AN.write_text(html, encoding="utf-8")
    job.append(f"✅ Analyzer 완료 ({len(sections)}/{total}개)")
    return {"ok": True, "html": OUT_AN.name}

def api_analyzer_bg(tickers: List[str], period: str = "2y", account: float = 100_000) -> str:
    job = _new_job()
    def _w(): job.finish(_do_analyzer(job, tickers, period, account))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def _do_backtest(job: _Job, tier_filter: str = "all",
                 mode: str = "full", tickers: str = "") -> dict:
    """
    백테스트 실행 후 결과를 전량 저장.

    mode='full'   → 전체 유니버스 (BT_MODE=full 환경변수)
    mode='custom' → 티커 직접 입력 (BT_TICKERS=... 환경변수)
    tier_filter   → 완료 후 signals 저장 필터 (full 모드에서만 의미있음)
    """
    mod = _MODS.get("backtest")
    if not mod: job.append("❌ backtest 모듈 없음"); return {"ok": False}

    import os as _os

    if mode == "custom":
        raw_tickers = tickers.strip()
        if not raw_tickers:
            job.append("❌ 티커를 입력하세요"); return {"ok": False}
        job.append(f"🔬 커스텀 백테스트 시작… 입력 티커: {raw_tickers}")
        job.append("⏱  다운로드 중 (종목 수에 따라 수 초~수 분)")
        _os.environ["BT_TICKERS"] = raw_tickers
        _os.environ.pop("BT_MODE", None)
        try:
            bt_results = _run_bg(job, mod.main) or {}
        finally:
            _os.environ.pop("BT_TICKERS", None)
    else:
        job.append("🔬 전체 유니버스 백테스트 시작…")
        job.append("⏱  [1/3] 데이터 다운로드  (캐시 없을 시 60~90분)")
        _os.environ["BT_MODE"] = "full"
        try:
            bt_results = _run_bg(job, mod.main) or {}
        finally:
            _os.environ.pop("BT_MODE", None)

    # HTML 파일 탐색
    bt_src = str(getattr(mod, "__file__", str(HERE)))
    candidates = [
        OUT_BT,
        HERE / "zeus_backtest_report.html",
        HERE / "backtest_smartscore.html",
        Path(bt_src).parent / "backtest_smartscore.html",
        Path("/mnt/user-data/outputs/backtest_smartscore.html"),
    ]
    html = None
    for p in candidates:
        try:
            if p and p.exists():
                html = p.read_text(encoding="utf-8")
                job.append(f"📄 리포트: {p.name}")
                break
        except Exception: pass

    if not html:
        job.append("❌ 리포트 HTML 없음 (경로 확인)")
        return {"ok": False}

    OUT_BT.write_text(html, encoding="utf-8")

    # ── 종목 추출: results["top_tickers_ranked"] 우선, fallback 로그 파싱 ──
    ranked = bt_results.get("top_tickers_ranked", {})
    if ranked:
        # cs_combo 순위 기반 전체 목록에서 tier_filter 적용
        if tier_filter in ("tier1", "tier2", "tier3"):
            top_tickers = ranked.get(tier_filter, [])
            job.append(f"📊 {tier_filter.upper()} 종목 {len(top_tickers)}개 (cs_combo 순위, 제한 없음)")
        else:
            top_tickers = ranked.get("all", [])
            t1 = len(ranked.get("tier1", []))
            t2 = len(ranked.get("tier2", []))
            t3 = len(ranked.get("tier3", []))
            job.append(f"📊 전체 {len(top_tickers)}종목 저장 (TIER1={t1} TIER2={t2} TIER3={t3})")
        job.append(f"   기준날짜: {ranked.get('as_of_date', '?')}")
    else:
        # fallback: 로그에서 다운로드 성공 종목 파싱
        all_log = "\n".join(job.lines)
        found = re.findall(r"✅\s+([A-Z0-9\-]{1,6}):", all_log)
        # SPY는 signals에서 제외
        top_tickers = [t for t in dict.fromkeys(found) if t != "SPY"]
        job.append(f"⚠️  fallback: 로그 파싱 {len(top_tickers)}종목 (results 없음)")

    scores = ranked.get("scores", {})
    _save_signals(top_tickers, bucket_stats={"scores": scores, "tier_filter": tier_filter})
    job.append(f"✅ 백테스트 완료  {len(top_tickers)}종목 signals 저장")
    return {"ok": True, "html": OUT_BT.name, "top_tickers": top_tickers,
            "n_tickers": len(top_tickers)}

def api_backtest_bg(top_n: int = 20, tier_filter: str = "all",
                    mode: str = "full", tickers: str = "") -> str:
    """
    mode='full'   → 전체 유니버스 (BT_MODE=full)
    mode='custom' → 티커 직접 입력 (BT_TICKERS=... 환경변수 전달)
    """
    job = _new_job()
    def _w(): job.finish(_do_backtest(job, tier_filter, mode=mode, tickers=tickers))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def _do_commander(job: _Job, tickers: List[str], capital: float, strategy: str) -> dict:
    mod = _MODS.get("trading")
    if not mod: job.append("❌ trading 모듈 없음"); return {"ok": False}
    regime = _load_regime()
    max_w  = regime.get("max_weight_override", 0.30)
    orig_mw = getattr(mod, "MAX_WEIGHT", 0.30)
    mod.MAX_WEIGHT = max_w
    job.append(f"⚡ Commander  {len(tickers)}종목  ${capital:,.0f}")
    if regime: job.append(f"   국면:{regime.get('regime','')}  F&G:{regime.get('fg_score','')}  max_w={max_w*100:.0f}%")
    tk_str = ", ".join(tickers)
    with _FakeInput([tk_str, str(capital), strategy, ""]):
        html_sec = _run_bg(job, mod.run_commander)
    mod.MAX_WEIGHT = orig_mw
    if html_sec and isinstance(html_sec, str):
        full = mod.generate_full_html([html_sec], datetime.now().strftime("%Y-%m-%d %H:%M"))
        OUT_TRD.write_text(full, encoding="utf-8")
        job.append("✅ Commander 완료")
        return {"ok": True, "html": OUT_TRD.name}
    job.append("❌ Commander: 결과 없음")
    return {"ok": False}

def api_commander_bg(tickers: List[str], capital: float, strategy: str = "1") -> str:
    job = _new_job()
    def _w(): job.finish(_do_commander(job, tickers, capital, strategy))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def api_tracker() -> dict:
    mod = _MODS.get("trading")
    if not mod: return {"ok": False, "log": "trading 모듈 없음"}
    pos = _load_positions()
    if not pos: return {"ok": False, "log": "포지션 없음"}
    tickers = list(pos.keys())
    prices, l1 = _run(mod.fetch_current_prices, tickers)
    pnl_df, l2 = _run(mod.calc_pnl, pos, prices)
    ret_df, l3  = _run(mod.fetch_period_returns, tickers)
    if pnl_df is not None and ret_df is not None:
        html_sec, l4 = _run(mod.build_tracker_html, pnl_df, ret_df)
        if html_sec:
            full = mod.generate_full_html([html_sec], datetime.now().strftime("%Y-%m-%d %H:%M"))
            OUT_TRD.write_text(full, encoding="utf-8")
            return {"ok": True, "log": l1+l2+l3+l4, "html": OUT_TRD.name}
    return {"ok": False, "log": l1+l2+l3}

def _do_sentiment(job: _Job, tickers: List[str]) -> dict:
    mod = _MODS.get("trading")
    if not mod: job.append("❌ trading 모듈 없음"); return {"ok": False}
    if not tickers: tickers = list(_load_positions().keys())
    if not tickers: job.append("❌ 티커 없음"); return {"ok": False}
    job.append(f"🧠 감성 스캔: {', '.join(tickers)}")
    with _FakeInput([",".join(tickers)]):
        html_sec = _run_bg(job, mod.run_sentiment)
    if html_sec and isinstance(html_sec, str):
        full = mod.generate_full_html([html_sec], datetime.now().strftime("%Y-%m-%d %H:%M"))
        OUT_TRD.write_text(full, encoding="utf-8")
        job.append("✅ 감성 스캔 완료")
        return {"ok": True, "html": OUT_TRD.name}
    job.append("❌ 결과 없음"); return {"ok": False}

def api_sentiment_bg(tickers: List[str]) -> str:
    job = _new_job()
    def _w(): job.finish(_do_sentiment(job, tickers))
    threading.Thread(target=_w, daemon=True).start()
    return job.id

def api_pos_add(ticker: str, shares: int, avg_cost: float, buy_date: str, note: str = "") -> dict:
    mod = _MODS.get("trading")
    if not mod: return {"ok": False, "error": "trading 모듈 없음"}
    try:
        pos = mod.load_positions()
        sl = tp = None
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period="3mo", interval="1d")
            if not df.empty:
                cl = df["Close"].squeeze(); hi = df.get("High",cl).squeeze(); lo = df.get("Low",cl).squeeze()
                atr = mod.calc_atr(hi, lo, cl)
                sl, tp, _, _ = mod.calc_sl_tp(float(cl.iloc[-1]), float(atr))
        except Exception: pass
        mod.add_position(pos, ticker, shares, avg_cost, buy_date, sl=sl, tp=tp, note=note)
        mod.save_positions(pos)
        return {"ok": True, "sl": sl, "tp": tp}
    except Exception as e: return {"ok": False, "error": str(e)}

def api_pos_remove(ticker: str) -> dict:
    mod = _MODS.get("trading")
    if not mod: return {"ok": False, "error": "trading 모듈 없음"}
    try:
        pos = mod.load_positions()
        if ticker not in pos: return {"ok": False, "error": f"{ticker} 없음"}
        mod.remove_position(pos, ticker); mod.save_positions(pos)
        return {"ok": True}
    except Exception as e: return {"ok": False, "error": str(e)}

# ── HTML 빌더 ────────────────────────────────────────────────────────
_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0a0b0e;--bg2:#12141a;--panel:#16181f;--panel2:#1c1f28;
  --border:#1f2333;--border2:#252a3a;
  --accent:#5b6fff;--accent-soft:rgba(91,111,255,.12);--accent-glow:rgba(91,111,255,.25);
  --green:#22c55e;--green-soft:rgba(34,197,94,.12);
  --red:#ef4444;--red-soft:rgba(239,68,68,.1);
  --yellow:#f59e0b;--yellow-soft:rgba(245,158,11,.1);
  --text:#e8eaf0;--text2:#9aa0b8;--text3:#555d7a;--muted:#9aa0b8;
  --mono:'JetBrains Mono',monospace;--r:12px;--r2:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Pretendard','Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6}
#hdr{position:sticky;top:0;z-index:1000;background:rgba(10,11,14,.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--border);display:flex;align-items:center;height:56px;padding:0 24px;}
#logo{font-size:1.05rem;font-weight:700;color:var(--text);letter-spacing:-.3px;white-space:nowrap;margin-right:28px;display:flex;align-items:center;gap:8px;}
#logo .logo-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);}
#logo span{color:var(--text3);font-weight:400;font-size:.76rem;margin-left:2px}
#nav{display:flex;gap:2px;flex:1}
.ztab{padding:6px 14px;border-radius:var(--r2);border:none;cursor:pointer;font-size:.81rem;font-weight:500;color:var(--text3);background:transparent;transition:all .15s;white-space:nowrap;font-family:inherit;}
.ztab:hover{color:var(--text);background:var(--panel)}
.ztab.on{color:var(--accent);background:var(--accent-soft)}
.ztab .bdg{margin-left:4px;padding:1px 5px;border-radius:9px;font-size:.67rem;font-weight:600;background:var(--accent);color:#fff;}
#sbar{display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:0 24px;height:34px;font-size:.74rem;color:var(--text3);background:var(--bg2);border-bottom:1px solid var(--border);}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;margin-right:3px}
.dg{background:var(--green)}.dy{background:var(--yellow)}.dr{background:var(--red)}
#lnk{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:0 24px;height:34px;font-size:.74rem;background:var(--bg2);border-bottom:1px solid var(--border);}
.lbdg{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:9px;font-size:.69rem;font-weight:500}
.lok{background:rgba(34,197,94,.12);color:var(--green)}
.lwrn{background:rgba(245,158,11,.12);color:var(--yellow)}
.lno{background:rgba(239,68,68,.1);color:var(--red)}
.zpnl{display:none}.zpnl.on{display:block;overflow-y:auto;max-height:calc(100vh - 124px);animation:fadeIn .18s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
.zframe{width:100%;border:none;min-height:calc(100vh - 124px)}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:24px;}
.card h2{font-size:.93rem;font-weight:600;color:var(--text);margin-bottom:4px;letter-spacing:-.2px;}
.card p{color:var(--text2);font-size:.81rem;line-height:1.7;margin-bottom:16px}
.frow{display:flex;flex-direction:column;gap:4px;margin-bottom:12px}
.frow label{font-size:.74rem;color:var(--text3);font-weight:500}
.frow input,.frow select,.frow textarea{padding:9px 12px;border-radius:var(--r2);border:1px solid var(--border2);background:var(--bg2);color:var(--text);font-size:.85rem;font-family:inherit;transition:border-color .15s;}
.frow input:focus,.frow select:focus,.frow textarea:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);}
.frow textarea{resize:vertical;min-height:70px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px}
.acts{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.btn{padding:8px 16px;border-radius:var(--r2);border:1px solid var(--border2);cursor:pointer;font-size:.81rem;font-weight:600;transition:all .15s;background:transparent;color:var(--text2);font-family:inherit;}
.btn:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.btn.pri{background:var(--accent);color:#fff;border-color:var(--accent);box-shadow:0 2px 12px var(--accent-glow);}
.btn.pri:hover{background:#4a5cf0}
.btn.ok{border-color:var(--green);color:var(--green)}
.btn.ok:hover{background:rgba(34,197,94,.1)}
.btn.warn{border-color:var(--yellow);color:var(--yellow)}
.btn.warn:hover{background:rgba(245,158,11,.1)}
.btn:disabled{opacity:.35;cursor:not-allowed;pointer-events:none}
.logbox{display:none;margin-top:12px;padding:12px;border-radius:var(--r2);background:var(--bg);border:1px solid var(--border);font-family:var(--mono);font-size:.74rem;min-height:56px;max-height:200px;overflow-y:auto;line-height:1.7;}
.logbox.on{display:block}
.ll{color:var(--text2)}.ls{color:var(--green)}.lw{color:var(--yellow)}.le{color:var(--red)}
.prog-wrap{height:2px;border-radius:1px;background:var(--border);margin-top:12px;overflow:hidden;display:none;}
.prog-wrap.on{display:block}
.prog-bar{height:100%;width:0;background:var(--accent);border-radius:1px;transition:width .3s;animation:prog-pulse 1.8s ease-in-out infinite;}
@keyframes prog-pulse{0%,100%{opacity:1}50%{opacity:.4}}
.sp{display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;animation:spin .7s linear infinite;margin-right:5px;vertical-align:middle;}
@keyframes spin{to{transform:rotate(360deg)}}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{padding:4px 10px;border-radius:20px;font-size:.75rem;font-weight:600;background:var(--accent-soft);color:var(--accent);border:1px solid rgba(91,111,255,.18);cursor:pointer;transition:all .15s;}
.chip:hover{background:var(--accent);color:#fff}
.tgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;padding:20px 24px}
.radio-card{display:flex;align-items:center;gap:10px;padding:11px 14px;border-radius:var(--r2);border:1px solid var(--border2);background:var(--bg2);cursor:pointer;transition:all .15s;font-size:.83rem;color:var(--text2);}
.radio-card:hover{border-color:var(--accent);color:var(--text)}
.radio-card input[type=radio]{accent-color:var(--accent)}
.mode-group{display:flex;gap:8px;margin-bottom:14px}
.mode-group .radio-card{flex:1}
.res{padding:0 24px 24px;display:none}.res.on{display:block}
.resfrm{width:100%;border:none;min-height:700px;border-radius:var(--r);}
.panel-wrap{max-width:660px;margin:24px auto;display:flex;flex-direction:column;gap:16px;padding:0 24px}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.input-strip{position:sticky;top:0;z-index:100;background:rgba(10,11,14,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px;}
.an-inject{padding:16px 24px}
</style>"""

_JS = r"""<script>
function sw(n){
  for(let i=0;i<5;i++){
    document.getElementById('t'+i).classList.toggle('on',i===n);
    document.getElementById('p'+i).classList.toggle('on',i===n);
  }
  localStorage.setItem('zt',n);
}
(()=>sw(+localStorage.getItem('zt')||0))();
(function tick(){
  const el=document.getElementById('clk');
  if(el){const n=new Date(Date.now()+9*3600e3);
    el.textContent='KST '+n.toISOString().replace('T',' ').slice(0,19);}
  setTimeout(tick,1000);
})();

async function apiPost(ep,body={}){
  try{
    const r=await fetch('/run/'+ep,{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    return await r.json();
  }catch(e){return{ok:false,error:'네트워크 오류: '+e}}
}
async function apiGet(url){
  try{const r=await fetch(url);return await r.json();}catch(e){return{};}
}
function setBtn(id,lbl,dis=false){
  const el=document.getElementById(id);
  if(!el)return;el.innerHTML=lbl;el.disabled=dis;
}
function showRes(resId,frmId,src){
  // fetch -> inject (iframe cache bypass)
  const el=document.getElementById(resId);
  if(el)el.classList.add('on');
  const frm=document.getElementById(frmId);
  if(!frm)return;
  fetch('/'+src+'?t='+Date.now())
    .then(r=>r.text())
    .then(html=>{
      frm.srcdoc=html; // srcdoc는 매번 덮어쓰므로 캐시 없음
    })
    .catch(()=>{frm.src=src+'?t='+Date.now();});
}
async function injectHtml(injId,htmlFile){
  const inj=document.getElementById(injId);
  if(!inj)return;
  try{
    const res=await fetch('/'+htmlFile+'?t='+Date.now());
    const raw=await res.text();
    const m=raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
    inj.innerHTML=m?m[1]:raw;
    inj.scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){inj.innerHTML='<p style="color:var(--red);padding:20px">❌ 로드 실패: '+e+'</p>';}
}
function addLine(box,text){
  const cls=text.startsWith('❌')?'le':text.startsWith('⚠')?'lw':text.startsWith('✅')?'ls':'ll';
  const d=document.createElement('div');d.className=cls;d.textContent=text;
  box.appendChild(d);box.scrollTop=box.scrollHeight;
}

// 실시간 로그 폴링
async function pollJob(jobId,logBoxId,progId,onDone){
  const box=document.getElementById(logBoxId);
  const prog=progId?document.getElementById(progId):null;
  const bar=prog?prog.querySelector('.prog-bar'):null;
  if(box){box.className='logbox on';box.innerHTML='';}
  if(prog)prog.classList.add('on');
  let lastLen=0;
  while(true){
    const d=await apiGet('/run/job_log?id='+jobId);
    if(!d||!Array.isArray(d.lines)){await new Promise(r=>setTimeout(r,2000));continue;}
    const newLines=d.lines.slice(lastLen);lastLen=d.lines.length;
    if(box&&newLines.length){newLines.forEach(l=>addLine(box,l));}
    if(bar){
      const j=d.lines.join('\n');
      let pct=5;
      if(j.includes('[1/3]'))pct=10;
      const ticks=(j.match(/✅.*?:/g)||[]).length;
      if(ticks>0)pct=Math.min(60,10+ticks);
      if(j.includes('[2/3]'))pct=65;
      if(j.includes('[3/3]'))pct=85;
      if(d.done)pct=100;
      bar.style.width=pct+'%';
    }
    if(d.done){if(onDone)onDone(d.result);break;}
    await new Promise(r=>setTimeout(r,2000));
  }
}

// Dashboard
async function refreshDash(){
  setBtn('rfbtn','<span class="sp"></span> 갱신 중…',true);
  const d=await apiPost('dashboard_bg');
  if(d.job_id){
    document.getElementById('dash-log').className='logbox on';
    document.getElementById('dash-log').innerHTML='';
    await pollJob(d.job_id,'dash-log',null,r=>{
      if(r&&r.ok){
        document.getElementById('dash-frm').src=r.html+'?t='+Date.now();
        setTimeout(()=>location.reload(),1500);
      }
    });
  }
  setBtn('rfbtn','🔄 갱신',false);
}

// Analyzer
async function runAnalyzer(){
  const tk=document.getElementById('an-tk').value.trim();
  if(!tk){alert('티커 입력 필요');return;}
  if({str(PUBLIC_SITE).lower()} && /[,\s]+/.test(tk)){alert('공개 Analyzer는 티커 1개만 입력할 수 있습니다.');return;}
  setBtn('an-btn','<span class="sp"></span> 분석 중…',true);
  const inj=document.getElementById('an-inject');
  if(inj)inj.innerHTML='';
  const d=await apiPost('analyzer_bg',{
    tickers:tk,period:document.getElementById('an-pd').value,
    account:+document.getElementById('an-cap').value});
  if(!d.job_id){
    const b=document.getElementById('an-log');b.className='logbox on';
    b.innerHTML='<div class="le">❌ '+(d.error||'실행 실패')+'</div>';
    setBtn('an-btn','▶ 분석 실행',false);return;
  }
  await pollJob(d.job_id,'an-log','an-prog',async r=>{
    setBtn('an-btn','▶ 분석 실행',false);
    if(r&&r.ok&&r.html&&inj){
      // HTML 파일을 fetch해서 body 내용만 추출해 직접 주입
      try{
        const res=await fetch('/'+r.html+'?t='+Date.now());
        const raw=await res.text();
        // <body> 태그 내부만 추출
        const m=raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
        inj.innerHTML=m?m[1]:raw;
      }catch(e){
        inj.innerHTML='<p style="color:var(--red);padding:20px">❌ 결과 로드 실패: '+e+'</p>';
      }
      // 결과 상단으로 부드럽게 스크롤
      inj.scrollIntoView({behavior:'smooth',block:'start'});
    }
  });
}

// Backtest 모드 토글
function btModeChange(){
  const isFull = document.getElementById('bt-mode-full').checked;
  document.getElementById('bt-full-opts').style.display   = isFull ? '' : 'none';
  document.getElementById('bt-custom-opts').style.display = isFull ? 'none' : '';
  const lblFull   = document.getElementById('bt-lbl-full');
  const lblCustom = document.getElementById('bt-lbl-custom');
  if(lblFull)   lblFull.classList.toggle('active', isFull);
  if(lblCustom) lblCustom.classList.toggle('active', !isFull);
}

// Backtest
async function runBacktest(){
  setBtn('bt-btn','<span class="sp"></span> 실행 중…',true);
  const isFull = document.getElementById('bt-mode-full').checked;
  let payload = {};
  if(isFull){
    payload = {mode:'full', tier_filter: document.getElementById('bt-tier').value};
  } else {
    const raw = document.getElementById('bt-tickers').value.trim();
    if(!raw){ alert('티커를 입력해주세요. 예) AAPL MSFT NVDA'); setBtn('bt-btn','▶ 백테스트 실행',false); return; }
    payload = {mode:'custom', tickers: raw};
  }
  const d=await apiPost('backtest_bg', payload);
  if(!d.job_id){
    const b=document.getElementById('bt-log');b.className='logbox on';
    b.innerHTML='<div class="le">❌ '+(d.error||'실행 실패')+'</div>';
    setBtn('bt-btn','▶ 백테스트 실행',false);return;
  }
  await pollJob(d.job_id,'bt-log','bt-prog',r=>{
    setBtn('bt-btn','▶ 백테스트 실행',false);
    if(r&&r.ok){
      if(r.top_tickers&&r.top_tickers.length){
        document.getElementById('sig-chips').innerHTML=
          r.top_tickers.map(t=>`<span class="chip">${t}</span>`).join('');
        document.getElementById('sig-card').style.display='block';
        // 종목 수 표시
        const cnt=document.getElementById('sig-count');
        if(cnt) cnt.textContent=r.n_tickers+'종목';
      }
      if(r.html)showRes('bt-res','bt-frm',r.html);
    }
  });
}
async function loadSigs(){
  const d=await apiPost('load_signals');
  if(d.top_tickers&&d.top_tickers.length){
    document.getElementById('sig-chips').innerHTML=
      d.top_tickers.map(t=>`<span class="chip">${t}</span>`).join('');
    document.getElementById('sig-card').style.display='block';
    const b=document.getElementById('bt-log');b.className='logbox on';b.innerHTML='';
    addLine(b,'📥 저장 신호: '+d.top_tickers.join(', '));
    addLine(b,'생성: '+(d.generated_at||'—').slice(0,16));
  }else{
    const b=document.getElementById('bt-log');b.className='logbox on';b.innerHTML='';
    addLine(b,'저장된 신호 없음 → 백테스트 먼저 실행');
  }
}
function sigToTrading(){
  const chips=[...document.querySelectorAll('#sig-chips .chip')];
  document.getElementById('cmd-tk').value=chips.map(c=>c.textContent).join(', ');sw(4);
}

// Commander
async function runCmd(){
  let tk=document.getElementById('cmd-tk').value.trim();
  if(!tk){
    const d=await apiPost('load_signals');tk=(d.top_tickers||[]).join(', ');
    if(!tk){alert('티커 없음 → 직접 입력하거나 Backtest 먼저 실행');return;}
    document.getElementById('cmd-tk').value=tk;
  }
  setBtn('cmd-btn','<span class="sp"></span> 계산 중…',true);
  const d=await apiPost('commander_bg',{
    tickers:tk,capital:+document.getElementById('cmd-cap').value,
    strategy:document.getElementById('cmd-st').value});
  if(!d.job_id){
    const b=document.getElementById('cmd-log');b.className='logbox on';
    b.innerHTML='<div class="le">❌ '+(d.error||'실행 실패')+'</div>';
    setBtn('cmd-btn','▶ 포지션 계산',false);return;
  }
  await pollJob(d.job_id,'cmd-log',null,r=>{
    setBtn('cmd-btn','▶ 포지션 계산',false);
    if(r&&r.ok&&r.html)injectHtml('trd-inject',r.html);
  });
}
async function loadSigTk(){
  const d=await apiPost('load_signals');
  if(d.top_tickers&&d.top_tickers.length)
    document.getElementById('cmd-tk').value=d.top_tickers.join(', ');
  else alert('신호 없음 → Backtest 먼저 실행');
}

// Tracker
async function runTracker(){
  const b=document.getElementById('trk-log');b.className='logbox on';b.innerHTML='';
  const inj=document.getElementById('trd-inject');
  addLine(b,'📈 P&L 조회 중…');
  const d=await apiPost('tracker');
  if(d.log){d.log.split('\n').filter(Boolean).forEach(l=>addLine(b,l));}
  if(d.ok&&d.html&&inj){
    try{
      const res=await fetch('/'+d.html+'?t='+Date.now());
      const raw=await res.text();
      const m=raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
      inj.innerHTML=m?m[1]:raw;
      inj.scrollIntoView({behavior:'smooth',block:'start'});
    }catch(e){inj.innerHTML='<p style="color:var(--red);padding:20px">❌ 로드 실패: '+e+'</p>';}
  }else if(!d.ok) addLine(b,'❌ '+(d.log||'실패'));
}
function showAddPos(){
  const f=document.getElementById('add-pos');
  f.style.display=f.style.display==='none'?'block':'none';
  document.getElementById('pos-dt').value=new Date().toISOString().slice(0,10);
}
async function doAddPos(){
  const tk=document.getElementById('pos-tk').value.trim().toUpperCase();
  const sh=+document.getElementById('pos-sh').value;
  const cost=+document.getElementById('pos-cost').value;
  const dt=document.getElementById('pos-dt').value;
  const note=document.getElementById('pos-note').value;
  if(!tk||!sh||!cost){alert('티커·수량·단가 필수');return;}
  const d=await apiPost('add_position',{ticker:tk,shares:sh,avg_cost:cost,buy_date:dt,note});
  const b=document.getElementById('trk-log');b.className='logbox on';b.innerHTML='';
  if(d.ok){
    addLine(b,`✅ ${tk} 추가 완료`);
    if(d.sl)addLine(b,`SL: $${d.sl.toFixed(2)}  TP: ${d.tp?'$'+d.tp.toFixed(2):'—'}`);
    document.getElementById('add-pos').style.display='none';location.reload();
  }else addLine(b,'❌ '+(d.error||''));
}
async function doRemovePos(){
  const tk=prompt('청산할 티커:');if(!tk)return;
  const d=await apiPost('remove_position',{ticker:tk.trim().toUpperCase()});
  const b=document.getElementById('trk-log');b.className='logbox on';b.innerHTML='';
  addLine(b,d.ok?'✅ '+tk+' 청산':'❌ '+(d.error||''));
  if(d.ok)location.reload();
}

// Screener
async function runScreener(){
  const mode=document.querySelector('input[name="sc-mode"]:checked');
  const modeVal=mode?mode.value:'all';
  const tkInput=document.getElementById('sc-custom-tk');
  const tickers=modeVal==='custom'?(tkInput?tkInput.value.trim():''):'';
  if(modeVal==='custom'&&!tickers){
    alert('티커를 입력해 주세요 (예: AAPL, NVDA, MSFT)');return;
  }
  const label=modeVal==='custom'?`직접입력 ${tickers.split(',').filter(t=>t.trim()).length}종목`:'전체 ~3000종목 (3~5분)';
  setBtn('sc-btn',`<span class="sp"></span> 스캔 중… (${label})`,true);
  document.getElementById('sc-card').style.display='none';
  const inj=document.getElementById('sc-chips');if(inj)inj.innerHTML='';
  const d=await apiPost('screener_bg',{tickers});
  if(!d.job_id){
    const b=document.getElementById('sc-log');b.className='logbox on';
    b.innerHTML='<div class="le">❌ '+(d.error||'실행 실패')+'</div>';
    setBtn('sc-btn','▶ 스크리닝 실행',false);return;
  }
  await pollJob(d.job_id,'sc-log','sc-prog',r=>{
    setBtn('sc-btn','▶ 스크리닝 실행',false);
    if(r&&r.ok&&r.tickers&&r.tickers.length){
      const chips=document.getElementById('sc-chips');
      if(chips) chips.innerHTML=r.tickers.map(t=>`<span class="chip">${t}</span>`).join('');
      document.getElementById('sc-card').style.display='block';
      const cnt=document.getElementById('sc-count');
      if(cnt)cnt.textContent=r.count+'종목';
    }
  });
}
async function loadScreenerResult(){
  const d=await apiPost('load_screener');
  const tickers=d.tickers||[];
  if(tickers.length){
    const chips=document.getElementById('sc-chips');
    if(chips)chips.innerHTML=tickers.map(t=>`<span class="chip">${t}</span>`).join('');
    document.getElementById('sc-card').style.display='block';
    const cnt=document.getElementById('sc-count');if(cnt)cnt.textContent=d.count+'종목';
    const gen=d.generated_at?d.generated_at.slice(0,16):'';
    const b=document.getElementById('sc-log');b.className='logbox on';b.innerHTML='';
    addLine(b,'📥 저장 결과 로드: '+gen);
  }else{
    const b=document.getElementById('sc-log');b.className='logbox on';b.innerHTML='';
    addLine(b,'저장된 스크리닝 결과 없음 → 스크리닝 먼저 실행');
  }
}
function scModeChange(radio){
  const inp=document.getElementById('sc-custom-tk');
  const lbl_all=document.getElementById('sc-lbl-all');
  const lbl_custom=document.getElementById('sc-lbl-custom');
  const isCustom=radio.value==='custom';
  if(inp){
    inp.disabled=!isCustom;
    inp.style.opacity=isCustom?'1':'0.4';
    inp.style.cursor=isCustom?'text':'not-allowed';
    if(isCustom) setTimeout(()=>inp.focus(),50);
  }
  if(lbl_all) lbl_all.style.borderColor=isCustom?'var(--border)':'var(--accent)';
  if(lbl_custom) lbl_custom.style.borderColor=isCustom?'var(--accent)':'var(--border)';
}
function scToTrading(){
  const chips=[...document.querySelectorAll('#sc-chips .chip')];
  document.getElementById('cmd-tk').value=chips.map(c=>c.textContent).join(', ');sw(4);}

// Sentiment
async function runSent(){
  const tk=document.getElementById('sent-tk').value.trim();
  setBtn('sent-btn','<span class="sp"></span> 스캔 중…',true);
  const d=await apiPost('sentiment_bg',{tickers:tk});
  if(!d.job_id){
    const b=document.getElementById('sent-log');b.className='logbox on';
    b.innerHTML='<div class="le">❌ '+(d.error||'')+'</div>';
    setBtn('sent-btn','▶ 감성 스캔',false);return;
  }
  await pollJob(d.job_id,'sent-log',null,r=>{
    setBtn('sent-btn','▶ 감성 스캔',false);
    if(r&&r.ok&&r.html)injectHtml('trd-inject',r.html);
  });
}
</script>"""

def _wrap_dark(title, body):
    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title>{_CSS}</head><body>'
            f'<div style="padding:16px 20px 40px">{body}</div></body></html>')

def _lbdg(ok, partial=False):
    if ok:     return '<span class="lbdg lok">🔗 연동</span>'
    if partial:return '<span class="lbdg lwrn">🔶 부분</span>'
    return '<span class="lbdg lno">— 미실행</span>'

def _fnum(value, default=0.0) -> float:
    try:
        out = float(value)
        return out if out == out and out not in (float("inf"), float("-inf")) else float(default)
    except Exception:
        return float(default)

def _load_dashboard_detail_data() -> Dict[str, List[dict]]:
    if not OUT_DASH.exists():
        return {}
    try:
        raw = OUT_DASH.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"const\s+DETAIL_DATA\s*=\s*(\{.*?\});", raw, re.S)
        if not match:
            return {}
        data = json.loads(match.group(1))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _dashboard_sector_meta() -> Dict[str, Dict[str, str]]:
    return {
        "XLE": {"name": "에너지", "emoji": "⛽", "theme": "lime"},
        "XLU": {"name": "유틸리티", "emoji": "⚡", "theme": "mint"},
        "SKYY": {"name": "클라우드", "emoji": "☁", "theme": "sky"},
        "FINX": {"name": "핀테크", "emoji": "💳", "theme": "blue"},
        "CIBR": {"name": "사이버보안", "emoji": "🔐", "theme": "indigo"},
        "XLC": {"name": "통신", "emoji": "📡", "theme": "violet"},
        "XLK": {"name": "기술", "emoji": "💻", "theme": "blue"},
        "XLRE": {"name": "부동산", "emoji": "🏠", "theme": "sand"},
        "XAR": {"name": "항공우주", "emoji": "🚀", "theme": "orange"},
        "ICLN": {"name": "청정에너지", "emoji": "🌱", "theme": "green"},
        "SOXX": {"name": "반도체", "emoji": "🧠", "theme": "cyan"},
        "ITA": {"name": "방산", "emoji": "🛡", "theme": "orange"},
        "XLI": {"name": "산업", "emoji": "🏭", "theme": "amber"},
        "XLF": {"name": "금융", "emoji": "🏦", "theme": "blue"},
        "XBI": {"name": "바이오", "emoji": "🧬", "theme": "pink"},
        "XLP": {"name": "필수소비재", "emoji": "🛒", "theme": "lime"},
        "XLY": {"name": "임의소비재", "emoji": "🛍", "theme": "rose"},
        "XLV": {"name": "헬스케어", "emoji": "🏥", "theme": "emerald"},
        "LIT": {"name": "리튬/배터리", "emoji": "🔋", "theme": "yellow"},
        "XLB": {"name": "소재", "emoji": "🪨", "theme": "sand"},
    }

def _score_tier(score: float) -> str:
    s = _fnum(score)
    if s >= 80:
        return "tier-s"
    if s >= 65:
        return "tier-a"
    if s >= 50:
        return "tier-b"
    if s >= 35:
        return "tier-c"
    return "tier-d"

def _refine_rows_with_tree(rows: List[dict]) -> List[dict]:
    if len(rows) < 8:
        return rows
    try:
        from sklearn.tree import DecisionTreeRegressor
    except Exception:
        return rows

    feats = []
    targets = []
    for row in rows:
        score = _fnum(row.get("score"))
        r1 = _fnum(row.get("ret_1d"))
        r5 = _fnum(row.get("ret_5d"))
        price = _fnum(row.get("price"), 1.0)
        signal = 1.0 if row.get("signal") else 0.0
        holding = 1.0 if row.get("holding") else 0.0
        feats.append([score, r1, r5, math.log1p(max(price, 0.0)), signal, holding])
        targets.append(score + 0.9 * r5 + 0.45 * r1 + 5.0 * signal + 2.5 * holding)

    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)
    model.fit(feats, targets)
    preds = [float(v) for v in model.predict(feats)]
    pred_min = min(preds) if preds else 0.0
    pred_max = max(preds) if preds else 0.0

    refined = []
    for row, pred in zip(rows, preds):
        base = _fnum(row.get("score"))
        if pred_max - pred_min < 1e-9:
            tree_score = base
        else:
            tree_score = 100.0 * (pred - pred_min) / (pred_max - pred_min)
        fused_score = max(0.0, min(100.0, 0.62 * base + 0.38 * tree_score))
        out = dict(row)
        out["smart_score"] = round(base, 1)
        out["tree_score"] = round(tree_score, 1)
        out["fused_score"] = round(fused_score, 1)
        out["base_score"] = round(base, 1)
        out["dt_score"] = round(tree_score, 1)
        out["score"] = round(fused_score, 1)
        out["score_tier"] = _score_tier(fused_score)
        refined.append(out)
    return refined

def _sector_heat_score(rows: List[dict], avg_1d: float, avg_5d: float,
                       positive_ratio: int, signal_count: int, holding_count: int) -> float:
    if not rows:
        return 0.0
    top_n = max(3, min(6, len(rows) // 4 or 3))
    leaders = rows[:top_n]
    avg_score = sum(item["score"] for item in rows) / len(rows)
    leader_mean = sum(item["score"] for item in leaders) / len(leaders)
    excess_mean = sum(max(item["score"] - 50.0, 0.0) for item in rows) / len(rows)
    momentum = max(-18.0, min(18.0, avg_5d * 4.2 + avg_1d * 2.1))
    breadth = positive_ratio * 0.24
    leadership = max(0.0, leader_mean - 55.0) * 0.7
    signal_boost = min(12.0, signal_count * 5.0)
    holding_boost = min(6.0, holding_count * 2.0)
    heat = 28.0 + excess_mean * 1.9 + max(0.0, avg_score - 48.0) * 0.8 + leadership + breadth + momentum + signal_boost + holding_boost
    return round(max(0.0, min(100.0, heat)), 1)

def _build_dashboard_home(sigs: dict, regime: dict, pos: dict) -> Tuple[str, str]:
    detail_data = _load_dashboard_detail_data()
    meta_map = _dashboard_sector_meta()
    signal_tickers = set(sigs.get("top_tickers", []))
    holding_tickers = set(pos.keys())
    sectors = []
    details_for_js = {}
    attention_rows = []

    for etf, rows in detail_data.items():
        if not isinstance(rows, list) or not rows:
            continue
        meta = meta_map.get(etf, {"name": etf, "emoji": "◌", "theme": "blue"})
        normalized = []
        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            score = _fnum(row.get("cs_combo", row.get("smart_score", row.get("cs_score", 0.0))))
            normalized.append({
                "ticker": ticker,
                "price": round(_fnum(row.get("price")), 2),
                "score": round(score, 1),
                "ret_1d": round(_fnum(row.get("ret_1d")), 2),
                "ret_5d": round(_fnum(row.get("ret_5d")), 2),
                "grade": row.get("cs_grade") or row.get("grade") or "-",
                "signal": ticker in signal_tickers,
                "holding": ticker in holding_tickers,
                "signals_count": len(row.get("accum_tags", []) or []),
                "accum_tags": list(row.get("accum_tags", []) or []),
            })
        normalized = _refine_rows_with_tree(normalized)
        normalized.sort(key=lambda item: (item["score"], item["ret_5d"], item["ret_1d"]), reverse=True)
        if not normalized:
            continue
        for item in normalized:
            if item.get("signals_count", 0) >= 3:
                attention_rows.append({**item, "etf": etf, "sector_name": meta["name"]})
        avg_score = round(sum(item["score"] for item in normalized) / len(normalized), 1)
        avg_1d = round(sum(item["ret_1d"] for item in normalized) / len(normalized), 2)
        avg_5d = round(sum(item["ret_5d"] for item in normalized) / len(normalized), 2)
        positive_ratio = round(sum(1 for item in normalized if item["ret_1d"] > 0) / len(normalized) * 100)
        signal_count = sum(1 for item in normalized if item["signal"])
        holding_count = sum(1 for item in normalized if item["holding"])
        leader = normalized[0]
        heat_score = _sector_heat_score(normalized, avg_1d, avg_5d, positive_ratio, signal_count, holding_count)
        sectors.append({
            "etf": etf,
            "name": meta["name"],
            "emoji": meta["emoji"],
            "theme": meta["theme"],
            "count": len(normalized),
            "avg_score": avg_score,
            "heat_score": heat_score,
            "avg_1d": avg_1d,
            "avg_5d": avg_5d,
            "positive_ratio": positive_ratio,
            "leader": leader["ticker"],
            "leader_score": leader["score"],
            "score_tier": _score_tier(heat_score),
            "signal_count": signal_count,
            "holding_count": holding_count,
        })
        details_for_js[etf] = {
            "meta": meta,
            "stats": {
                "count": len(normalized),
                "avg_score": avg_score,
                "heat_score": heat_score,
                "avg_1d": avg_1d,
                "avg_5d": avg_5d,
                "positive_ratio": positive_ratio,
                "signal_count": signal_count,
                "holding_count": holding_count,
            },
            "rows": normalized[:18],
        }

    sectors.sort(key=lambda item: (item["heat_score"], item["signal_count"], item["avg_5d"]), reverse=True)
    lead_sectors = sectors[:3]
    visible_sectors = sectors[:12]
    signal_scores = sigs.get("bucket_stats", {}).get("scores", {}) if isinstance(sigs.get("bucket_stats", {}), dict) else {}
    avg_signal_score = round(sum(_fnum(v) for v in signal_scores.values()) / len(signal_scores), 1) if signal_scores else 0.0
    attention_rows.sort(key=lambda item: (item.get("signals_count", 0), item["score"], item["ret_5d"]), reverse=True)
    total_positions = len(pos)
    total_value = sum(_fnum(p.get("shares")) * _fnum(p.get("avg_cost")) for p in pos.values())
    regime_key = regime.get("regime", "")
    regime_label = {"bull": "리스크 온", "bear": "리스크 오프", "sideways": "중립 박스권"}.get(regime_key, "데이터 대기")
    regime_tone = {"bull": "good", "bear": "bad", "sideways": "flat"}.get(regime_key, "flat")

    lead_html = "".join(
        f"""<button class="hero-sector-card tone-{sec['theme']} {sec['score_tier']}" onclick="openSectorDetail('{sec['etf']}')"><div class="hero-sector-top"><span class="hero-sector-icon">{sec['emoji']}</span><div><div class="hero-sector-name">{sec['name']}</div><div class="hero-sector-etf">{sec['etf']} · {sec['count']}종목</div></div></div><div class="hero-sector-metrics"><div><span>섹터 Heat</span><strong class="score-number {sec['score_tier']}">{sec['heat_score']}</strong></div><div><span>5일 수익률</span><strong class="{'up' if sec['avg_5d'] >= 0 else 'down'}">{sec['avg_5d']:+.2f}%</strong></div></div><div class="hero-sector-foot">리더 {sec['leader']} · {sec['leader_score']:.1f}점 · 평균 {sec['avg_score']:.1f}</div></button>"""
        for sec in lead_sectors
    ) or '<div class="empty-state">섹터 요약 데이터를 아직 불러오지 못했습니다.</div>'

    sector_cards_html = "".join(
        f"""<button class="sector-spot-card tone-{sec['theme']} {sec['score_tier']}" onclick="openSectorDetail('{sec['etf']}')"><div class="sector-spot-head"><div class="sector-spot-title"><span class="sector-spot-icon">{sec['emoji']}</span><div><div class="sector-spot-name">{sec['name']}</div><div class="sector-spot-sub">{sec['etf']} · 리더 {sec['leader']}</div></div></div><div class="sector-spot-score {sec['score_tier']}">{sec['heat_score']:.1f}</div></div><div class="sector-spot-grid"><div><span>1일</span><strong class="{'up' if sec['avg_1d'] >= 0 else 'down'}">{sec['avg_1d']:+.2f}%</strong></div><div><span>5일</span><strong class="{'up' if sec['avg_5d'] >= 0 else 'down'}">{sec['avg_5d']:+.2f}%</strong></div><div><span>상승 비중</span><strong>{sec['positive_ratio']}%</strong></div><div><span>시그널</span><strong>{sec['signal_count']}개</strong></div></div><div class="sector-spot-foot">평균 {sec['avg_score']:.1f} · 보유 {sec['holding_count']} · 상세 보기</div></button>"""
        for sec in visible_sectors
    ) or '<div class="empty-state">대시보드 결과를 먼저 갱신하면 섹터 카드가 채워집니다.</div>'

    holdings_html = "".join(
        f'<div class="holding-chip"><span>{tk}</span><strong>{_fnum(info.get("shares"), 0):.0f}주</strong><em>${_fnum(info.get("avg_cost"), 0):.2f}</em></div>'
        for tk, info in list(pos.items())[:8]
    ) or '<div class="subtle-note">현재 등록된 포지션이 없습니다.</div>'

    signals_html = "".join(
        f'<div class="signal-pill {_score_tier(item["score"])}"><span>{item["ticker"]}</span><strong>{item["score"]:.1f}</strong><em>{item["signals_count"]} signals · {item["sector_name"]}</em></div>'
        for item in attention_rows[:8]
    ) or '<div class="subtle-note">현재 3개 이상 동시 신호가 잡힌 종목이 없습니다.</div>'

    html = f"""
    <div class="dash-shell">
      <div id="dash-log" class="logbox dash-log"></div>
      <iframe id="dash-frm" src="{OUT_DASH.name if OUT_DASH.exists() else 'about:blank'}" style="display:none"></iframe>
      <section class="hero-board">
        <div class="hero-copy">
          <div class="hero-kicker">ZEUS Integrated Market Home</div>
          <h1>숫자를 그냥 나열하지 않고, 지금 읽을 만한 흐름만 앞에 배치했습니다.</h1>
          <p>토스나 카카오페이증권처럼 상단에서 시장 맥락을 먼저 보여주고, 아래에서 섹터 카드와 종목 상세를 자연스럽게 탐색하도록 구성했습니다.</p>
          <div class="hero-badges">
            <span class="hero-badge tone-{regime_tone}">시장 국면 · {regime_label}</span>
            <span class="hero-badge">F&G {int(_fnum(regime.get('fg_score'), 0))}</span>
            <span class="hero-badge">시그널 {len(sigs.get('top_tickers', []))}종목</span>
            <span class="hero-badge">포지션 {total_positions}종목</span>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-stat-grid">
            <div class="hero-stat-card"><span>평균 시그널 점수</span><strong>{avg_signal_score:.1f}</strong><em>백테스트 상위 종목 기준</em></div>
            <div class="hero-stat-card"><span>포지션 평가 원가</span><strong>${total_value:,.0f}</strong><em>{total_positions}개 보유 종목</em></div>
            <div class="hero-stat-card"><span>활성 섹터</span><strong>{len(sectors)}</strong><em>원본 dashboard 기반</em></div>
            <div class="hero-stat-card"><span>마지막 시그널</span><strong>{sigs.get('generated_at', '')[:16].replace('T', ' ') or '-'}</strong><em>자동 연동</em></div>
          </div>
        </div>
      </section>
      <section class="hero-sector-strip">{lead_html}</section>
      <section class="summary-grid">
        <article class="summary-card"><div class="summary-head"><div><span class="summary-eyebrow">Signal Radar</span><h3>오늘 바로 확인할 종목</h3></div></div><div class="signal-pill-wrap">{signals_html}</div></article>
        <article class="summary-card"><div class="summary-head"><div><span class="summary-eyebrow">My Positions</span><h3>현재 포지션 요약</h3></div></div><div class="holding-chip-wrap">{holdings_html}</div></article>
      </section>
      <section class="sector-board">
        <div class="board-head"><div><span class="summary-eyebrow">Sector Board</span><h2>섹터별 카드 보드</h2></div><p>카드를 누르면 종목, 점수, 현재가, 1일/5일 등락률을 우측 패널에서 바로 확인합니다.</p></div>
        <div class="sector-card-grid">{sector_cards_html}</div>
      </section>
    </div>
    <div class="sector-detail-overlay" id="sector-detail-overlay" onclick="closeSectorDetail(event)">
      <aside class="sector-detail-panel" id="sector-detail-panel" onclick="event.stopPropagation()">
        <div class="sector-detail-head"><div><span class="summary-eyebrow" id="sector-detail-kicker">Sector Detail</span><h3 id="sector-detail-title">섹터 상세</h3></div><button class="sector-close-btn" onclick="closeSectorDetail()">닫기</button></div>
        <div class="sector-detail-stats" id="sector-detail-stats"></div>
        <div class="sector-detail-table-wrap"><table class="sector-detail-table"><thead><tr><th>종목</th><th>점수</th><th>현재가</th><th>1일</th><th>5일</th><th>상태</th></tr></thead><tbody id="sector-detail-body"></tbody></table></div>
      </aside>
    </div>
    """
    return html, json.dumps(details_for_js, ensure_ascii=False)

def _chart_to_b64(mod, ticker: str, df) -> str:
    """draw_chart() 결과를 base64 PNG로 변환. matplotlib 없으면 빈 문자열."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io as _io, base64 as _b64
        mod.draw_chart(ticker, df)
        buf = _io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close("all")
        buf.seek(0)
        return _b64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _an_to_html(res, mod=None, df=None) -> str:
    """run_analysis() 전체 결과를 HTML로 변환.
    M1 IC/ICIR, M2 Walk-Forward, M3 거래비용, M4 포지션,
    M5 Bootstrap, M6 스트레스, M7 BH-FDR, M8 최적화 + 차트."""
    import math

    def F(v, fmt=".4f"):
        try:
            f = float(v)
            return f"{f:{fmt}}" if not math.isnan(f) else "—"
        except Exception:
            return "—"

    def ic_color(v):
        try: return "#3fb950" if float(str(v).replace("+","")) > 0 else "#f85149"
        except Exception:
            return "#c9d1d9"

    def card(title, body):
        return (f'<div style="background:#0d1117;border:1px solid #1e2d45;'
                f'border-radius:8px;padding:14px;margin-bottom:10px">'
                f'<div style="font-size:.78rem;font-weight:700;color:#00d4ff;'
                f'margin-bottom:8px">{title}</div>{body}</div>')

    def tbl(headers, rows):
        ths = "".join(
            f'<th style="padding:4px 8px;color:#6e7681;text-align:right;'
            f'font-size:.75rem;white-space:nowrap">{h}</th>' for h in headers)
        trs = ""
        for row in rows:
            tds = ""
            for ci, cell in enumerate(row):
                align = "left" if ci == 0 else "right"
                tds += (f'<td style="padding:3px 8px;text-align:{align};'
                        f'font-family:monospace;font-size:.78rem;white-space:nowrap;'
                        f'color:#c9d1d9">{cell}</td>')
            trs += f"<tr>{tds}</tr>"
        return (f'<table style="width:100%;border-collapse:collapse;'
                f'border-top:1px solid #1e2d45">'
                f'<thead><tr style="border-bottom:1px solid #1e2d45">{ths}</tr></thead>'
                f'<tbody>{trs}</tbody></table>')

    tk    = res.get("ticker", "?")
    price = res.get("price", 0)
    n     = res.get("n", 0)
    note  = res.get("note", "")
    parts = []

    # ── 헤더 ─────────────────────────────────────────────────────────
    header = (f'<div style="display:flex;align-items:center;gap:12px;'
              f'flex-wrap:wrap;margin-bottom:16px">'
              f'<span style="font-size:1.4rem;font-weight:700;color:#00d4ff;'
              f'font-family:monospace">{tk}</span>'
              f'<span style="background:rgba(0,212,255,.12);padding:3px 12px;'
              f'border-radius:20px;font-size:.9rem;color:#00d4ff">'
              f'${price:,.2f}</span>'
              f'<span style="color:#6e7681;font-size:.8rem">{n}일 데이터</span>'
              f'{"<span style=color:#d29922;font-size:.75rem>" + note + "</span>" if note else ""}'
              f'</div>')

    # ── 차트 ─────────────────────────────────────────────────────────
    if mod is not None and df is not None:
        b64 = _chart_to_b64(mod, tk, df)
        if b64:
            parts.append(
                f'<div style="margin-bottom:4px;grid-column:1/-1">'
                f'<a href="data:image/png;base64,{b64}" target="_blank" title="클릭하면 풀사이즈 열림">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:8px;border:1px solid #1e2d45;cursor:zoom-in" '
                f'alt="{tk} chart"></a></div>')
        else:
            parts.append(
                f'<div style="padding:8px 12px;background:#0a0f16;border:1px dashed #1e2d45;'
                f'border-radius:8px;color:#6e7681;font-size:.76rem;margin-bottom:10px;'
                f'grid-column:1/-1">⚠️ 차트 없음 — pip install matplotlib</div>')

    # ── M1: IC/ICIR ──────────────────────────────────────────────────
    ic = res.get("ic", {})
    if "_err" in ic:
        parts.append(card("📐 M1 IC/ICIR",
            f'<span style="color:#f85149">❌ {ic["_err"]}</span>'))
    elif ic:
        rows = []
        all_nan = True  # scipy Mock 여부 감지
        for sig, v in ic.items():
            if sig.startswith("_"): continue
            ic_v = F(v.get("ic",   float("nan")))
            icir = F(v.get("icir", float("nan")))
            t_v  = F(v.get("t_stat", float("nan")), ".2f")
            p_v  = F(v.get("p_val",  float("nan")), ".4f")
            bh   = "✅" if v.get("bh_sig") else "—"
            col  = ic_color(ic_v)
            if ic_v != "—": all_nan = False
            rows.append([sig,
                f'<span style="color:{col};font-weight:700">{ic_v}</span>',
                icir, t_v, p_v, bh])
        if all_nan:
            # scipy.stats Mock 주입 상태: spearmanr이 실제 계산 불가
            body = ('<div style="color:#d29922;font-size:.82rem;line-height:1.8">' +
                    '⚠️  IC 계산값이 모두 NaN — scipy가 Mock 모드입니다.<br>' +
                    '<code style="background:#0a0f16;padding:2px 8px;border-radius:4px;color:#00d4ff">' +
                    'pip install scipy</code> 설치 후 서버 재시작하면 실제 IC 값이 표시됩니다.</div>')
            parts.append(card(
                "📐 M1 IC/ICIR "
                "<small style='color:#6e7681;font-weight:400'>Grinold-Kahn 2000</small>",
                body))
        else:
            parts.append(card(
                "📐 M1 IC/ICIR "
                "<small style='color:#6e7681;font-weight:400'>"
                "Grinold-Kahn 2000 — IC&gt;0.05, ICIR&gt;0.5</small>",
                tbl(["신호","IC","ICIR","t-stat","p-val","BH유의"], rows)))
    elif not ic:
        parts.append(card("📐 M1 IC/ICIR",
            '<span style="color:#6e7681">데이터 없음 (분석 결과에 ic 키 없음)</span>'))

    # ── M7: BH-FDR ───────────────────────────────────────────────────
    mt = res.get("mt", {})
    if "_err" not in mt and mt:
        rows = []
        for sig, v in mt.items():
            if not isinstance(v, dict): continue
            pr = F(v.get("p_raw",    float("nan")), ".5f")
            pa = F(v.get("p_adj_bh", float("nan")), ".5f")
            rb = "✅" if v.get("reject_bh")   else "—"
            rf = "✅" if v.get("reject_bonf") else "—"
            rows.append([sig, pr, pa, rb, rf])
        if rows:
            n_rej = sum(1 for v in mt.values()
                        if isinstance(v, dict) and v.get("reject_bh"))
            parts.append(card(
                f"🔬 M7 BH-FDR 복수검정 "
                f"<small style='color:#6e7681;font-weight:400'>"
                f"Benjamini-Hochberg 1995 — 유의 {n_rej}/{len(rows)}개</small>",
                tbl(["신호","p_raw","p_adj_BH","BH✓","Bonf✓"], rows)))

    # ── M5: Bootstrap CI ─────────────────────────────────────────────
    bt = res.get("boot", {})
    if "_err" not in bt and bt:
        sig  = bt.get("signal", "?")
        mean = F(bt.get("mean",  float("nan")))
        lo   = bt.get("ci_lo",  float("nan"))
        hi   = bt.get("ci_hi",  float("nan"))
        B    = bt.get("B", 0)
        blk  = bt.get("block", 0)
        try:
            if float(lo) > 0:
                verdict = '<span style="color:#3fb950">✅ CI 하한&gt;0 → IC 양수 통계적 유의</span>'
            elif float(hi) < 0:
                verdict = '<span style="color:#f85149">🔴 CI 상한&lt;0 → IC 음수 (역방향)</span>'
            else:
                verdict = '<span style="color:#d29922">⚠️ CI가 0 포함 → 신호 불안정</span>'
        except Exception:
            verdict = ""
        body = (f'<div style="font-family:monospace;font-size:.82rem;color:#c9d1d9">'
                f'신호: <b style="color:#00d4ff">{sig}</b> &nbsp; B={B} &nbsp; 블록={blk}<br>'
                f'IC 평균={mean} &nbsp; 95% CI=[{F(lo)}, {F(hi)}]<br>'
                f'{verdict}</div>')
        parts.append(card(
            "📊 M5 Bootstrap CI "
            "<small style='color:#6e7681;font-weight:400'>Politis-Romano 1994</small>",
            body))

    # ── M2: Walk-Forward ─────────────────────────────────────────────
    wf = res.get("wf", {})
    if "_err" not in wf and wf:
        nw  = wf.get("n_windows", 0)
        ism = F(wf.get("is_mean",  float("nan")))
        oom = F(wf.get("oos_mean", float("nan")))
        sr  = wf.get("stable", float("nan"))
        dg  = F(wf.get("degrad",   float("nan")))
        try:    sr_pct = f"{float(sr):.0%}"; sr_ok = float(sr) >= 0.70
        except Exception:
            sr_pct = "—"; sr_ok = False
        verdict = (f'<span style="color:{"#3fb950" if sr_ok else "#d29922"}">'
                   f'{"✅" if sr_ok else "⚠️"} OOS 안정비율 {sr_pct} '
                   f'{"≥" if sr_ok else "<"} 70%</span>')
        body = (f'<div style="font-family:monospace;font-size:.82rem;color:#c9d1d9;'
                f'margin-bottom:8px">'
                f'윈도우={nw} &nbsp; IS IC={ism} → OOS IC={oom}<br>'
                f'성과저하율={dg} &nbsp; {verdict}</div>')
        wins_rows = []
        for w in wf.get("wins", [])[:6]:
            is_ic  = F(w.get("is_ic",  float("nan")))
            oos_ic = F(w.get("oos_ic", float("nan")))
            col_o  = ic_color(oos_ic)
            wins_rows.append([
                w.get("split", "?"), w.get("signal", "?"),
                f'<span style="color:{ic_color(is_ic)}">{is_ic}</span>',
                f'<span style="color:{col_o}">{oos_ic}</span>'])
        if wins_rows:
            body += tbl(["기간","신호","IS IC","OOS IC"], wins_rows)
        parts.append(card(
            "🔄 M2 Walk-Forward OOS "
            "<small style='color:#6e7681;font-weight:400'>Pardo 2008 — TRAIN=252d TEST=63d</small>",
            body))

    # ── M6: 스트레스 테스트 ──────────────────────────────────────────
    st = res.get("stress", {})
    if "_err" not in st and st:
        mdd  = F(st.get("_mdd_pct", float("nan")), ".1f")
        rows = []
        for nm in ["GFC_2008","COVID_2020","TECH_CRASH_22",
                   "DOTCOM_2000","UKRAINE_2022","normal"]:
            v = st.get(nm)
            if not isinstance(v, dict): continue
            ic_v = F(v.get("ic",     float("nan")))
            t_v  = F(v.get("t_stat", float("nan")), ".2f")
            nv   = str(v.get("n", 0))
            ir   = "✅" if v.get("in_range") else "—"
            rows.append([nm,
                f'<span style="color:{ic_color(ic_v)}">{ic_v}</span>',
                t_v, nv, ir])
        body = (f'<div style="font-size:.8rem;color:#d29922;margin-bottom:6px">'
                f'포트폴리오 MDD = {mdd}%</div>')
        body += tbl(["구간","IC","t-stat","N","포함"], rows)
        parts.append(card(
            "⚡ M6 스트레스 테스트 "
            "<small style='color:#6e7681;font-weight:400'>Bookstaber 2007</small>",
            body))

    # ── M8: 신호 최적화 ──────────────────────────────────────────────
    opt = res.get("opt", {})
    if "_err" in opt:
        parts.append(card("⚖️  M8 신호 최적화",
            f'<span style="color:#f85149">❌ {opt["_err"]}</span>'))
    elif opt is not None:  # 빈 dict {}도 표시 시도
        ei_icw = F(opt.get("exp_ic_icw", float("nan")))
        ei_eq  = F(opt.get("exp_ic_eq",  float("nan")))
        n_pos  = opt.get("n_pos", 0)
        w_ic   = opt.get("w_ic", {})
        w_eq   = opt.get("w_eq", {})
        rows = []
        for sig in sorted(w_ic, key=lambda x: -w_ic.get(x, 0)):
            wi  = w_ic.get(sig, 0.0)
            we  = w_eq.get(sig, 0.0)
            bar = ("▓" * int(wi * 20)).ljust(20, "░")
            rows.append([sig, f"{wi:.4f}", f"{we:.4f}",
                f'<span style="color:#00d4ff;font-size:.7rem">{bar}</span>'])
        if not rows:
            # scipy Mock → IC 전부 NaN → 가중치 계산 불가
            body = ('<div style="color:#d29922;font-size:.82rem;line-height:1.8">' +
                    '⚠️  신호 가중치 계산 불가 — scipy가 Mock 모드입니다.<br>' +
                    '<code style="background:#0a0f16;padding:2px 8px;border-radius:4px;color:#00d4ff">' +
                    'pip install scipy</code> 설치 후 서버 재시작하면 표시됩니다.</div>')
        else:
            body = (f'<div style="font-size:.8rem;color:#c9d1d9;font-family:monospace;' +
                    f'margin-bottom:6px">' +
                    f'기대IC(IC가중)={ei_icw} &nbsp; 기대IC(등가중)={ei_eq} &nbsp; ' +
                    f'양수신호={n_pos}개</div>')
            body += tbl(["신호","IC가중","등가중","비중"], rows)
        parts.append(card(
            "⚖️  M8 신호 최적화 "
            "<small style='color:#6e7681;font-weight:400'>"
            "Markowitz 1952 + Grinold-Kahn IC가중</small>",
            body))

    # ── M3: 거래비용 ─────────────────────────────────────────────────
    c = res.get("cost", {})
    if "_err" not in c and c:
        rows = [
            ["편도 합계",
             f'<b>{F(c.get("total_pct",      float("nan")),".5f")}%</b>'],
            ["  수수료",    f'{F(c.get("commission_pct", float("nan")),".5f")}%'],
            ["  슬리피지",  f'{F(c.get("slippage_pct",   float("nan")),".5f")}%'],
            ["  스프레드",  f'{F(c.get("spread_pct",     float("nan")),".5f")}%'],
            ["왕복 합계",
             f'<b style="color:#d29922">'
             f'{F(c.get("roundtrip_pct", float("nan")),".5f")}%</b>'],
            ["참여율",      f'{F(c.get("participation",  float("nan")),".6f")}'],
        ]
        parts.append(card(
            "💸 M3 거래비용 "
            "<small style='color:#6e7681;font-weight:400'>Almgren-Chriss 2001</small>",
            tbl(["항목","값"], rows)))

    # ── M4: 포지션 사이징 ────────────────────────────────────────────
    p = res.get("pos", {})
    if "_err" not in p and p:
        if not p.get("valid"):
            body = '<span style="color:#d29922">⚠️ 포지션 0 (음수 Kelly 또는 ATR 오류)</span>'
        else:
            def kpi(label, val, color="#c9d1d9"):
                return (f'<div style="background:#161b22;padding:8px 14px;'
                        f'border-radius:8px;border:1px solid #1e2d45;text-align:center">'
                        f'<div style="font-size:.68rem;color:#6e7681">{label}</div>'
                        f'<div style="font-size:1rem;font-weight:700;color:{color};'
                        f'font-family:monospace">{val}</div></div>')
            body = (f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
                + kpi("ATR",       F(p.get("atr",          float("nan"))))
                + kpi("손절폭",    f'{F(p.get("stop_pct",  float("nan")),".2f")}%', "#d29922")
                + kpi("손절가",    f'${F(p.get("stop_price",float("nan")),".2f")}', "#f85149")
                + kpi("ATR기준",   f'{p.get("atr_shares","—")}주')
                + kpi("Kelly기준", f'{p.get("kelly_shares","—")}주')
                + kpi("최종(min)", f'{p.get("final_shares","—")}주', "#3fb950")
                + kpi("포지션금액",f'${F(p.get("position_value",float("nan")),",.0f")}', "#00d4ff")
                + kpi("포지션비중",f'{F(p.get("position_pct",float("nan")),".1f")}%', "#7c3aed")
                + '</div>')
        parts.append(card(
            "📏 M4 포지션 사이징 "
            "<small style='color:#6e7681;font-weight:400'>Kelly 1956 + Wilder ATR</small>",
            body))

    # ── 조합 ─────────────────────────────────────────────────────────
    # 차트는 full-width 위에, 나머지 카드들은 그리드로
    chart_parts = [p for p in parts if 'grid-column:1/-1' in p or '<img ' in p or 'pip install' in p]
    card_parts  = [p for p in parts if p not in chart_parts]

    chart_html = "".join(chart_parts)
    cards_html = (f'<div style="display:grid;'
                  f'grid-template-columns:repeat(auto-fit,minmax(460px,1fr));'
                  f'gap:10px;margin-top:12px">'
                  + "".join(card_parts) + "</div>") if card_parts else ""

    return (f'<div style="background:#161b22;border:1px solid #1e2d45;'
            f'border-radius:10px;padding:20px;margin-bottom:20px">'
            f'{header}{chart_html}{cards_html}</div>')


def build_unified_html(static_mode: bool = False):
    sigs=_load_signals(); regime=_load_regime(); pos=_load_positions()
    dashboard_home, dashboard_detail_json = _build_dashboard_home(sigs, regime, pos)
    sc_data=_load_screener()
    top_tks=sigs.get("top_tickers",[]); sig_dt=sigs.get("generated_at","")[:10]
    sc_tks=sc_data.get("tickers",[]) if isinstance(sc_data,dict) else []
    sc_display="display:block" if sc_tks else "display:none"
    sc_chips_html="".join(f'<span class="chip">{t}</span>' for t in sc_tks[:20])
    sc_generated=(sc_data.get("generated_at","")[:16] if isinstance(sc_data,dict) else "")
    an_generated = datetime.fromtimestamp(OUT_AN.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if OUT_AN.exists() else ""
    bt_generated = datetime.fromtimestamp(OUT_BT.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if OUT_BT.exists() else ""
    rg=regime.get("regime",""); fg=regime.get("fg_score",""); max_w=regime.get("max_weight_override",0.30)
    rg_lbl={"bull":"🟢 상승장","bear":"🔴 하락장","sideways":"🟡 횡보장"}.get(rg,"—")
    rg_col={"bull":"#3fb950","bear":"#f85149","sideways":"#d29922"}.get(rg,"#6e7681")
    pos_cnt=len(pos); pos_val=sum(p.get("shares",0)*p.get("avg_cost",0) for p in pos.values())
    now=datetime.now().strftime("%Y-%m-%d %H:%M")
    chips_html="".join(f'<span class="chip">{t}</span>' for t in top_tks) if top_tks else '<span style="color:var(--muted)">Backtest 실행 후 자동 반영</span>'
    cmd_tk_val=", ".join(top_tks[:10]) if top_tks else ""
    bt_display="display:block" if top_tks else "display:none"
    pos_note=f"✅ {pos_cnt}개 포지션  ${pos_val:,.0f}" if pos_cnt else "보유 포지션 없음"
    pos_col="var(--green)" if pos_cnt else "var(--muted)"
    analyzer_enabled = PUBLIC_SITE and not static_mode
    public_title = "GitHub Pages Read-Only" if static_mode else "Public Read-Only"
    trading_regime_banner=(f"<div style='margin:14px 20px;padding:9px 14px;border-radius:8px;border:1px solid {rg_col};font-size:.84rem'>국면: <b style='color:{rg_col}'>{rg_lbl}</b>  F&G: <b>{fg}</b>  → max_weight <b style='color:var(--yellow)'>{int(max_w*100)}%</b> 자동 적용</div>" if rg else "")
    trading_overview=(f"<div class='card' style='margin:0 20px 20px'><h2>Trading Overview</h2><p>Backtest 신호와 Screener 결과를 현재 포지션 흐름에 바로 이어서 볼 수 있게 기본 상태를 채웠습니다.</p><div class='chips'>{''.join(f'<span class=\"chip\">{t}</span>' for t in top_tks[:8]) or '<span class=\"chip\">신호 대기</span>'}{''.join(f'<span class=\"chip\">{t}</span>' for t in sc_tks[:6])}</div></div>")
    public_note = f"<div class='card' style='margin:0 20px 20px;border-color:rgba(245,158,11,.22)'><h2>{public_title}</h2><p>이 사이트는 공개용으로 동작하며, 방문자는 저장된 결과만 볼 수 있습니다.</p></div>" if (PUBLIC_SITE or static_mode) else ""
    analyzer_public_note = (
        "<div class='card' style='margin:0 20px 20px;border-color:rgba(79,109,245,.18)'><h2>Public Analyzer</h2><p>Analyzer만 공개 사용이 가능하며, 방문자는 5분마다 티커 1개씩만 분석할 수 있습니다.</p></div>"
        if analyzer_enabled else
        ("<div class='card' style='margin:0 20px 20px;border-color:rgba(79,109,245,.18)'><h2>GitHub Pages Analyzer</h2><p>GitHub Pages는 정적 사이트이므로 Analyzer 실행은 지원하지 않습니다. 최신 저장 결과만 아래에 노출됩니다.</p></div>" if static_mode else "")
    )
    bt_primary_attrs = "disabled" if (PUBLIC_SITE or static_mode) else 'onclick="runBacktest()"'
    sc_primary_attrs = "disabled" if (PUBLIC_SITE or static_mode) else 'onclick="runScreener()"'
    an_primary_attrs = 'onclick="runAnalyzer()"' if analyzer_enabled else "disabled"
    sig_move_button = "" if (PUBLIC_SITE or static_mode) else '<button class="btn ok" style="padding:2px 9px;font-size:.75rem;margin-left:8px" onclick="sigToTrading()">Trading으로 이동</button>'
    sc_move_button = "" if (PUBLIC_SITE or static_mode) else '<button class="btn ok" style="padding:2px 9px;font-size:.75rem;margin-left:8px" onclick="scToTrading()">Trading으로 이동</button>'
    mod_status=" &nbsp; ".join(
        f"<span style='color:{'#3fb950' if _MODS.get(k) else '#f85149'};font-size:.75rem'>"
        f"{'✅' if _MODS.get(k) else '❌'} {k}{'  <small style=color:#6e7681>('+_MOD_NAMES[k]+')</small>' if k in _MOD_NAMES else ''}</span>"
        for k in ["dashboard","analyzer","backtest","trading","screener"])
    dashboard_style = """
<style>
.dash-shell{padding:28px 24px 40px;background:radial-gradient(circle at top left, rgba(91,111,255,.22), transparent 28%),radial-gradient(circle at top right, rgba(38,208,124,.12), transparent 24%),linear-gradient(180deg,#f6fbff 0%,#f4f7fb 100%);min-height:calc(100vh - 124px);color:#152033}
.dash-log{margin:0 0 18px;background:rgba(255,255,255,.72);border-color:rgba(32,54,96,.08)}
.hero-board{display:grid;grid-template-columns:1.35fr .95fr;gap:18px;align-items:stretch;margin-bottom:18px}
.hero-copy,.hero-side,.summary-card,.sector-spot-card,.hero-sector-card,.sector-detail-panel{background:rgba(255,255,255,.82);backdrop-filter:blur(18px);border:1px solid rgba(28,42,76,.08);box-shadow:0 20px 60px rgba(18,31,62,.08)}
.hero-copy{padding:28px;border-radius:28px;position:relative;overflow:hidden}.hero-copy::after{content:'';position:absolute;right:-40px;top:-50px;width:180px;height:180px;background:radial-gradient(circle, rgba(91,111,255,.18), transparent 65%);pointer-events:none}
.hero-kicker,.summary-eyebrow{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:999px;background:#eef3ff;color:#4564d1;font-size:.72rem;font-weight:700;letter-spacing:.02em;text-transform:uppercase}
.hero-copy h1{margin:16px 0 12px;font-size:2rem;line-height:1.18;letter-spacing:-.04em;color:#0e1a2f}.hero-copy p{font-size:.95rem;line-height:1.7;color:#5a6883;max-width:720px}
.hero-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}.hero-badge{padding:10px 14px;border-radius:999px;background:#f4f7fb;border:1px solid rgba(21,32,51,.06);font-size:.82rem;font-weight:600;color:#32425f}.hero-badge.tone-good{background:#e9fbf0;color:#15803d}.hero-badge.tone-bad{background:#fff1f1;color:#dc2626}.hero-badge.tone-flat{background:#fff7e9;color:#b7791f}
.hero-side{padding:18px;border-radius:28px}.hero-stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;height:100%}.hero-stat-card{padding:18px;border-radius:22px;background:linear-gradient(180deg,#ffffff,#f7faff);border:1px solid rgba(48,66,112,.07);display:flex;flex-direction:column;justify-content:space-between;min-height:120px}.hero-stat-card span,.sector-spot-grid span{font-size:.76rem;color:#75829c;font-weight:600}.hero-stat-card strong{font-size:1.55rem;color:#0f1d34;letter-spacing:-.03em;margin:8px 0}.hero-stat-card em{font-style:normal;color:#6a7892;font-size:.8rem}
.hero-sector-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:18px}.hero-sector-card,.sector-spot-card{border-radius:24px;padding:18px;text-align:left;cursor:pointer;transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease}.hero-sector-card:hover,.sector-spot-card:hover{transform:translateY(-4px);box-shadow:0 24px 60px rgba(18,31,62,.14);border-color:rgba(91,111,255,.22)}
.hero-sector-top,.sector-spot-head,.summary-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.hero-sector-icon,.sector-spot-icon{display:grid;place-items:center;width:48px;height:48px;border-radius:16px;background:#f0f4ff;font-size:1.35rem}.hero-sector-name,.sector-spot-name{font-size:1rem;font-weight:800;color:#10203a}.hero-sector-etf,.sector-spot-sub,.subtle-note{font-size:.8rem;color:#70809a}
.hero-sector-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:18px 0 12px}.hero-sector-metrics div,.sector-spot-grid div{padding:10px 12px;border-radius:16px;background:#f7f9fc;border:1px solid rgba(20,32,58,.05)}.hero-sector-metrics span,.sector-spot-foot{font-size:.72rem;color:#73819a}.hero-sector-metrics strong,.sector-spot-score{display:block;margin-top:4px;font-size:1.08rem;color:#0f1d34}.hero-sector-foot{font-size:.8rem;color:#4f5f7d;font-weight:600}.score-number{display:inline-flex;align-items:center;justify-content:center;min-width:74px;padding:10px 14px;border-radius:18px;font-size:1.05rem;letter-spacing:-.03em}.tier-s{border-color:rgba(16,185,129,.22) !important;box-shadow:0 24px 60px rgba(16,185,129,.12) !important}.tier-a{border-color:rgba(59,130,246,.18) !important;box-shadow:0 20px 52px rgba(59,130,246,.10) !important}.tier-b{border-color:rgba(99,102,241,.14) !important}.tier-c{border-color:rgba(245,158,11,.16) !important}.tier-d{border-color:rgba(239,68,68,.18) !important}.score-number.tier-s,.sector-spot-score.tier-s,.signal-pill.tier-s strong{background:linear-gradient(135deg,#e7fff4,#d7f8e9);color:#0f8d5f}.score-number.tier-a,.sector-spot-score.tier-a,.signal-pill.tier-a strong{background:linear-gradient(135deg,#eef5ff,#e0edff);color:#2756d8}.score-number.tier-b,.sector-spot-score.tier-b,.signal-pill.tier-b strong{background:linear-gradient(135deg,#f1f3ff,#e7eafe);color:#5b63d6}.score-number.tier-c,.sector-spot-score.tier-c,.signal-pill.tier-c strong{background:linear-gradient(135deg,#fff6e8,#ffefd6);color:#ba6d12}.score-number.tier-d,.sector-spot-score.tier-d,.signal-pill.tier-d strong{background:linear-gradient(135deg,#fff0f0,#ffe2e2);color:#d1465b}.signal-pill.tier-s,.signal-pill.tier-a,.signal-pill.tier-b,.signal-pill.tier-c,.signal-pill.tier-d{box-shadow:inset 0 1px 0 rgba(255,255,255,.65)}.signal-pill strong,.sector-spot-score{padding:10px 12px;border-radius:16px}.hero-sector-card.tier-s::after,.hero-sector-card.tier-a::after,.hero-sector-card.tier-b::after,.hero-sector-card.tier-c::after,.hero-sector-card.tier-d::after,.sector-spot-card.tier-s::after,.sector-spot-card.tier-a::after,.sector-spot-card.tier-b::after,.sector-spot-card.tier-c::after,.sector-spot-card.tier-d::after{content:'';position:absolute;inset:auto -10% -32% auto;width:120px;height:120px;border-radius:999px;opacity:.55;pointer-events:none}.hero-sector-card,.sector-spot-card{position:relative;overflow:hidden}.hero-sector-card.tier-s::after,.sector-spot-card.tier-s::after{background:radial-gradient(circle,rgba(16,185,129,.18),transparent 70%)}.hero-sector-card.tier-a::after,.sector-spot-card.tier-a::after{background:radial-gradient(circle,rgba(59,130,246,.16),transparent 70%)}.hero-sector-card.tier-b::after,.sector-spot-card.tier-b::after{background:radial-gradient(circle,rgba(99,102,241,.15),transparent 70%)}.hero-sector-card.tier-c::after,.sector-spot-card.tier-c::after{background:radial-gradient(circle,rgba(245,158,11,.15),transparent 70%)}.hero-sector-card.tier-d::after,.sector-spot-card.tier-d::after{background:radial-gradient(circle,rgba(239,68,68,.14),transparent 70%)}.zpnl#p1,.zpnl#p2,.zpnl#p3,.zpnl#p4{background:radial-gradient(circle at top left, rgba(91,111,255,.18), transparent 26%),linear-gradient(180deg,#f6fbff 0%,#f3f6fb 100%);min-height:calc(100vh - 124px)}#p1 .card,#p2 .card,#p3 .card,#p4 .card,#p2 .res,#trd-inject > div{background:rgba(255,255,255,.84);backdrop-filter:blur(18px);border:1px solid rgba(28,42,76,.08);box-shadow:0 20px 60px rgba(18,31,62,.08);border-radius:28px;color:#16233b}#p1 .logbox,#p2 .logbox,#p3 .logbox,#p4 .logbox{background:#f7f9fd;border:1px solid rgba(20,32,58,.08);color:#30415f}#p1 input,#p1 select,#p1 textarea,#p2 input,#p2 select,#p2 textarea,#p3 input,#p3 select,#p3 textarea,#p4 input,#p4 select,#p4 textarea{background:#f8fbff !important;border:1px solid rgba(34,50,86,.10) !important;color:#16233b !important;box-shadow:inset 0 1px 0 rgba(255,255,255,.9)}#p1 label,#p2 label,#p3 label,#p4 label,#p1 p,#p2 p,#p3 p,#p4 p{color:#62718d !important}#p1 h2,#p2 h2,#p3 h2,#p4 h2{color:#10203a !important;letter-spacing:-.03em}#p1 .btn,#p2 .btn,#p3 .btn,#p4 .btn{border-radius:16px;border:1px solid rgba(54,73,118,.08);box-shadow:0 10px 28px rgba(18,31,62,.08)}#p1 .btn.pri,#p2 .btn.pri,#p3 .btn.pri,#p4 .btn.pri{background:linear-gradient(135deg,#4f6df5,#6c82ff)}#p1 .btn.ok,#p2 .btn.ok,#p3 .btn.ok,#p4 .btn.ok{background:linear-gradient(135deg,#18b779,#36c78d)}#p1 .btn.warn,#p2 .btn.warn,#p3 .btn.warn,#p4 .btn.warn{background:linear-gradient(135deg,#ff8a70,#ff6d7b)}#p2 .resfrm{background:#fff;border-radius:24px;border:1px solid rgba(20,32,58,.08)}#p3 #sc-lbl-all,#p3 #sc-lbl-custom,#p2 .radio-card{background:linear-gradient(180deg,#fff,#f7faff) !important;border:1px solid rgba(39,57,98,.09) !important}#p4 .tgrid{padding:14px 20px 30px;gap:18px}#p4 .card{padding:24px}#p1 .an-inject,#trd-inject{padding:8px 20px 32px}
.summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}.summary-card{padding:20px;border-radius:24px}.summary-card h3,.board-head h2,.sector-detail-head h3{margin-top:10px;font-size:1.2rem;color:#0f1d34;letter-spacing:-.03em}.signal-pill-wrap,.holding-chip-wrap{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.signal-pill,.holding-chip{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:18px;background:#f6f8fc;border:1px solid rgba(18,31,62,.06)}.signal-pill span,.holding-chip span{font-weight:800;color:#10203a}.signal-pill strong,.holding-chip strong{color:#4c63d8}.signal-pill em,.holding-chip em{font-style:normal;color:#70809a;font-size:.75rem}
.board-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:18px}.board-head p{max-width:420px;color:#697790;font-size:.9rem;line-height:1.6}.sector-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.sector-spot-score{padding:10px 12px;border-radius:16px;background:#eef3ff;min-width:68px;text-align:center}.sector-spot-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px}.up{color:#11a66a !important}.down{color:#e85b6b !important}.empty-state{padding:28px;border-radius:22px;background:rgba(255,255,255,.7);border:1px dashed rgba(30,45,90,.14);color:#73819a}
.tone-blue .hero-sector-icon,.tone-blue .sector-spot-icon{background:#ecf2ff}.tone-indigo .hero-sector-icon,.tone-indigo .sector-spot-icon{background:#eef0ff}.tone-lime .hero-sector-icon,.tone-lime .sector-spot-icon{background:#eefcf2}.tone-mint .hero-sector-icon,.tone-mint .sector-spot-icon{background:#ecfffb}.tone-sky .hero-sector-icon,.tone-sky .sector-spot-icon{background:#ecf8ff}.tone-violet .hero-sector-icon,.tone-violet .sector-spot-icon{background:#f4efff}.tone-sand .hero-sector-icon,.tone-sand .sector-spot-icon{background:#fff6eb}.tone-orange .hero-sector-icon,.tone-orange .sector-spot-icon{background:#fff1ea}.tone-green .hero-sector-icon,.tone-green .sector-spot-icon{background:#eefdf3}.tone-cyan .hero-sector-icon,.tone-cyan .sector-spot-icon{background:#ebfbff}.tone-amber .hero-sector-icon,.tone-amber .sector-spot-icon{background:#fff7e8}.tone-pink .hero-sector-icon,.tone-pink .sector-spot-icon{background:#fff0f5}.tone-rose .hero-sector-icon,.tone-rose .sector-spot-icon{background:#fff0f2}.tone-emerald .hero-sector-icon,.tone-emerald .sector-spot-icon{background:#ebfff7}.tone-yellow .hero-sector-icon,.tone-yellow .sector-spot-icon{background:#fffbe8}
.sector-detail-overlay{position:fixed;inset:0;background:rgba(9,15,27,.36);backdrop-filter:blur(10px);display:none;align-items:stretch;justify-content:flex-end;z-index:1200}.sector-detail-overlay.on{display:flex}.sector-detail-panel{width:min(720px,100%);height:100%;padding:24px 22px 26px;border-radius:32px 0 0 32px;overflow-y:auto;background:rgba(253,254,255,.95)}.sector-detail-head{display:flex;align-items:start;justify-content:space-between;gap:12px;margin-bottom:18px}.sector-close-btn{border:none;background:#eef3ff;color:#4564d1;padding:10px 14px;border-radius:999px;font-weight:700;cursor:pointer}.sector-detail-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:16px}.sector-detail-stats .stat{padding:12px 14px;border-radius:18px;background:#f6f8fc;border:1px solid rgba(20,32,58,.06)}.sector-detail-stats .stat span{display:block;font-size:.72rem;color:#73819a}.sector-detail-stats .stat strong{display:block;margin-top:6px;font-size:1rem;color:#10203a}.sector-detail-table-wrap{border:1px solid rgba(21,32,51,.08);border-radius:20px;overflow:hidden;background:#fff}.sector-detail-table{width:100%;border-collapse:collapse}.sector-detail-table th,.sector-detail-table td{padding:13px 14px;border-bottom:1px solid rgba(21,32,51,.06);text-align:left;font-size:.88rem}.sector-detail-table th{background:#f7f9fc;color:#6d7b94;font-size:.74rem;text-transform:uppercase;letter-spacing:.04em}.sector-detail-table td strong{font-size:.95rem;color:#10203a}.row-tags{display:flex;gap:6px;flex-wrap:wrap}.row-tag{padding:5px 8px;border-radius:999px;background:#eef3ff;color:#4564d1;font-size:.7rem;font-weight:700}.row-score-main{padding:7px 12px;font-size:.82rem;background:linear-gradient(135deg,#243b8a,#5b6fff);color:#fff;box-shadow:0 8px 22px rgba(79,109,245,.22)}
@media (max-width:1200px){.hero-board,.summary-grid,.sector-card-grid,.hero-sector-strip{grid-template-columns:1fr 1fr}.board-head{display:block}.board-head p{margin-top:10px}}
@media (max-width:820px){.dash-shell{padding:18px 14px 30px}.hero-board,.summary-grid,.sector-card-grid,.hero-sector-strip,.hero-stat-grid,.sector-detail-stats{grid-template-columns:1fr}.hero-copy h1{font-size:1.55rem}.sector-detail-panel{width:100%;border-radius:28px 28px 0 0;height:auto;max-height:90vh;margin-top:auto}}
</style>"""
    dashboard_script = f"""<script>
const DASHBOARD_SECTOR_DATA = {dashboard_detail_json};
function openSectorDetail(etf){{
  const payload = DASHBOARD_SECTOR_DATA[etf];
  if(!payload) return;
  const overlay = document.getElementById('sector-detail-overlay');
  document.getElementById('sector-detail-kicker').textContent = payload.meta.name + ' · ' + etf;
  document.getElementById('sector-detail-title').textContent = '섹터 상세';
  document.getElementById('sector-detail-stats').innerHTML = [
    ['섹터 Heat', payload.stats.heat_score],
    ['평균 점수', payload.stats.avg_score],
    ['1일 평균', (payload.stats.avg_1d>=0?'+':'') + payload.stats.avg_1d.toFixed(2) + '%'],
    ['5일 평균', (payload.stats.avg_5d>=0?'+':'') + payload.stats.avg_5d.toFixed(2) + '%']
  ].map(item=>`<div class="stat"><span>${{item[0]}}</span><strong>${{item[1]}}</strong></div>`).join('');
  document.getElementById('sector-detail-body').innerHTML = payload.rows.map(row=>`<tr><td><strong>${{row.ticker}}</strong></td><td><div class="row-tags"><span class="row-tag row-score-main ${{row.score_tier || ''}}">fused ${{row.score.toFixed(1)}}</span><span class="row-tag">smart ${{Number(row.smart_score ?? row.base_score ?? row.score).toFixed(1)}}</span><span class="row-tag">tree ${{Number(row.tree_score ?? row.dt_score ?? row.score).toFixed(1)}}</span></div></td><td>${{row.price.toFixed(2)}}</td><td class="${{row.ret_1d>=0?'up':'down'}}">${{row.ret_1d>=0?'+':''}}${{row.ret_1d.toFixed(2)}}%</td><td class="${{row.ret_5d>=0?'up':'down'}}">${{row.ret_5d>=0?'+':''}}${{row.ret_5d.toFixed(2)}}%</td><td><div class="row-tags">${{row.signal?'<span class="row-tag">SIGNAL</span>':''}}${{row.holding?'<span class="row-tag">HOLDING</span>':''}}<span class="row-tag">${{row.grade}}</span><span class="row-tag">signals ${{row.signals_count || 0}}</span></div></td></tr>`).join('');
  overlay.classList.add('on');
  document.body.style.overflow = 'hidden';
}}
function closeSectorDetail(ev){{
  if(ev && ev.target && ev.target.id && ev.target.id !== 'sector-detail-overlay') return;
  const overlay = document.getElementById('sector-detail-overlay');
  if(overlay) overlay.classList.remove('on');
  document.body.style.overflow = '';
}}
</script>"""

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZEUS Integrated Dashboard</title>{_CSS}{dashboard_style}</head><body>
<div id="hdr">
  <div id="logo"><div class="logo-dot"></div>ZEUS <span>Trading System</span></div>
  <nav id="nav">
    <button class="ztab" id="t0" onclick="sw(0)">Dashboard</button>
    <button class="ztab" id="t1" onclick="sw(1)">Analyzer</button>
    <button class="ztab" id="t2" onclick="sw(2)">Backtest{"<span class='bdg'>"+str(len(top_tks))+"</span>" if top_tks else ""}</button>
    <button class="ztab" id="t3" onclick="sw(3)">Screener</button>
    <button class="ztab" id="t4" onclick="sw(4)">Trading{"<span class='bdg'>"+str(pos_cnt)+"</span>" if pos_cnt else ""}</button>
  </nav>
  <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
    <span id="clk" style="font-family:monospace;font-size:.75rem;color:var(--muted)"></span>
    <button class="btn pri" id="rfbtn" {'disabled' if (PUBLIC_SITE or static_mode) else 'onclick="refreshDash()"'}>{'GitHub Pages' if static_mode else ('공개모드' if PUBLIC_SITE else '갱신')}</button>
  </div>
</div>
<div id="lnk">
  <span style="color:var(--muted);font-weight:600;font-size:.76rem">모듈</span> {mod_status}
  <span style="color:var(--border)">│</span>
  Backtest→Trading {_lbdg(bool(top_tks), not top_tks)}
  Screener→Trading {_lbdg(SCREENER_JSON.exists())}
  국면→Trading {_lbdg(bool(rg), not rg)}
  {"<span style='color:var(--muted)'>│</span> 신호 <b style='color:var(--accent)'>"+str(len(top_tks))+"종목</b> ("+sig_dt+")" if top_tks else ""}
  {"<span style='color:var(--muted)'>│</span> <b style='color:"+rg_col+"'>"+rg_lbl+"</b> F&G:"+str(fg)+" → max_w <b style='color:var(--yellow)'>"+str(int(max_w*100))+"%</b>" if rg else ""}
  <span style="margin-left:auto;color:var(--muted);font-size:.75rem">{now}</span>
</div>
<div id="sbar">
  <span><span class="dot dg"></span>정상</span>
  <span>포지션 <b>{pos_cnt}종목</b></span>
  {"<span>평가금액 <b>$"+f"{pos_val:,.0f}"+"</b></span>" if pos_val else ""}
</div>

<!-- 탭0: Dashboard -->
<div class="zpnl" id="p0">
  {dashboard_home}
</div>

<!-- 탭1: Analyzer -->
<div class="zpnl" id="p1">
  {analyzer_public_note if PUBLIC_SITE else ""}
  <!-- 입력 폼: 상단 고정 스트립 -->
  <div style="position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px">
    <div style="flex:2;min-width:200px">
      <label style="font-size:.75rem;color:var(--muted);display:block;margin-bottom:4px">티커 {('(한 번에 1개)' if PUBLIC_SITE else '(콤마 구분)')}</label>
      <input id="an-tk" type="text" placeholder="NVDA"
        style="width:100%;padding:7px 10px;border-radius:7px;border:1px solid var(--border);background:#0d1117;color:var(--text);font-size:.88rem">
    </div>
    <div style="flex:1;min-width:110px">
      <label style="font-size:.75rem;color:var(--muted);display:block;margin-bottom:4px">기간</label>
      <select id="an-pd" style="width:100%;padding:7px 10px;border-radius:7px;border:1px solid var(--border);background:#0d1117;color:var(--text);font-size:.88rem">
        <option value="2y" selected>2년 (권장)</option><option value="1y">1년</option>
        <option value="3y">3년</option><option value="5y">5년</option>
      </select>
    </div>
    <div style="flex:1;min-width:120px">
      <label style="font-size:.75rem;color:var(--muted);display:block;margin-bottom:4px">계좌 (USD)</label>
      <input id="an-cap" type="number" value="100000" min="1000"
        style="width:100%;padding:7px 10px;border-radius:7px;border:1px solid var(--border);background:#0d1117;color:var(--text);font-size:.88rem">
    </div>
    <div style="display:flex;flex-direction:column;gap:5px">
      <button class="btn pri" id="an-btn" {an_primary_attrs} style="white-space:nowrap">분석 실행</button>
    </div>
  </div>
  <!-- 진행 바 + 로그 (컴팩트) -->
  <div style="padding:0 20px">
    <div class="prog-wrap" id="an-prog"><div class="prog-bar"></div></div>
    <div class="logbox" id="an-log"></div>
  </div>
  <div class="card" style="margin:0 20px 16px">
    <h2>저장된 Analyzer 결과{("  <small style='color:var(--muted);font-weight:400;font-size:.8rem'>("+an_generated+")</small>" if an_generated else "")}</h2>
    <p>{("최신 분석 결과가 아래에 유지되며, 새 요청이 완료되면 이 영역이 교체됩니다." if analyzer_enabled else "최신 저장 결과가 아래에 표시됩니다.")}</p>
  </div>
  <!-- 결과 직접 주입 영역 (스크롤은 탭 패널이 담당) -->
  <div id="an-inject" class="an-inject">{('<iframe class="resfrm" id="an-frm" src="'+OUT_AN.name+'?t='+str(int(OUT_AN.stat().st_mtime))+'" style="min-height:900px;background:#fff"></iframe>' if OUT_AN.exists() else '<div class="card" style="margin:0 20px 20px"><p>저장된 Analyzer 결과가 아직 없습니다.</p></div>')}</div>
</div>

<!-- 탭2: Backtest -->
<div class="zpnl" id="p2">
  {public_note if PUBLIC_SITE else ""}
  <div class="panel-wrap">
    <div class="card">
      <div style="font-size:.7rem;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px">Backtest Engine</div>
      <h2>SmartScore 팩터 검증</h2>
      <p>완료 후 상위 종목이 <b style="color:var(--text)">zeus_signals.json</b>에 자동 저장됩니다.<br>
         전체모드 최초 실행 90~150분 · 종목 직접 입력 수 초~수 분</p>

      <!-- 실행 모드 선택 -->
      <div class="mode-group">
        <label class="radio-card" id="bt-lbl-full">
          <input type="radio" name="bt-mode" value="full" id="bt-mode-full" checked onchange="btModeChange()">
          <div>
            <div style="font-weight:600;color:var(--text);font-size:.84rem">전체 유니버스</div>
            <div style="font-size:.74rem;color:var(--text3);margin-top:2px">~900종목 · 90~150분</div>
          </div>
        </label>
        <label class="radio-card" id="bt-lbl-custom">
          <input type="radio" name="bt-mode" value="custom" id="bt-mode-custom" onchange="btModeChange()">
          <div>
            <div style="font-weight:600;color:var(--text);font-size:.84rem">종목 직접 입력</div>
            <div style="font-size:.74rem;color:var(--text3);margin-top:2px">수 초~수 분</div>
          </div>
        </label>
      </div>

      <!-- 전체 모드 옵션 -->
      <div id="bt-full-opts">
        <div class="frow">
          <label>대상 티어</label>
          <select id="bt-tier">
            <option value="all" selected>전체 (~900종목)</option>
            <option value="tier1">TIER1 — 대형주 200종목</option>
            <option value="tier2">TIER2 — 중형 모멘텀 200종목</option>
            <option value="tier3">TIER3 — 소형/고베타 84종목</option>
            <option value="tier4">TIER4 — Russell 2000 확장</option>
          </select>
        </div>
      </div>

      <!-- 커스텀 모드 입력 -->
      <div id="bt-custom-opts" style="display:none">
        <div class="frow">
          <label>티커 입력 (콤마 또는 공백 구분)</label>
          <textarea id="bt-tickers" placeholder="예) AAPL MSFT NVDA TSLA PLTR"></textarea>
          <small style="color:var(--text3);font-size:.73rem">SPY는 자동 포함 · 같은 TIER 종목 2개 이상 권장</small>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
        <span id="sig-count" style="color:var(--green);font-weight:700;font-size:.82rem">—</span>
        <span style="color:var(--text3);font-size:.76rem">저장된 종목</span>
      </div>

      <div class="acts">
        <button class="btn pri" id="bt-btn" {bt_primary_attrs}>백테스트 실행</button>
        <button class="btn ok" onclick="loadSigs()">신호 로드</button>
      </div>
    <div class="prog-wrap" id="bt-prog"><div class="prog-bar"></div></div>
    <div class="logbox" id="bt-log"></div>
  </div>
  <div class="card" id="sig-card" style="margin:0 20px 16px;{bt_display}">
    <h2>신호 종목{"  <small style='color:var(--muted);font-weight:400;font-size:.8rem'>("+sig_dt+")</small>" if sig_dt else ""}</h2>
    <div class="chips" id="sig-chips">{chips_html}</div>
    <div style="margin-top:10px;font-size:.8rem;color:var(--muted)">
      {('공개 사이트에서는 저장된 백테스트 결과만 노출됩니다.' if PUBLIC_SITE else 'Trading Commander에 자동 로드됩니다.')}
      {sig_move_button}
    </div>
  </div>
  <div class="res" id="bt-res" style="padding:0 20px 20px"><iframe class="resfrm" id="bt-frm" src="{OUT_BT.name+'?t='+str(int(OUT_BT.stat().st_mtime)) if OUT_BT.exists() else 'about:blank'}" style="min-height:700px"></iframe></div>
  {"<div class='card' style='margin:0 20px 20px'><p>최신 백테스트 저장 시각: <b>"+bt_generated+"</b></p></div>" if bt_generated else ""}
</div>

<!-- 탭3: Screener -->
<div class="zpnl" id="p3">
  {public_note if PUBLIC_SITE else ""}
  <div class="panel-wrap"><div class="card">
    <div style="font-size:.7rem;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px">Screener</div><h2>반등 응축 종목 탐색</h2>
    <p>낙폭 15%↑ · 반등 10~30% · 이평선 수렴 · OBV 상승 · Volume Dry-up<br>완료 후 <b style="color:var(--green)">Trading 탭 자동 연동</b></p>

    <!-- 모드 선택 -->
    <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:14px">
      <label style="display:flex;align-items:center;gap:10px;cursor:pointer;
                    padding:10px 14px;border-radius:8px;border:1px solid var(--border);
                    background:#0a0f16" id="sc-lbl-all">
        <input type="radio" name="sc-mode" value="all" checked
               onchange="scModeChange(this)"
               {'disabled' if (PUBLIC_SITE or static_mode) else ''}
               style="accent-color:var(--accent);width:15px;height:15px">
        <span>
          <b style="color:var(--text)">전체 자동 수집</b>
          <span style="color:var(--muted);font-size:.8rem;margin-left:8px">
            S&amp;P500 + S&amp;P400 + S&amp;P600 + Russell2000 ≈ 3,000종목
            <b style="color:var(--yellow)"> (약 3~5분)</b>
          </span>
        </span>
      </label>
      <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer;
                    padding:10px 14px;border-radius:8px;border:1px solid var(--border);
                    background:#0a0f16" id="sc-lbl-custom">
        <input type="radio" name="sc-mode" value="custom"
               onchange="scModeChange(this)"
               {'disabled' if (PUBLIC_SITE or static_mode) else ''}
               style="accent-color:var(--accent);width:15px;height:15px;margin-top:3px">
        <span style="flex:1">
          <b style="color:var(--text)">직접 입력</b>
          <span style="color:var(--muted);font-size:.8rem;margin-left:8px">원하는 종목만 빠르게 검사 (수초)</span>
          <input id="sc-custom-tk" type="text" disabled
            placeholder="예: AAPL, NVDA, MSFT, TSLA, AMD"
            style="display:block;width:100%;margin-top:8px;padding:7px 10px;
                   border-radius:7px;border:1px solid var(--border);
                   background:#0d1117;color:var(--text);font-size:.88rem;
                   opacity:.4;cursor:not-allowed"
            onkeydown="if(event.key==='Enter')runScreener()">
        </span>
      </label>
    </div>

    <div class="acts">
      <button class="btn pri" id="sc-btn" {sc_primary_attrs}>스크리닝 실행</button>
      <button class="btn ok" onclick="loadScreenerResult()">📥 저장 결과 로드</button>
    </div>
    <div class="prog-wrap" id="sc-prog"><div class="prog-bar"></div></div>
    <div class="logbox" id="sc-log"></div>
  </div>
  <div class="card" id="sc-card" style="margin:0 20px 16px;{sc_display}">
    <h2>🎯 통과 종목 &nbsp;<span id="sc-count" style="color:var(--green);font-weight:700">{len(sc_tks)}종목</span></h2>
    <div class="chips" id="sc-chips">{sc_chips_html or '<span class="chip">저장 결과 없음</span>'}</div>
    <div style="margin-top:10px;font-size:.8rem;color:var(--muted)">
      {sc_generated or '저장된 Screener 결과가 없으면 스캔 실행 후 여기 표시됩니다.'}
      {sc_move_button}
    </div>
  </div>
</div>

<!-- 탭4: Trading -->
<div class="zpnl" id="p4">
  {trading_regime_banner}
  <div class="card" style="margin:0 20px 20px;border-color:rgba(239,68,68,.18)">
    <h2>Trading Locked</h2>
    <p>공개 사이트에서는 주문, 포지션 계산, P&amp;L 추적, 감성 스캔을 모두 막았습니다. 운영자는 백엔드에서 결과를 생성한 뒤 저장 산출물만 이 페이지에 반영하면 됩니다.</p>
    <div style="margin-bottom:12px;font-size:.84rem;color:{pos_col}">{pos_note}</div>
  </div>
  {trading_overview}
</div>
{_JS}{dashboard_script}
</body></html>"""

def export_github_pages() -> None:
    GH_PAGES_DIR.mkdir(exist_ok=True)
    (GH_PAGES_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (GH_PAGES_DIR / "index.html").write_text(build_unified_html(static_mode=True), encoding="utf-8")
    for src in [OUT_AN, OUT_BT, OUT_DASH, OUT_TRD, SIGNALS_FILE, REGIME_FILE, POSITIONS_FILE, SCREENER_JSON]:
        if src.exists():
            (GH_PAGES_DIR / src.name).write_bytes(src.read_bytes())

# ── HTTP 서버 ────────────────────────────────────────────────────────
class ZeusHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _json(self, d, st=200):
        body=json.dumps(d,ensure_ascii=False,default=str).encode()
        self.send_response(st)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",len(body))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(body)

    def _html(self, p):
        if not p.exists(): self.send_error(404); return
        body=p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",len(body))
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()

    def do_GET(self):
        path=self.path.split("?")[0]
        qs=self.path[len(path):]
        if path in ("/","/index.html"): self._html(UNIFIED); return
        if path=="/run/job_log":
            jid=qs.split("id=")[-1].split("&")[0] if "id=" in qs else ""
            with _JOB_LOCK: job=_JOBS.get(jid)
            self._json(job.snapshot() if job else
                       {"id":jid,"done":True,"lines":["❌ job 없음"],"result":{}})
            return
        if path.endswith(".html"): self._html(HERE/path.lstrip("/")); return
        self.send_error(404)

    def do_POST(self):
        n=int(self.headers.get("Content-Length",0))
        bd=json.loads(self.rfile.read(n) or b"{}")
        ep=self.path.replace("/run/","").strip("/")
        try:
            if PUBLIC_SITE and ep not in {"analyzer_bg", "load_screener", "load_signals"}:
                self._json({"ok":False,"error":"public_read_only"},403)
                return
            if ep=="dashboard_bg":
                self._json({"job_id":api_dashboard_bg()})
            elif ep=="analyzer_bg":
                tks=[t.strip().upper() for t in bd.get("tickers","").split(",") if t.strip()]
                if PUBLIC_SITE:
                    blocked = _check_public_analyzer_access(self, tks)
                    if blocked:
                        self._json({"ok":False,"error":blocked},429)
                        return
                self._json({"job_id":api_analyzer_bg(tks,bd.get("period","2y"),float(bd.get("account",100_000)))})
            elif ep=="backtest_bg":
                self._json({"job_id":api_backtest_bg(
                    tier_filter=bd.get("tier_filter","all"),
                    mode=bd.get("mode","full"),
                    tickers=bd.get("tickers",""),
                )})
            elif ep=="screener_bg":
                self._json({"job_id":api_screener_bg(bd.get("tickers",""))})
            elif ep=="load_screener":
                self._json(_load_screener())
            elif ep=="commander_bg":
                raw=bd.get("tickers","").strip()
                tks=([t.strip().upper() for t in raw.split(",") if t.strip()] if raw else _load_signals().get("top_tickers",[]))
                self._json({"job_id":api_commander_bg(tks,float(bd.get("capital",100_000)),bd.get("strategy","1"))})
            elif ep=="sentiment_bg":
                raw=bd.get("tickers","").strip()
                tks=[t.strip().upper() for t in raw.split(",") if t.strip()] if raw else []
                self._json({"job_id":api_sentiment_bg(tks)})
            elif ep=="dashboard": self._json(api_dashboard())
            elif ep=="tracker": self._json(api_tracker())
            elif ep=="load_signals": self._json(_load_signals())
            elif ep=="add_position":
                self._json(api_pos_add(bd["ticker"],int(bd["shares"]),float(bd["avg_cost"]),
                                       bd.get("buy_date",date.today().isoformat()),bd.get("note","")))
            elif ep=="remove_position": self._json(api_pos_remove(bd["ticker"]))
            else: self._json({"ok":False,"error":f"unknown: {ep}"},404)
        except Exception as e:
            self._json({"ok":False,"error":str(e),"trace":traceback.format_exc()},500)

# ── 진입점 ──────────────────────────────────────────────────────────
def _open(url):
    """크롬 우선 → 없으면 기본 브라우저"""
    chrome_paths_win = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
    chrome_paths_mac = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    chrome_paths_linux = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    try:
        if sys.platform == "win32":
            for cp in chrome_paths_win:
                if os.path.exists(cp):
                    subprocess.Popen([cp, url], creationflags=0x00000008)
                    return
            # 레지스트리에서 chrome 경로 탐색
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe")
                cp = winreg.QueryValue(key, None)
                if cp and os.path.exists(cp):
                    subprocess.Popen([cp, url], creationflags=0x00000008)
                    return
            except Exception: pass
            # 폴백: start 명령 (기본 브라우저)
            subprocess.Popen(f'start "" "{url}"', shell=True, creationflags=0x00000008)
        elif sys.platform == "darwin":
            for cp in chrome_paths_mac:
                if os.path.exists(cp):
                    subprocess.Popen([cp, url])
                    return
            subprocess.Popen(["open", url])
        else:
            for cp in chrome_paths_linux:
                if os.path.exists(cp):
                    subprocess.Popen([cp, url])
                    return
            subprocess.Popen(["xdg-open", url])
    except Exception: pass

def main():
    live   = "--live"   in sys.argv
    server = "--server" in sys.argv
    print("\n"+"═"*56+"\n  ⚡ ZEUS INTEGRATED DASHBOARD  v2.1\n"+"="*56)
    print("\n  📦 모듈 로딩…"); load_all()
    if _MODS.get("dashboard") and not OUT_DASH.exists():
        print("\n  🌊 Dashboard 초기 생성…"); api_dashboard()
    UNIFIED.write_text(build_unified_html(),encoding="utf-8")
    export_github_pages()
    print(f"\n  💾 통합 HTML: {UNIFIED}")
    print(f"  🌍 GitHub Pages 폴더: {GH_PAGES_DIR}")
    if server:
        bind_host = PUBLIC_BIND_HOST if PUBLIC_SITE else ""
        public_hint = f"http://<your-server-ip-or-domain>:{PORT}" if PUBLIC_SITE else f"http://localhost:{PORT}"
        print(f"\n  🌐 {public_hint}  (Ctrl+C 종료)\n")
        if not PUBLIC_SITE:
            _open(f"http://localhost:{PORT}")
        _stop_event = threading.Event()

        if live:
            def _auto():
                while not _stop_event.wait(REFRESH_SEC):
                    sys.__stdout__.write(f"  🔄 [{datetime.now():%H:%M:%S}] 자동 갱신 시작…\n")
                    sys.__stdout__.flush()
                    try:
                        api_dashboard()
                        UNIFIED.write_text(build_unified_html(), encoding="utf-8")
                        export_github_pages()
                        sys.__stdout__.write(f"  ✅ [{datetime.now():%H:%M:%S}] 갱신 완료 (다음: {REFRESH_SEC//60}분 후)\n")
                        sys.__stdout__.flush()
                    except Exception as e:
                        sys.__stdout__.write(f"  ❌ 갱신 오류: {e}\n")
                        sys.__stdout__.flush()
            threading.Thread(target=_auto, daemon=True).start()

        httpd = _ThreadingHTTPServer((bind_host, PORT), ZeusHandler)
        # Windows에서 Ctrl+C 즉시 감지: poll_interval 짧게 + timeout 설정
        httpd.timeout = 0.5

        print("  Ctrl+C 로 종료\n")
        try:
            httpd.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            _stop_event.set()
            httpd.server_close()
            print("\n  ✅ 서버 종료")
    elif live:
        _open(str(UNIFIED.resolve()))
        try:
            while True:
                time.sleep(REFRESH_SEC)
                sys.__stdout__.write(f"  🔄 [{datetime.now():%H:%M:%S}] 자동 갱신 시작…\n")
                sys.__stdout__.flush()
                api_dashboard()
                UNIFIED.write_text(build_unified_html(), encoding="utf-8")
                export_github_pages()
                sys.__stdout__.write(f"  ✅ [{datetime.now():%H:%M:%S}] 갱신 완료 (다음: {REFRESH_SEC//60}분 후)\n")
                sys.__stdout__.flush()
        except KeyboardInterrupt:
            print("\n  ⏹  종료")
    else:
        print(f"\n  📂 {UNIFIED.resolve()}")
        print("  ※ 버튼 기능은 --server 모드 필요\n     py zeus_integrated_dashboard.py --server\n")
        _open(str(UNIFIED.resolve()))

if __name__=="__main__": main()
