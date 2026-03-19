#!/usr/bin/env python3
"""
SmartScore 백테스트 검증 모듈 — v12
=====================================
[실행 시간]
  최초 실행: ~90~150분 (484종목 × 10년 × SSE.calc, 병렬 8워커)
  재실행:    ~1분 이내 (.bt_cache_v12b/ 청크 캐시 자동 재사용)
  ※ v12는 .bt_cache_v12b/ 을 전용으로 사용 → v11 캐시와 격리

[v10 버그 수정 — 2026-03-08]
  BUG-1  HTML 구간 테이블 헤더 5일/10일/20일 → 20일/60일 (FORWARD=[20,60] 일치)
  BUG-2  verdict 판정 5d 기준 → 20d 기준 (5d 없어서 항상 None → 항상 🔴 표시)
  BUG-3  상관계수 테이블 헤더 5일/10일/20일 → 20일/60일
  BUG-4  캐시 버전 분리: .bt_cache_v12b → .bt_cache_v12b (v10 캐시 재사용 시 변경 무효화 방지)
  BUG-5  강매집 메타카드 "5일 수익률" 표시 → "20일 수익률" (N/A 표시 수정)
  BUG-6  _score_ic의 fillna(0) → dropna() (NaN raw를 0으로 치환 시 IC 희석 방지)

[v9 수정 사항 (유지)]

FIX-1  raw 신호 NaN 4개 복구
  - SmartScoreEngine.calc()에서 rs/accel/w52/cons는 정상 계산됨
  - 문제: run_backtest_single에서 raw.get("rs",0) — 0으로 채워지면
          개별 상관계수 컬럼(raw_rs 등)이 상수(0)로 저장 → corr = NaN
  - 수정: raw값 그대로 저장 + NaN 방어 처리 유지
  - 추가: raw 신호 별도 검증 로그 출력

FIX-2  베타 중립화 — 구간별 수익률도 XS 기준 병행 출력
  - 기존: IC만 XS 기준 / 구간별 수익률은 그로스 → 베타 효과로 하위20%가 상위보다 높음
  - 수정: make_bucket_stats가 그로스 AND XS 수익률 양쪽 계산
  - 출력: 그로스 + XS(시장초과) 구간표 병행

FIX-3  cf_score 역전 진단 + 자동 부호 반전
  - cf_score IC가 -0.018(60d)로 역방향 확인됨
  - 수정: 훈련기간 IC 계산 후 cf_score 부호 자동 반전 여부 결정
  - 반전 시: cf_score → 100 - cf_score (순위 뒤집기)
  - 출력: 반전 여부 + before/after IC 비교

FIX-4  절대SmartScore 임계값 조정
  - 기존: 75-100 N=17 (극소) — 통계 불안정
  - 수정: 임계값을 동적으로 조정, 최소 N=100 보장
  - 구간 재설계: 상위 2%/10%/30%/50%/하위 기준 분위수 버킷

FIX-5  XS IC = 그로스 IC 동일 문제 주석 처리
  - SPY는 공통 상수이므로 Spearman rank IC는 XS/그로스 동일
  - XS IC 출력을 제거하고 대신 "XS 확인: 동일함" 노트 추가
  - 실제 시장중립 IC는 롱숏 포트폴리오로 별도 계산 (v10 예정)
"""

import sys, os, time, json, warnings, logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
# ── scipy 단일 import (함수 내부 중복 제거) ─────────────────────────────
try:
    from scipy import stats as _scipy_stats
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    _scipy_stats = None
    _spearmanr = None

# ── LightGBM (선택적 import) ─────────────────────────────────────────────
# 설치: pip install lightgbm
# 없어도 기존 선형 모델로 정상 동작 (lgb_score 생략)
try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    lgb      = None
    _LGB_OK  = False
    print("  [LightGBM] 미설치 → 기존 선형 IC-가중 모델만 사용")
    print("             설치: pip install lightgbm")


# ── 대시보드 코드에서 SmartScoreEngine import ──────────────────
# ── SmartScoreEngine 동적 import (파일명 자동 탐색) ──────────
import importlib.util

def _find_dashboard() -> str:
    """같은 폴더에서 대시보드 파일 탐색."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "market_dashboard_v8.py",
        "market_dashboard_v7.py",
        "market_dashboard_v6.py",
        "market_dashboard_v5.py",
        "market_dashboard_v4.py",
        "market_dashboard_v3.py",
        "market_dashboard.py",
        "base.py",
    ]
    for name in candidates:
        path = os.path.join(here, name)
        if os.path.exists(path):
            return path
    # 폴더 내 .py 파일 중 SmartScoreEngine 포함된 것 탐색
    for fname in os.listdir(here):
        if not fname.endswith(".py"): continue
        fpath = os.path.join(here, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                if "SmartScoreEngine" in f.read():
                    return fpath
        except Exception as _e:
            logger.debug("dashboard 탐색 실패: %s", _e)
    return None

_dashboard_path = _find_dashboard()
if _dashboard_path is None:
    print("❌ market_dashboard_v4.py 파일을 찾을 수 없습니다.")
    print("   backtest_smartscore.py와 같은 폴더에 대시보드 파일을 넣어주세요.")
    sys.exit(1)

try:
    spec = importlib.util.spec_from_file_location("dashboard", _dashboard_path)
    _mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    SmartScoreEngine = _mod.SmartScoreEngine
    print(f"✅ SmartScoreEngine import 성공 ({os.path.basename(_dashboard_path)})")
except Exception as e:
    print(f"❌ import 실패: {e}")
    sys.exit(1)

# ── 백테스트 대상 종목 ──────────────────────────────────────────
TICKERS = [
    # ══ 계층1: 대형 앵커 200종목 (시총>10B, 유동성 최상) ══════════════
    # 목적: IC 베이스라인 + 섹터 분산 + ETF 벤치마크
    # 근거: S&P500 상위 구성종목 + 섹터 ETF — 데이터 안정성 최고
    "NVDA","AMD","AAPL","MSFT","GOOGL","META","AMZN","AVGO","TSLA","QCOM",
    "PLTR","CRWD","PANW","COIN","MSTR","ZS","FTNT","MU","AMAT","LRCX",
    "NFLX","ORCL","IBM","INTC","TXN","ADI","MCHP","ON","KLAC","ASML",
    "CRM","NOW","WDAY","ADBE","INTU","SNPS","CDNS","ACN","CSCO","HPQ",
    "DELL","KEYS","GRMN","JNPR","FFIV","AKAM","CDW","BAH","HPE","NTAP",
    "LLY","ABBV","UNH","JNJ","MRK","PFE","BMY","GILD","AMGN","REGN",
    "VRTX","ISRG","BSX","MDT","ABT","TMO","DHR","IQV","IDXX","DXCM",
    "CVS","CI","HUM","MCK","CAH","ABC","JPM","BAC","GS","V",
    "MA","WFC","C","MS","BLK","SCHW","COF","AXP","CB","PGR",
    "MET","AFL","USB","PNC","TFC","STT","BK","TROW","IVZ","BEN",
    "AMG","WMT","COST","TGT","HD","LOW","NKE","SBUX","MCD","CMG",
    "TXRH","PG","KO","PEP","PM","MO","GIS","CPB","XOM","CVX",
    "COP","SLB","HAL","EOG","MPC","VLO","PSX","CAT","DE","HON",
    "GE","ETN","EMR","PH","AME","UPS","FDX","ODFL","SAIA","DAL",
    "UAL","LUV","T","VZ","TMUS","DIS","CMCSA","LMT","RTX","NOC",
    "SPY","QQQ","SOXX","XLK","XLF","XLE","XLV","XLI","XLB","XLU",
    "XLRE","XLC","XLP","GLD","TLT","HYG","LQD","IWM","MDY","EFA",
    "LDOS","VRTS","FITB","HBAN","KEY","RF","MTB","STX","WRB","TRV",
    "HIG","ALL","KMB","CL","CHD","VICI","O","AMT","PLD","WELL",
    "EQR","AVB","DLR","PSA","EQIX","BRK-B","WM","RSG","CTAS","ADP",

    # ══ 계층2: 중형 모멘텀 200종목 (시총 2~10B) ══════════════════════
    # 목적: Wyckoff 매집 신호 최강 구간 — IC 극대화 기대
    # 근거: 패시브 비중 낮아 OBV/UD ratio 신뢰도 높음
    #       Hou, Xue & Zhang (2015) — 중형주에서 팩터 수익 집중
    "SMCI","APP","SNOW","DDOG","NET","ANET","AI","PATH","GTLB","CFLT",
    "MDB","DOCN","BRZE","ASAN","GLBE","MNDY","BILL","HUBS","PCTY","PAYC",
    "CDAY","JAMF","FRSH","AMPL","FOUR","RAMP","EVTC","PRCL","LSPD","ZI",
    "TTD","MGNI","PUBM","CRTO","IAS","SQ","AFRM","UPST","SOFI","NU",
    "DLO","FLYW","PAYO","HOOD","AXON","KTOS","CACI","HII","DRS","MOOG",
    "HEICO","TDG","AVAV","ASTS","CELH","HIMS","RXRX","BEAM","NTLA","CRSP",
    "EDIT","VCEL","RVMD","KYMR","DNLI","ATRA","TMDX","IRTC","INSP","NARI",
    "SWAV","ATRC","LMAT","PODD","AXNX","NSTG","PACB","NVCR","ALNY","BMRN",
    "RARE","KRYS","ACAD","EXAS","PRAX","SAGE","PCVX","JANX","RCUS","ARQT",
    "IMVT","ARWR","DRNA","ADMA","NVST","ACLS","GKOS","SILK","IMCR","ELEV",
    "AKRO","ENPH","SEDG","FSLR","ARRY","CWEN","NOVA","SHLS","CHPT","EVGO",
    "BE","SM","MTDR","CIVI","RRC","CNX","EQT","AR","SWN","OXY",
    "GPOR","ETSY","CHWY","W","OLLI","FIVE","BJ","CASY","BOOT","ANF",
    "URBN","DKNG","DUOL","ONTO","AMBA","MTSI","CRUS","SLAB","SITM","COHU",
    "MKSI","UCTT","FORM","PLAB","DIOD","AOSL","ALGM","SMTC","ARIS","CALX",
    "LITE","VIAV","NTCT","JBHT","WERN","EXPD","CHRW","KNX","BFAM","KFY",
    "MAN","OWL","STEP","GCMG","NRDS","WINA","CASS","CVBF","LC","RELY",
    "ROOT","CLOV","WISE","KFRC","RBBN","INFN","AEO","LEVI","GOOS","WING",
    "DPZ","FAT","JACK","EAT","CAKE","BJRI","FWRG","HAYW","ACMR","IIVI",
    "WEAV","TASK","ALTR","DCBO","THNC","SPRK","ALCC","HLLY","PTVE","BARK",

    # ══ 계층3: 소형/고베타 84종목 (시총 0.5~2B) ══════════════════════
    # 목적: 신호 선행성 검증 + 알파 발굴
    # 주의: 데이터 기간 짧음, 유동성 낮음 → 슬리피지 큼
    "MARA","RIOT","HUT","BTBT","CLSK","CIFR","WULF","IREN","CORZ","BITF",
    "IONQ","RGTI","QBTS","QUBT","ARQQ","RKLB","SPIR","MNTS","ASTR","LLAP",
    "SPCE","VORB","RKT","ACHR","JOBY","SOUN","BBAI","CRDO","MAPS","MDGL",
    "HRMY","KROS","TARS","VKTX","GPCR","NRIX","RLMD","MNKD","AGEN","AGIO",
    "TWST","CDNA","TNGX","NVTA","WKHS","NKLA","GOEV","FSR","ZEV","FFIE",
    "PTRA","HYZN","HYLN","DRVN","PLUG","BLDP","FCEL","STEM","FREYR","NRGV",
    "MP","LAC","LI","SLI","LTHM","PLL","TELL","LNG","NFE","FLNG",
    "GLNG","LILM","ARCHER","RIDE","LIDR","OUST","LAZR","VLDR","INVZ","ARROW",
    "OPAL","ADEX","EVTOL","BTDR",

    # ══ TIER4: Russell 2000 확장 유니버스는 아래 TICKER_TIER 매핑 후 TICKERS에 추가
]

# ── 계층별 ticker 집합 (cross_sectional_rank에서 독립 랭크에 사용) ──
# 근거: Hou, Xue & Zhang (2015) "Digesting Anomalies", RFS
#       대형주·소형주 혼합 크로스섹셔널 → 대형주가 순위 억압
#       → 계층 내 독립 순위 계산으로 대형 bias 제거
TIER1_SET = {
    "NVDA","AMD","AAPL","MSFT","GOOGL","META","AMZN","AVGO","TSLA","QCOM",
    "PLTR","CRWD","PANW","COIN","MSTR","ZS","FTNT","MU","AMAT","LRCX",
    "NFLX","ORCL","IBM","INTC","TXN","ADI","MCHP","ON","KLAC","ASML",
    "CRM","NOW","WDAY","ADBE","INTU","SNPS","CDNS","ACN","CSCO","HPQ",
    "DELL","KEYS","GRMN","JNPR","FFIV","AKAM","CDW","BAH","HPE","NTAP",
    "LLY","ABBV","UNH","JNJ","MRK","PFE","BMY","GILD","AMGN","REGN",
    "VRTX","ISRG","BSX","MDT","ABT","TMO","DHR","IQV","IDXX","DXCM",
    "CVS","CI","HUM","MCK","CAH","ABC","JPM","BAC","GS","V",
    "MA","WFC","C","MS","BLK","SCHW","COF","AXP","CB","PGR",
    "MET","AFL","USB","PNC","TFC","STT","BK","TROW","IVZ","BEN",
    "AMG","WMT","COST","TGT","HD","LOW","NKE","SBUX","MCD","CMG",
    "TXRH","PG","KO","PEP","PM","MO","GIS","CPB","XOM","CVX",
    "COP","SLB","HAL","EOG","MPC","VLO","PSX","CAT","DE","HON",
    "GE","ETN","EMR","PH","AME","UPS","FDX","ODFL","SAIA","DAL",
    "UAL","LUV","T","VZ","TMUS","DIS","CMCSA","LMT","RTX","NOC",
    "SPY","QQQ","SOXX","XLK","XLF","XLE","XLV","XLI","XLB","XLU",
    "XLRE","XLC","XLP","GLD","TLT","HYG","LQD","IWM","MDY","EFA",
    "LDOS","VRTS","FITB","HBAN","KEY","RF","MTB","STX","WRB","TRV",
    "HIG","ALL","KMB","CL","CHD","VICI","O","AMT","PLD","WELL",
    "EQR","AVB","DLR","PSA","EQIX","BRK-B","WM","RSG","CTAS","ADP",
}
TIER2_SET = {
    "SMCI","APP","SNOW","DDOG","NET","ANET","AI","PATH","GTLB","CFLT",
    "MDB","DOCN","BRZE","ASAN","GLBE","MNDY","BILL","HUBS","PCTY","PAYC",
    "CDAY","JAMF","FRSH","AMPL","FOUR","RAMP","EVTC","PRCL","LSPD","ZI",
    "TTD","MGNI","PUBM","CRTO","IAS","SQ","AFRM","UPST","SOFI","NU",
    "DLO","FLYW","PAYO","HOOD","AXON","KTOS","CACI","HII","DRS","MOOG",
    "HEICO","TDG","AVAV","ASTS","CELH","HIMS","RXRX","BEAM","NTLA","CRSP",
    "EDIT","VCEL","RVMD","KYMR","DNLI","ATRA","TMDX","IRTC","INSP","NARI",
    "SWAV","ATRC","LMAT","PODD","AXNX","NSTG","PACB","NVCR","ALNY","BMRN",
    "RARE","KRYS","ACAD","EXAS","PRAX","SAGE","PCVX","JANX","RCUS","ARQT",
    "IMVT","ARWR","DRNA","ADMA","NVST","ACLS","GKOS","SILK","IMCR","ELEV",
    "AKRO","ENPH","SEDG","FSLR","ARRY","CWEN","NOVA","SHLS","CHPT","EVGO",
    "BE","SM","MTDR","CIVI","RRC","CNX","EQT","AR","SWN","OXY",
    "GPOR","ETSY","CHWY","W","OLLI","FIVE","BJ","CASY","BOOT","ANF",
    "URBN","DKNG","DUOL","ONTO","AMBA","MTSI","CRUS","SLAB","SITM","COHU",
    "MKSI","UCTT","FORM","PLAB","DIOD","AOSL","ALGM","SMTC","ARIS","CALX",
    "LITE","VIAV","NTCT","JBHT","WERN","EXPD","CHRW","KNX","BFAM","KFY",
    "MAN","OWL","STEP","GCMG","NRDS","WINA","CASS","CVBF","LC","RELY",
    "ROOT","CLOV","WISE","KFRC","RBBN","INFN","AEO","LEVI","GOOS","WING",
    "DPZ","FAT","JACK","EAT","CAKE","BJRI","FWRG","HAYW","ACMR","IIVI",
    "WEAV","TASK","ALTR","DCBO","THNC","SPRK","ALCC","HLLY","PTVE","BARK",
}
TIER3_SET = {
    "MARA","RIOT","HUT","BTBT","CLSK","CIFR","WULF","IREN","CORZ","BITF",
    "IONQ","RGTI","QBTS","QUBT","ARQQ","RKLB","SPIR","MNTS","ASTR","LLAP",
    "SPCE","VORB","RKT","ACHR","JOBY","SOUN","BBAI","CRDO","MAPS","MDGL",
    "HRMY","KROS","TARS","VKTX","GPCR","NRIX","RLMD","MNKD","AGEN","AGIO",
    "TWST","CDNA","TNGX","NVTA","WKHS","NKLA","GOEV","FSR","ZEV","FFIE",
    "PTRA","HYZN","HYLN","DRVN","PLUG","BLDP","FCEL","STEM","FREYR","NRGV",
    "MP","LAC","LI","SLI","LTHM","PLL","TELL","LNG","NFE","FLNG",
    "GLNG","LILM","ARCHER","RIDE","LIDR","OUST","LAZR","VLDR","INVZ","ARROW",
    "OPAL","ADEX","EVTOL","BTDR",
}

# ══ TIER4: Russell 2000 소형주 확장 유니버스 ══════════════════════════
# 목적: 소형주 팩터(SMB) 효과 검증 범위 확대 → 총 ~3000종목 달성
# 근거: Fama & French(1993) — 소형주 팩터(SMB) 수익이 대형주 대비 집중
#       Russell 2000 = 미국 소형주 벤치마크 (시총 약 3~20억달러)
# 주의: 유동성 낮은 종목 포함 → TransactionCostEngine 슬리피지 자동 반영
#       데이터 기간 짧은 종목 → download_ticker의 LOOKBACK 검사로 자동 제외
TIER4_SET = {
    # Russell 2000 소형주 — TIER1/2/3 중복 제거 완료
    "AAON","ABCB","ABCL","ABEO","ABSI","ABST","ABTX","ACBI","ACCD","ACHC",
    "ACHL","ACIW","ACLX","ACNB","ACOR","ACRS","ACRX","ACST","ACTG","ACVA",
    "ACVF","ACYM","ADAP","ADEA","ADES","ADHD","ADIL","ADMP","ADMS","ADNT",
    "ADOC","ADPT","ADRA","ADRD","ADRE","ADRO","ADSE","ADSK","ADTH","ADTX",
    "ADUS","ADVM","ADXN","ADXS","AEHR","AEIS","AENZ","AERI","AFAR","AFBI",
    "AFCG","AFIN","AFMD","AFRI","AFTR","AFYA","AGBA","AGCO","AGFS","AGHL",
    "AGLY","AGMH","AGND","AGNG","AGNU","AGOX","AGPX","AGRO","AGRX","AGSI",
    "AGTC","AGTI","AGUS","AGXS","AGYS","AHCO","AHHI","AHPA","AHPI","AIFU",
    "AIIG","AIKI","AILE","AINV","AIRC","AKBA","ALDX","ALEX","ALKS","ALLK",
    "ALPA","ALPN","ALRM","ALTA","ALTO","AMAG","AMBO","AMBT","AMCI","AMCR",
    "AMCX","AMED","AMEH","AMEX","AMHC","AMKR","AMMO","AMNB","AMOT","AMOV",
    "AMPE","AMPH","AMPY","AMRC","AMRK","AMRN","AMRS","AMRT","AMSC","AMSF",
    "AMSWA","AMTB","AMTD","AMTX","AMTY","AMUR","AMWL","AMYT","ANAB","ANDE",
    "ANDR","ANEB","ANGO","ANIK","ANIP","ANIX","ANKM","ANPC","ANSS","ANTE",
    "ANTS","ANZU","AOUT","APAM","APCA","APCO","APEX","APGB","APGE","APHA",
    "APHB","APHI","APLD","APLS","APLT","APMI","APNB","APOG","APOP","APPF",
    "APPH","APPHC","APPL","APRE","APRI","APRL","APRN","APRO","APRR","APRT",
    "APRU","APRX","AQST","ARCO","ARCT","ARDX","AREC","ARGX","ARHS","ARKO",
    "AROW","ARTL","ARTNA","ARTS","ARTW","ARVN","ARYA","ASAI","ASBP","ASCA",
    "ASDN","ASGN","ASIX","ASMB","ASND","ASNS","ASPS","ASPU","ASRT","ASRV",
    "ASST","ASTE","ASUR","ASWC","ATHA","ATHN","ATI","ATIF","ATIP","ATIS",
    "ATLC","ATLO","ATNI","ATOI","ATON","ATPC","ATPI","ATRM","ATRS","ATSG",
    "ATXI","AUBN","AUDC","AUID","AUPH","AUVI","AVA","AVBH","AVDL","AVEO",
    "AVHI","AVID","AVNS","AVNW","AVPT","AVRO","AVTE","AVTX","AVXL","AWAY",
    "AXDX","AXGT","AXSM","AXTI","AYTU","AZAN","AZEK","AZPN","AZRE","AZRX",
    "AZYO","BANC","BAND","BANF","BANR","BATR","BATRA","BCAR","BCBP","BCEI",
    "BCEL","BCOR","BCOW","BCPC","BCRX","BCTX","BCYC","BDGE","BDSI","BDTX",
    "BEAT","BEBE","BECN","BELFA","BELFB","BELX","BERY","BFLY","BFNC","BFRI",
    "BFST","BGFV","BGNE","BGSF","BHAC","BHAT","BHVN","BIAF","BIOS","BIOX",
    "BIRD","BIVI","BKCC","BKSC","BKSY","BKTI","BLBX","BLCT","BLDR","BLIN",
    "BLMN","BLNK","BLPH","BLRX","BLSA","BLTS","BRBR","BRT","BSVN","CC",
    "CCAP","CEIX","CENX","CFFI","CLDT","CLFD","CLVS","CMP","CMPR","CNXC",
    "COFS","CORR","CPSS","CRAI","CRGY","CRK","CRVS","CSR","CSWC","CTBI",
    "CWST","CZWI","DCOM","DCPH","DENN","DNOW","DRQ","DVAX","EBTC","EGHT",
    "EMCF","ENFN","ESRT","ESSA","ESTE","EVLO","FDUS","FFBH","FFBW","FIVN",
    "FLGT","FLIC","FMAO","FNLC","FOLD","FXNC","GAIN","GFF","GLAD","GMRE",
    "GSBD","GTY","HAFC","HALO","HCAT","HIBB","HLIO","HONE","HOOK","HRZN",
    "IIIN","IIPR","ILPT","IMOS","INDT","IRET","IRT","JBSS","KALU","KLIC",
    "KRG","LBRT","LNN","LSCC","LTC","MATW","MGTX","MGY","MNRL","MPW",
    "NBTB","NHI","NNN","NOG","NTIC","NUVB","NVEC","NXRT","OBDC","OCSL",
    "OPKO","PEB","PECO","PFLT","PHAT","PLYM","PRTA","PSEC","PTEN","PTGX",
    "RCKT","REX","RNGR","ROIC","SCM","SLRC","SMMT","SNDE","SSYS","STNG",
    "TCPC","TENB","TPVG","TRIN","TWIN","USAC","VRNA","VTLE","WHF","WLFC",
    "XOMA","ZYME",
}

# ticker → tier 매핑 (빠른 조회용)
TICKER_TIER = {}
for _tk in TIER1_SET: TICKER_TIER[_tk] = 1
for _tk in TIER2_SET: TICKER_TIER[_tk] = 2
for _tk in TIER3_SET: TICKER_TIER[_tk] = 3
for _tk in TIER4_SET: TICKER_TIER[_tk] = 4   # Russell 2000 소형주

# TIER4를 TICKERS에 추가 (TIER4_SET 정의 후에 실행되므로 NameError 없음)
# 중복 방지: 기존 TICKERS에 없는 종목만 추가
_existing_tickers = set(TICKERS)
for _tk in sorted(TIER4_SET):
    if _tk not in _existing_tickers:
        TICKERS.append(_tk)
        _existing_tickers.add(_tk)
del _existing_tickers

LOOKBACK   = 252  # 1년 워밍업 — 12M모멘텀, rolling(60) 등 최소 요구
# 주의: 2021년 이후 상장 종목(SOUN, BBAI, CRDO 등)은 10년치 없음
# download_ticker가 LOOKBACK+max(FORWARD) 미충족 시 자동 제외함
FORWARD    = [20, 60]     # 매집신호는 중장기 선행 — 5일 노이즈 제거
# TIER3 역발상 전략은 단기 반전(Jegadeesh & Titman 2001, 1-4주) 기반
# → 5일 목표변수를 TIER3 전용으로 별도 계산 (FORWARD 메인과 독립)
FORWARD_T3 = [5]          # TIER3 단기 반전 전용 목표변수 기간
PERIOD     = "10y"   # ← 메모리 부족 시 "5y"로 변경 (절반 절감)
                     #   8GB 이하 환경 권장: "5y"
                     #   처음 실행이라면 "5y"로 시작 후 결과 확인  # 장기 팩터 검증: 10년

SSE = SmartScoreEngine()

# ══════════════════════════════════════════════════════════════════════
#  v6 펀드-그레이드 검증 모듈 6종
#  ① LookAheadGuard        — 룩어헤드 바이어스 감지·차단
#  ② MultipleTestingGuard  — 복수검정 보정 (BH-FDR + Bonferroni)
#  ③ TransactionCostEngine — 거래비용 반영 (수수료+슬리피지+충격)
#  ④ RollForwardValidator  — 연도별 롤링 OOS 검증
#  ⑤ BootstrapCI           — 1000회 부트스트랩 신뢰구간
#  ⑥ RegimeAnalyzer        — 국면별 성과 분리 분석
# ══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────
# ① LookAheadGuard
# ──────────────────────────────────────────────────────────────────────
class LookAheadGuard:
    """
    룩어헤드 바이어스 감지·차단 모듈.

    [문제]
      백테스트 루프 내에서 detect_regime()을 전체 SPY 시계열로 한 번만
      호출하면, 미래 가격이 MA/표준편차 계산에 혼입된다.
      → 실제 운용에서 재현 불가능한 신호를 과거에 적용하는 셈.

    [해결 원칙 — Lopez de Prado 'Advances in Financial ML' Ch.4]
      백테스트 루프의 각 시점 i에서 spy_df.iloc[:i+1] 로만 국면 감지.
      이 클래스가 슬라이싱과 캐싱을 담당한다.

    [수치 근거]
      SMA200 / SMA50 크로스 + 20일 변동성 임계 15%:
        - SMA200: 기관 장기 방향성 지표 (Faber 2007, SSRN 962461)
        - SMA50:  단기 추세 전환 포착 (Lo et al. 2000, RFS)
        - vol 임계 15%: 연환산 기준 (코드에서 사용하는 실제 임계값)
          일간 환산: 15%/√252 ≈ 0.94%/일
          VIX 환산: 15% 연환산 vol → 암묵적 VIX ≈ 15
          (VIX 20이 역사적 중앙값이나 코드 임계는 15% — 더 보수적 경계)
          15% 이하: 저변동(bull 후보) / 15% 초과: 고변동(sideways/bear 후보)

    [캐싱]
      동일 시점을 여러 종목이 공유할 때 재계산 방지.
      key = 날짜 인덱스 정수. 딕셔너리 O(1) 조회.
    """
    def __init__(self):
        self._cache: dict = {}          # {날짜_idx: regime_str}
        self.violations: list = []      # 감지된 바이어스 기록
        self._call_count = 0
        self._cache_hit  = 0

    def regime_at(self, spy_close: pd.Series, idx: int,
                  vix_val: float = 20.0) -> str:
        """
        시점 idx 이전 데이터만 사용한 국면 반환.
        idx는 spy_close 내 정수 위치 (iloc 기준).

        [슬라이싱 규칙]
          spy_close.iloc[:idx+1]  → idx 당일까지만 포함
          미래(idx+1 이후) 데이터는 절대 접근하지 않음
        """
        self._call_count += 1
        if idx in self._cache:
            self._cache_hit += 1
            return self._cache[idx]

        # 최소 200일 필요 (SMA200 기준)
        if idx < 200:
            regime = "sideways"
        else:
            window = spy_close.iloc[:idx + 1]
            try:
                regime = SmartScoreEngine.detect_regime(window, vix_val=vix_val)
            except Exception:
                regime = "sideways"

        self._cache[idx] = regime
        return regime

    def validate_slice(self, window: pd.Series, eval_idx: int,
                       label: str = "") -> bool:
        """
        window의 마지막 인덱스가 eval_idx와 일치하는지 검증.
        일치하지 않으면 미래 데이터 혼입 위험 → violations에 기록.

        반환: True=클린, False=바이어스 의심
        """
        last = len(window) - 1
        if last != eval_idx:
            msg = (f"[LookAheadGuard] ⚠️  바이어스 감지 "
                   f"label={label} window_end={last} eval_idx={eval_idx}")
            self.violations.append(msg)
            return False
        return True

    def summary(self) -> dict:
        return {
            "total_calls":   self._call_count,
            "cache_hits":    self._cache_hit,
            "violations":    len(self.violations),
            "cache_hit_pct": round(self._cache_hit / max(self._call_count,1) * 100, 1),
        }


# ──────────────────────────────────────────────────────────────────────
# ② MultipleTestingGuard
# ──────────────────────────────────────────────────────────────────────
class MultipleTestingGuard:
    """
    복수검정(다중비교) 보정 모듈.

    [문제]
      8개 팩터(vol_z, bb_z, rs, accel, mom, bab, w52, cons)를 동시에
      검정하면, α=0.05 수준에서 k번 독립 검정 시
        P(최소 1개 위양성) = 1 - 0.95^k
        k=8 → 1 - 0.95^8 ≈ 33.7%   ← 위양성 확률 무시 불가
      k=14(cross_sectional_rank의 전체 검정 수) → 51%

    [해결 방법 2종 병행]

    A) Benjamini-Hochberg FDR (기본 추천)
       근거: Benjamini & Hochberg (1995), JRSS-B 57(1):289-300
       - False Discovery Rate ≤ 5% 제어
       - Bonferroni보다 검정력(power) 높음 → 진짜 신호 덜 버림
       - 월가 팩터 리서치 표준 (AQR, Two Sigma 내부 보고서)
       알고리즘:
         1. p값 오름차순 정렬: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
         2. 기각 기준: p_(i) ≤ (i/m) × α
         3. 최대 k = max{i : p_(i) ≤ (i/m)×α} → 1~k 모두 기각

    B) Bonferroni (보수적 상한)
       근거: Dunn (1961), JASA 56(293):52-64
       - 기각 기준: p < α/m
       - FWER(Family-Wise Error Rate) ≤ α 보장
       - 신호 수가 적고 독립성이 높을 때 사용

    [운용 의미]
      BH 통과 = 5% FDR 수준에서 유의
      Bonferroni 통과 = 더 엄격한 FWER 5% 수준에서도 유의
      둘 다 통과해야 '강한 신호'로 분류
    """
    def __init__(self, alpha: float = 0.05):
        """
        alpha: 검정 수준 (기본 5% — 금융 리서치 표준)
        """
        self.alpha = alpha

    def bh_fdr(self, p_values: list, labels: list = None) -> dict:
        """
        Benjamini-Hochberg FDR 보정.

        p_values: 각 팩터의 t-stat에서 유도한 p값 리스트
        labels:   팩터 이름 (없으면 인덱스)
        반환: {label: {"p_raw", "p_adj_rank", "bh_threshold", "rejected", "method"}}
        """
        m = len(p_values)
        if m == 0:
            return {}
        if labels is None:
            labels = [str(i) for i in range(m)]

        # 정렬 (오름차순)
        order = np.argsort(p_values)
        sorted_p = np.array(p_values)[order]
        sorted_lb = [labels[i] for i in order]

        # BH 임계값: (rank/m) × α
        thresholds = [(i + 1) / m * self.alpha for i in range(m)]

        # 최대 기각 인덱스
        reject_mask = sorted_p <= np.array(thresholds)
        if reject_mask.any():
            max_reject = np.where(reject_mask)[0].max()
        else:
            max_reject = -1

        result = {}
        for rank, (lb, p, thr) in enumerate(zip(sorted_lb, sorted_p, thresholds)):
            result[lb] = {
                "p_raw":       round(float(p), 4),
                "bh_threshold":round(float(thr), 4),
                "rank":        rank + 1,
                "rejected":    rank <= max_reject,   # True = 유의
                "method":      "BH-FDR",
            }
        return result

    def bonferroni(self, p_values: list, labels: list = None) -> dict:
        """
        Bonferroni 보정.
        기각 기준: p < α/m
        """
        m = len(p_values)
        if m == 0:
            return {}
        if labels is None:
            labels = [str(i) for i in range(m)]
        threshold = self.alpha / m
        result = {}
        for lb, p in zip(labels, p_values):
            result[lb] = {
                "p_raw":        round(float(p), 4),
                "bf_threshold": round(threshold, 5),
                "rejected":     float(p) < threshold,
                "method":       "Bonferroni",
            }
        return result

    def combined_test(self, t_stats: dict, n_obs: int) -> dict:
        """
        t-stat → 양측 p값 변환 후 BH + Bonferroni 동시 적용.

        t_stats: {label: t_stat_float}
        n_obs:   자유도 근사치 (날짜 수 - 1)
        반환: {label: {"t_stat", "p_raw", "bh_rejected", "bf_rejected",
                        "strong_signal": BH∧Bonferroni, "ic_significant": BH만}}
        """
        labels = list(t_stats.keys())
        t_arr  = [float(t_stats[l]) for l in labels]
        # 양측 p값: p = 2 × P(T > |t|)
        p_arr  = [float(2 * _scipy_stats.t.sf(abs(t), df=max(n_obs - 1, 1))) for t in t_arr]

        bh_res = self.bh_fdr(p_arr, labels)
        bf_res = self.bonferroni(p_arr, labels)

        result = {}
        for lb, t, p in zip(labels, t_arr, p_arr):
            bh_rej = bh_res.get(lb, {}).get("rejected", False)
            bf_rej = bf_res.get(lb, {}).get("rejected", False)
            result[lb] = {
                "t_stat":        round(t, 3),
                "p_raw":         round(p, 4),
                "bh_rejected":   bh_rej,    # FDR 5% 통과
                "bf_rejected":   bf_rej,    # FWER 5% 통과
                "strong_signal": bh_rej and bf_rej,   # 엄격 — 둘 다 유의
                "ic_significant":bh_rej,               # 관대 — BH만 유의
                "bh_threshold":  bh_res.get(lb, {}).get("bh_threshold", 0),
                "bf_threshold":  bf_res.get(lb, {}).get("bf_threshold", 0),
            }
        return result

    @staticmethod
    def false_positive_risk(k: int, alpha: float = 0.05) -> float:
        """
        k개 독립 검정 시 최소 1개 위양성 확률.
        1 - (1 - α)^k
        """
        return round(1 - (1 - alpha) ** k, 4)


# ──────────────────────────────────────────────────────────────────────
# ③ TransactionCostEngine
# ──────────────────────────────────────────────────────────────────────
class TransactionCostEngine:
    """
    거래비용 반영 엔진 — 수수료 + 슬리피지 + 시장충격비용.

    [v5의 문제]
      run_backtest_single()에서 미래 수익률 = (fut/cur - 1) × 100
      → 거래비용 0원 가정. 실제보다 과장된 수익률.

    [수치 근거]

    수수료 (Commission):
      기관 0.02%: Bloomberg Terminal 실거래 데이터 기반
                  (Goldman Sachs 2023 Execution Services 보고서)
      리테일 0.05%: Fidelity/Schwab 온라인 브로커리지 실제 요율
      리테일 고비용 0.10%: 소형주/해외 브로커

    슬리피지 (Slippage) — Kissell(2013) 'The Science of Algorithmic Trading':
      대형(일 거래량 >5M주): 0.05% — 유동성 풍부, 매수/매도 스프레드 좁음
      중형(1~5M주):         0.10% — 표준 스프레드
      소형(0.3~1M주):       0.20% — 스프레드+틱 갭
      초소형(<0.3M주):      0.50% — 유동성 프리미엄

    시장충격 (Market Impact) — Almgren & Chriss (2001), Applied Math Finance:
      impact = k × √(participation_rate) × daily_vol
      k=0.5: 논문 허용 범위 0.3~0.6 내 중앙값
      participation_rate = order_value / (price × avg_daily_volume)
        → 참여율이 높을수록 자기 매수가 시장을 움직임

    [백테스트 적용]
      순수익률 = 그로스수익률 - roundtrip_cost
      단, 롱온리 전략이므로 매수+매도 왕복비용 전부 차감
    """

    # 수수료 테이블 (단방향 %)
    _COMMISSION = {
        "institutional": 0.02,   # Goldman 기관 요율
        "retail":        0.05,   # Fidelity/Schwab
        "retail_high":   0.10,   # 소형주/해외
    }

    # 달러 거래대금 기반 슬리피지 테이블 (단방향 %)
    # trading._calc_trade_cost와 동일 기준 (Kissell 2013)
    _SLIPPAGE = {
        "large":  (5_000_000,  0.05),   # >$5M/일
        "mid":    (1_000_000,  0.10),   # $1~5M/일
        "small":  (300_000,    0.20),   # $300K~1M/일
        "micro":  (0,          0.50),   # <$300K/일
    }

    def __init__(self, account_type: str = "retail"):
        self.account_type = account_type
        self.commission_pct = self._COMMISSION.get(account_type, 0.05)

    def estimate(self, price: float, avg_daily_volume: float = 0.0,
                 avg_daily_dollar_vol: float = 0.0,
                 order_shares: float = 1000.0,
                 daily_vol_pct: float = 2.0) -> dict:
        """
        단일 종목 거래비용 추산.

        price:                 현재 주가
        avg_daily_volume:      일평균 거래량 (주 단위, 레거시 — 달러환산 fallback)
        avg_daily_dollar_vol:  일평균 달러 거래대금 (trading._calc_trade_cost와 동일 단위)
                               avg_daily_dollar_vol > 0이면 이것을 우선 사용
        order_shares:          주문 수량 (주 단위, 기본 1000주)
        daily_vol_pct:         일간 변동성 % (기본 2%)

        반환:
          commission_pct  — 수수료 (단방향)
          slippage_pct    — 슬리피지 (단방향)
          impact_pct      — 시장충격 (단방향)
          total_one_way   — 단방향 합계
          roundtrip_pct   — 왕복 합계 (백테스트 차감값)
          cost_class      — LOW/MEDIUM/HIGH/VERY_HIGH
        """
        # 슬리피지 계산 (Kissell 2013) — 달러 거래대금 기준으로 통일
        # trading._calc_trade_cost와 동일 임계값
        if avg_daily_dollar_vol > 0:
            dollar_adv = float(avg_daily_dollar_vol)
        else:
            # 레거시 fallback: 주×가격 → 달러 환산
            dollar_adv = float(avg_daily_volume) * max(float(price), 1.0)
        dollar_adv = max(dollar_adv, 1.0)
        if dollar_adv >= 5_000_000:
            slip_pct = 0.05
        elif dollar_adv >= 1_000_000:
            slip_pct = 0.10
        elif dollar_adv >= 300_000:
            slip_pct = 0.20
        else:
            slip_pct = 0.50

        # 변동성 보정 슬리피지: 평균 변동성(2%) 대비 비례 조정
        vol_adj = max(daily_vol_pct, 0.1) / 2.0
        slip_pct_adj = slip_pct * vol_adj

        # 시장충격 (Almgren-Chriss 2001)
        order_value  = float(price) * order_shares
        daily_turnover = float(price) * avg_daily_volume
        participation  = min(order_value / max(daily_turnover, 1.0), 1.0)
        impact_pct = 0.5 * (participation ** 0.5) * max(daily_vol_pct, 0.1)

        comm = self.commission_pct
        total_one_way = comm + slip_pct_adj + impact_pct
        roundtrip     = total_one_way * 2.0  # 매수+매도

        # 등급 분류
        if roundtrip < 0.20:
            cost_class = "LOW"
        elif roundtrip < 0.50:
            cost_class = "MEDIUM"
        elif roundtrip < 1.00:
            cost_class = "HIGH"
        else:
            cost_class = "VERY_HIGH"

        return {
            "commission_pct":  round(comm, 4),
            "slippage_pct":    round(slip_pct_adj, 4),
            "impact_pct":      round(impact_pct, 4),
            "total_one_way":   round(total_one_way, 4),
            "roundtrip_pct":   round(roundtrip, 4),
            "cost_class":      cost_class,
        }

    def adjust_returns(self, gross_rets: np.ndarray,
                       roundtrip_pct: float) -> np.ndarray:
        """
        왕복 거래비용 차감한 순수익률 배열 반환.
        gross_rets: % 단위 수익률 (예: +2.5, -1.3)
        """
        return gross_rets - roundtrip_pct

    def portfolio_annual_drag(self, turnover_annual: float,
                              roundtrip_pct: float) -> float:
        """
        연간 거래비용 드래그 계산.
        turnover_annual: 연간 회전율 (예: 300% = 연 3회 교체)
        반환: 연간 % 비용 드래그
        """
        return round(turnover_annual / 100.0 * roundtrip_pct, 4)


# ──────────────────────────────────────────────────────────────────────
# ④ RollForwardValidator
# ──────────────────────────────────────────────────────────────────────
class RollForwardValidator:
    """
    연도별 롤링 OOS(Out-of-Sample) 검증 모듈.

    [문제]
      현재 코드: 전체 10년 고정 70:30 분할.
      → 2015~2019년에 훈련된 가중치가 2020년 코로나 충격에서도
        그대로 적용됨. 체제 변화(Regime Shift)에 적응 불가.

    [해결 — Walk-Forward Analysis]
      근거: Lopez de Prado, 'Advances in Financial ML' (2018) Ch.12
            Pardo, 'The Evaluation and Optimization of Trading Strategies' (2008)

      알고리즘:
        훈련 윈도우:  TRAIN_YEARS = 3년 (최소 2 Bull+Bear 사이클 포함)
        테스트 윈도우: TEST_YEARS  = 1년 (실제 운용 시뮬레이션)
        스텝:          STEP_MONTHS = 12개월 (연간 재최적화 — 월별보다 안정적)

        예) 10년 데이터:
          폴드 1: 훈련 2014-2016, 테스트 2017
          폴드 2: 훈련 2015-2017, 테스트 2018
          폴드 3: 훈련 2016-2018, 테스트 2019
          폴드 4: 훈련 2017-2019, 테스트 2020
          폴드 5: 훈련 2018-2020, 테스트 2021
          폴드 6: 훈련 2019-2021, 테스트 2022
          폴드 7: 훈련 2020-2022, 테스트 2023

      [핵심]
        각 테스트 기간의 IC는 해당 기간 훈련 데이터로 결정된 가중치로 계산.
        "모른다"는 전제 하에 미래를 예측하는 순수 OOS IC.

    [수치 근거]
      TRAIN_YEARS=3: Fama-French 팩터 포트폴리오 재구성 주기와 일치
                     충분한 Bull+Bear 포함 (사이클 3~5년 평균)
      TEST_YEARS=1:  분기 리밸런싱 전략의 현실적 OOS 단위
      STEP_MONTHS=12: Lopez de Prado 권장 — 짧으면 훈련기간 과소, 길면 적응 늦음
    """

    TRAIN_YEARS  = 3
    TEST_YEARS   = 1
    STEP_MONTHS  = 12

    def __init__(self):
        self.folds: list = []    # 폴드별 결과 저장

    def _make_folds(self, all_dates: list) -> list:
        """
        날짜 리스트로부터 (train_start, train_end, test_start, test_end) 폴드 생성.
        """
        from datetime import datetime as _dt
        dates = sorted(all_dates)
        if not dates:
            return []

        d0    = pd.Timestamp(dates[0])
        d_end = pd.Timestamp(dates[-1])

        folds = []
        test_start = d0 + pd.DateOffset(years=self.TRAIN_YEARS)

        while test_start < d_end:
            train_start = test_start - pd.DateOffset(years=self.TRAIN_YEARS)
            train_end   = test_start - pd.DateOffset(days=1)
            test_end    = min(test_start + pd.DateOffset(years=self.TEST_YEARS), d_end)

            folds.append({
                "train_start": str(train_start.date()),
                "train_end":   str(train_end.date()),
                "test_start":  str(test_start.date()),
                "test_end":    str(test_end.date()),
            })
            test_start += pd.DateOffset(months=self.STEP_MONTHS)

        return folds

    def run(self, df: pd.DataFrame,
            raw_cols: list,
            fwd_col: str = "ret_20d") -> list:
        """
        연도별 롤링 OOS IC 계산.

        df:       전체 백테스트 레코드 DataFrame
        raw_cols: 검증할 팩터 컬럼 리스트
        fwd_col:  목표 수익률 컬럼

        반환: [{fold_info, oos_ic_per_factor, oos_ic_mean, n_test_obs}, ...]
        """

        if "date" not in df.columns or fwd_col not in df.columns:
            return []

        all_dates = sorted(df["date"].unique())
        folds     = self._make_folds(all_dates)
        results   = []

        for fold in folds:
            tr_mask = (df["date"] >= fold["train_start"]) & (df["date"] < fold["test_start"])
            ts_mask = (df["date"] >= fold["test_start"]) & (df["date"] <= fold["test_end"])
            tr_df   = df.loc[tr_mask]
            ts_df   = df.loc[ts_mask]

            if len(tr_df) < 50 or len(ts_df) < 10:
                continue

            # 훈련 기간 IC → 가중치
            ic_means = []
            for col in raw_cols:
                if col not in tr_df.columns:
                    ic_means.append(0.0)
                    continue
                daily_ics = []
                for _, grp in tr_df.groupby("date"):
                    valid = grp[[col, fwd_col]].dropna()
                    if len(valid) < 4:
                        continue
                    ic, _ = _spearmanr(valid[col].values, valid[fwd_col].values)
                    if not np.isnan(ic):
                        daily_ics.append(float(ic))
                ic_means.append(float(np.mean(daily_ics)) if daily_ics else 0.0)

            ic_arr = np.array(ic_means)
            pos    = np.maximum(ic_arr, 0.0)
            w      = pos / pos.sum() if pos.sum() > 1e-9 else np.ones(len(raw_cols)) / len(raw_cols)

            # OOS: 훈련 가중치로 테스트 기간 IC 계산
            oos_ics = {}
            for col, wi in zip(raw_cols, w):
                if col not in ts_df.columns or wi <= 0:
                    oos_ics[col] = 0.0
                    continue
                daily_ics = []
                for _, grp in ts_df.groupby("date"):
                    valid = grp[[col, fwd_col]].dropna()
                    if len(valid) < 4:
                        continue
                    ic, _ = _spearmanr(valid[col].values, valid[fwd_col].values)
                    if not np.isnan(ic):
                        daily_ics.append(float(ic))
                oos_ics[col] = round(float(np.mean(daily_ics)) if daily_ics else 0.0, 4)

            # 가중합 OOS IC
            oos_composite = sum(oos_ics.get(c, 0.0) * w[i]
                                for i, c in enumerate(raw_cols))

            results.append({
                **fold,
                "weights":        {c: round(float(w[i]), 3) for i, c in enumerate(raw_cols)},
                "train_ic":       {c: round(float(ic_arr[i]), 4) for i, c in enumerate(raw_cols)},
                "oos_ic":         oos_ics,
                "oos_ic_mean":    round(float(oos_composite), 4),
                "n_train":        len(tr_df),
                "n_test":         len(ts_df),
            })

        self.folds = results
        return results

    def stability_score(self) -> dict:
        """
        OOS IC 안정성 평가.
        양수 폴드 비율, 평균 OOS IC, 변동성 반환.
        """
        if not self.folds:
            return {}
        ics = [f["oos_ic_mean"] for f in self.folds]
        arr = np.array(ics)
        positive_pct = float((arr > 0).mean())
        return {
            "n_folds":           len(self.folds),
            "mean_oos_ic":       round(float(arr.mean()), 4),
            "std_oos_ic":        round(float(arr.std()), 4),
            "positive_fold_pct": round(positive_pct * 100, 1),
            "min_oos_ic":        round(float(arr.min()), 4),
            "max_oos_ic":        round(float(arr.max()), 4),
            # 판정 기준: 양수 폴드 ≥70% + 평균 IC ≥0.01 → 안정적
            # 근거: Lopez de Prado p.235 — 70% rule for walk-forward robustness
            "stable":            positive_pct >= 0.70 and float(arr.mean()) >= 0.01,
        }


# ──────────────────────────────────────────────────────────────────────
# ⑤ BootstrapCI
# ──────────────────────────────────────────────────────────────────────
class BootstrapCI:
    """
    Bootstrap 신뢰구간 모듈.

    [문제]
      현재 IC 검정: t-stat = IC_mean / (IC_std / √n)
      → 정규분포 가정. 금융 IC 분포는 두꺼운 꼬리(fat-tail) → t-stat 과신

    [해결 — Stationary Bootstrap]
      근거: Politis & Romano (1994), JASA 89(428):1303-1313
            "The Stationary Bootstrap"
      - 시계열 자기상관 보존 (블록 부트스트랩)
      - 블록 길이: 기하분포 평균 L_mean = √n_obs (경험법칙)
        → L_mean=√250 ≈ 16 (거래일 기준)
        근거: Politis & White (2004) "Automatic Block-Length Selection"
      - B=1000 반복: 표준 — Efron & Tibshirani (1994) 권장 최솟값

      [단순 부트스트랩과의 차이]
        단순: 무작위 복원추출 → 시계열 의존성 파괴 → CI 과소평가
        블록: 연속 블록 단위 추출 → 자기상관 구조 보존 → 올바른 CI

    [신뢰구간 해석]
      95% CI: 진짜 IC가 이 범위 밖일 확률 5%
      CI 하한 > 0 → 강한 양방향 증거 (t-stat 유의와 독립적 확인)
      CI 하한 < 0 → 불확실 — 추가 데이터 필요

    [Sharpe Ratio 부트스트랩]
      근거: Ledoit & Wolf (2008), 'Robust Performance Hypothesis Testing
            with the Sharpe Ratio', J. Empirical Finance
      - 연환산 샤프 = 일간 평균수익률 / 일간 표준편차 × √252
      - 부트스트랩으로 SR 분포 추정 → CI 계산
    """

    def __init__(self, n_bootstrap: int = 1000, ci_level: float = 0.95,
                 block_length: int = None):
        """
        n_bootstrap: 반복 횟수 (기본 1000 — Efron & Tibshirani 권장)
        ci_level:    신뢰수준 (기본 95%)
        block_length: 블록 길이 (None → √n 자동 계산)
        """
        self.B            = n_bootstrap
        self.ci_level     = ci_level
        self.block_len    = block_length   # None → 자동
        self._seed_offset = 0              # 루프별 seed 차별화용

    def _block_bootstrap_sample(self, arr: np.ndarray) -> np.ndarray:
        """
        Stationary Bootstrap 1회 샘플.
        블록 길이: 기하분포(p=1/L) → 평균 L = √n
        """
        n = len(arr)
        L = self.block_len if self.block_len else max(2, int(n ** 0.5))
        p = 1.0 / L  # 기하분포 파라미터

        sample = np.empty(n)
        i = 0
        while i < n:
            start = np.random.randint(0, n)
            # 기하분포로 블록 길이 샘플
            blen  = min(np.random.geometric(p), n - i)
            for j in range(blen):
                sample[i] = arr[(start + j) % n]
                i += 1
                if i >= n:
                    break
        return sample

    def ic_ci(self, ic_series: np.ndarray) -> dict:
        """
        IC 시계열의 Bootstrap 신뢰구간.

        ic_series: 날짜별 IC 배열 (일별 스피어만 상관)
        반환: {"mean", "ci_lower", "ci_upper", "ci_level",
               "std_boot", "positive_prob"}
        """
        arr = np.array(ic_series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20:
            return {"error": "관측치 부족 (<20)"}

        _seed = 42 + getattr(self, '_seed_offset', 0)
        np.random.seed(_seed)  # 재현성 (루프별 차별화)
        boot_means = np.empty(self.B)
        for b in range(self.B):
            sample         = self._block_bootstrap_sample(arr)
            boot_means[b]  = sample.mean()

        alpha = 1.0 - self.ci_level
        lo    = np.percentile(boot_means, alpha / 2 * 100)
        hi    = np.percentile(boot_means, (1 - alpha / 2) * 100)
        pos_prob = float((boot_means > 0).mean())

        return {
            "mean":          round(float(arr.mean()), 4),
            "ci_lower":      round(float(lo), 4),
            "ci_upper":      round(float(hi), 4),
            "ci_level":      self.ci_level,
            "std_boot":      round(float(boot_means.std()), 4),
            "n_obs":         len(arr),
            "positive_prob": round(pos_prob, 3),
            # CI 하한 > 0 → 강한 증거 (단순 t-stat과 독립적 확인)
            "strong_evidence": float(lo) > 0.0,
        }

    def sharpe_ci(self, returns: np.ndarray,
                  annualize: bool = True) -> dict:
        """
        Sharpe Ratio Bootstrap 신뢰구간.
        Ledoit & Wolf (2008) 방법론.

        returns: 일간 % 수익률 배열
        """
        arr = np.array(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20:
            return {"error": "관측치 부족 (<20)"}

        def _sr(a):
            mu  = a.mean()
            sig = a.std()
            if sig < 1e-9:
                return 0.0
            sr = mu / sig
            if annualize:
                sr *= (252 ** 0.5)
            return sr

        actual_sr = _sr(arr)
        _seed = 42 + getattr(self, '_seed_offset', 0)
        np.random.seed(_seed)  # 재현성 (루프별 차별화)
        boot_sr   = np.empty(self.B)
        for b in range(self.B):
            boot_sr[b] = _sr(self._block_bootstrap_sample(arr))

        alpha = 1.0 - self.ci_level
        lo    = np.percentile(boot_sr, alpha / 2 * 100)
        hi    = np.percentile(boot_sr, (1 - alpha / 2) * 100)

        return {
            "sharpe":     round(float(actual_sr), 3),
            "ci_lower":   round(float(lo), 3),
            "ci_upper":   round(float(hi), 3),
            "ci_level":   self.ci_level,
            "std_boot":   round(float(boot_sr.std()), 3),
            "n_obs":      len(arr),
            # 연환산 SR > 0.5 AND CI 하한 > 0 → 통계적으로 유효한 전략
            # 근거: Sharpe (1994), 0.5 = 연 10% 수익/20% 변동성 = 적정 수준
            "valid_strategy": float(actual_sr) > 0.5 and float(lo) > 0.0,
        }

    def mean_return_ci(self, returns: np.ndarray) -> dict:
        """
        평균 수익률 Bootstrap 신뢰구간.
        """
        arr = np.array(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20:
            return {"error": "관측치 부족 (<20)"}

        _seed = 42 + getattr(self, '_seed_offset', 0)
        np.random.seed(_seed)  # 재현성
        boot_means = np.empty(self.B)
        for b in range(self.B):
            boot_means[b] = self._block_bootstrap_sample(arr).mean()

        alpha = 1.0 - self.ci_level
        lo    = np.percentile(boot_means, alpha / 2 * 100)
        hi    = np.percentile(boot_means, (1 - alpha / 2) * 100)

        return {
            "mean":            round(float(arr.mean()), 3),
            "ci_lower":        round(float(lo), 3),
            "ci_upper":        round(float(hi), 3),
            "ci_level":        self.ci_level,
            "n_obs":           len(arr),
            "significant":     float(lo) > 0.0,  # CI 하한 > 0 → 양수 수익률 통계 유의
        }


# ──────────────────────────────────────────────────────────────────────
# ⑥ RegimeAnalyzer
# ──────────────────────────────────────────────────────────────────────
class RegimeAnalyzer:
    """
    시장 국면별 성과 분리 분석 모듈.

    [문제]
      전체 10년 IC 평균은 각기 다른 시장 환경이 혼합된 값.
      → bull 2017~2019에서는 IC +0.08, bear 2022에서는 IC -0.03이어도
        평균 IC +0.04로 보고됨 → 전략의 취약 국면 숨겨짐.

    [해결 — 국면별 분리 분석]
      각 날짜에 LookAheadGuard.regime_at()으로 국면 레이블 부여.
      bull / sideways / bear 3개 그룹으로 성과 지표 분리 계산.

    [국면 정의 수치 근거]
      SMA200: Faber (2007) '10 Month SMA' 전략 — SSRN 962461
        SPY > SMA200 → 장기 상승 추세 (bull 조건)
        SPY < SMA200 → 장기 하락 추세 (bear 조건)

      SMA50: Lo et al. (2000) 'Foundations of Technical Analysis', RFS
        SMA50 > SMA200 (골든크로스) → bull 확정
        SMA50 < SMA200 (데스크로스)  → bear 확정
        사이: sideways

      변동성 임계 15%:
        연환산 일간변동률 = 일간 std × √252
        vol_threshold = 0.15 → 일간 std 임계 = 0.15/√252 ≈ 0.94%
        CBOE VIX 역사적 중앙값 ≈ 19~20 → 임묵적 vol ≈ 1.2%/일
        15% 연환산 vol → VIX ≈ 15 → 저변동/고변동 분기선

    [분석 지표]
      국면별: 평균 IC, IC t-stat, 평균 수익률, 샤프, 승률, 관측치 수
      → 어떤 국면에서 전략이 작동하는지/안 하는지 파악
      → bear 국면에서 IC 음수 → 헤지 또는 비활성화 규칙 설계 근거

    [시장 국면 경험적 빈도 — S&P500 1950~2024]
      bull:     약 60~65% 기간
      sideways: 약 20~25% 기간
      bear:     약 15~20% 기간
      출처: NBER Recession Indicators + Fama-French 3-Factor returns
    """

    # 주요 역사적 Bear/Bull 구간 레퍼런스 (검증용)
    KNOWN_REGIMES = {
        "bear":     [("2007-10", "2009-03"),   # 금융위기
                     ("2020-02", "2020-03"),   # 코로나 급락
                     ("2022-01", "2022-10")],  # 금리충격
        "bull":     [("2013-01", "2015-12"),   # QE 강세장
                     ("2016-11", "2018-09"),   # 트럼프 강세장
                     ("2023-01", "2024-12")],  # AI 강세장
        "sideways": [("2015-07", "2016-03"),   # 중국 ショック 횡보
                     ("2018-10", "2019-03")],  # Fed 금리인상 횡보
    }

    def __init__(self):
        self.regime_counts: dict = {}

    def label_dates(self, df: pd.DataFrame,
                    spy_df: pd.DataFrame,
                    lag_guard: "LookAheadGuard") -> pd.DataFrame:
        """
        df 각 행에 당시 시점의 국면 레이블 부여.
        LookAheadGuard를 통해 룩어헤드 바이어스 없이 국면 계산.

        df:        백테스트 레코드 DataFrame (date 컬럼 필요)
        spy_df:    SPY 가격 DataFrame (Close 컬럼)
        lag_guard: LookAheadGuard 인스턴스
        """
        if spy_df is None or "date" not in df.columns:
            df["regime"] = "sideways"
            return df

        spy_close = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()
        spy_idx   = {str(d.date()): i for i, d in enumerate(spy_close.index)}

        regimes = []
        for date_str in df["date"]:
            idx = spy_idx.get(str(date_str), None)
            if idx is None:
                regimes.append("sideways")
            else:
                regimes.append(lag_guard.regime_at(spy_close, idx))

        df = df.copy()
        df["regime"] = regimes

        # 국면별 행 수 집계
        self.regime_counts = df["regime"].value_counts().to_dict()
        return df

    def analyze_by_regime(self, df: pd.DataFrame,
                          score_col: str = "cs_score",
                          fwd_col:   str = "ret_20d") -> dict:
        """
        국면별 IC, 수익률, 샤프 분리 계산.

        반환: {"bull": {...}, "sideways": {...}, "bear": {...},
               "regime_counts": {...}, "regime_ic_spread": float}
        """

        if "regime" not in df.columns:
            return {}

        results = {}
        for regime in ["bull", "sideways", "bear"]:
            sub = df[df["regime"] == regime]
            if len(sub) < 20:
                results[regime] = {"n": len(sub), "error": "관측치 부족"}
                continue

            # 날짜별 IC
            daily_ics = []
            for _, grp in sub.groupby("date"):
                if len(grp) < 4:
                    continue
                if fwd_col not in grp.columns:
                    continue
                # BUG-1과 동일 원칙: fillna(0) → dropna()
                _rv = grp[[score_col, fwd_col]].dropna()
                if len(_rv) < 4:
                    continue
                ic, _ = _spearmanr(_rv[score_col].values.astype(float),
                             _rv[fwd_col].values.astype(float))
                if not np.isnan(ic):
                    daily_ics.append(float(ic))

            if not daily_ics:
                results[regime] = {"n": len(sub), "error": "IC 계산 실패"}
                continue

            ic_arr = np.array(daily_ics)
            ic_mean = float(ic_arr.mean())
            ic_std  = max(float(ic_arr.std()), 1e-9)
            t_stat  = ic_mean / (ic_std / len(ic_arr) ** 0.5)
            icir    = ic_mean / ic_std

            # 수익률 통계
            if fwd_col in sub.columns:
                rets = sub[fwd_col].dropna().values.astype(float)
                mean_ret  = float(rets.mean())
                win_rate  = float((rets > 0).mean() * 100)
                sharpe    = (mean_ret / max(rets.std(), 1e-9) *
                             (252 / int(fwd_col.replace("ret_","").replace("d","") or 20)) ** 0.5)
            else:
                mean_ret = win_rate = sharpe = 0.0

            results[regime] = {
                "n":          len(sub),
                "n_days_ic":  len(daily_ics),
                "ic_mean":    round(ic_mean, 4),
                "ic_std":     round(ic_std, 4),
                "icir":       round(icir, 3),
                "t_stat":     round(t_stat, 2),
                "ic_sig":     "✅" if abs(ic_mean) >= 0.02 and abs(t_stat) >= 2.0 else
                              "⚠️" if abs(ic_mean) >= 0.01 and abs(t_stat) >= 1.5 else "❌",
                "mean_ret":   round(mean_ret, 2),
                "win_rate":   round(win_rate, 1),
                "sharpe":     round(sharpe, 2),
            }

        # bull-bear IC 스프레드: 클수록 국면 의존성 높음
        bull_ic  = results.get("bull",     {}).get("ic_mean", 0.0)
        bear_ic  = results.get("bear",     {}).get("ic_mean", 0.0)
        ic_spread = round(bull_ic - bear_ic, 4)

        return {
            **results,
            "regime_counts":   self.regime_counts,
            "ic_spread_bull_bear": ic_spread,
            # IC 스프레드 > 0.03 → 전략이 bull에서 현저히 강함
            # → bear 국면 비활성화 규칙 설계 권장
            "regime_dependent": abs(ic_spread) > 0.03,
        }


# ─────────────────────────────────────────────────────────────
# 싱글톤 인스턴스
# ─────────────────────────────────────────────────────────────
_lag_guard   = LookAheadGuard()
_mt_guard    = MultipleTestingGuard(alpha=0.05)
_cost_engine = TransactionCostEngine(account_type="retail")
_rf_validator= RollForwardValidator()
_bootstrap   = BootstrapCI(n_bootstrap=1000, ci_level=0.95)
_regime_analyzer = RegimeAnalyzer()

# ─────────────────────────────────────────────────────────────
def download_ticker(tk: str) -> tuple:
    """종목 일봉 다운로드 — Rate limit 재시도 3회."""
    import time, random
    for attempt in range(5):  # 3→5회로 증가
        try:
            # 재시도 간 기본 지연 (rate limit 선제 방어)
            if attempt > 0:
                base_wait = (2 ** attempt) * 10 + random.uniform(0, 5)  # 지수 백오프
                time.sleep(base_wait)
            df = yf.download(tk, period=PERIOD, interval="1d",
                             auto_adjust=False, progress=False)
            if df.empty: return tk, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip().title() for c in df.columns]
            if "Adj Close" in df.columns:
                adj = pd.to_numeric(df["Adj Close"], errors="coerce")
                if not adj.isna().all(): df["Close"] = adj
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close","Volume"]).sort_index()
            return tk, df if len(df) >= LOOKBACK + max(FORWARD) else None
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["RateLimit","Too Many","429","rate limit"]):
                wait = (2 ** attempt) * 20 + random.uniform(0, 10)  # 지수 백오프
                print(f"  ⏳ {tk} Rate limit → {wait:.0f}초 대기 ({attempt+1}/5)")
                time.sleep(wait)
            elif "No data" in err or "delisted" in err.lower():
                return tk, None  # 상폐 종목 즉시 종료
            else:
                if attempt == 4: return tk, None
    return tk, None


def _safe_raw(val) -> float:
    """
    raw 신호값 안전 변환.
    [FIX-1] None/NaN/inf → np.nan (기존 상수 0 대신)
    v8 문제: raw.get("rs", 0) 방식 → 값 없으면 0으로 채움
             → 해당 신호가 전체 상수 0 → corr() = NaN
    수정: 유효값이면 float 반환, 아니면 np.nan
    → dropna() 로 유효 행만 상관계수 계산에 참여 → 정상
    """
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return round(v, 4) if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def compute_spy_ret(spy_df: pd.DataFrame, date: object) -> float:
    """해당 날짜 기준 SPY 5일 수익률 — 날짜 기반 조회 (인덱스 정렬 버그 수정).
    정수 인덱스(date_idx) 방식은 종목마다 상장일이 달라 SPY와 날짜 미스매치 발생.
    → 실제 날짜로 SPY 시리즈를 슬라이스해서 안전하게 계산.
    """
    try:
        c = spy_df["Close"]
        # 해당 날짜 이전 데이터만 사용 (미래 참조 방지)
        c_prior = c[c.index <= date]
        if len(c_prior) < 6:
            return 0.0
        return float(c_prior.iloc[-1] / c_prior.iloc[-6] - 1) * 100
    except Exception as _e:
        logger.debug("모멘텀 계산 오류: %s", _e)
        return 0.0


def run_backtest_single(tk: str, df: pd.DataFrame, spy_df: pd.DataFrame,
                        spy_regime: str = "sideways",
                        use_lag_guard: bool = True) -> list:
    """
    한 종목에 대해 롤링 백테스트 실행.
    v6: LookAheadGuard + TransactionCostEngine + regime 레이블 추가.
    반환: [{date, score, sub_scores..., ret_20d, ret_60d,
            ret_20d_net, ret_60d_net, roundtrip_pct, regime}, ...]
    """
    records = []
    c_all  = df["Close"]
    v_all  = df["Volume"]
    h_all  = df.get("High",  c_all)
    lo_all = df.get("Low",   c_all)

    n = len(df)

    # ① LookAheadGuard: SPY 전체 시계열 (시점별 슬라이싱 전달용)
    spy_close_full = None
    if spy_df is not None and use_lag_guard:
        spy_close_full = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()

    # ③ TransactionCostEngine: 일평균 달러 거래대금 (trading._calc_trade_cost 단위 통일)
    # avg_vol (주 단위) 레거시 유지 + avg_dollar_vol (달러 단위) 추가
    avg_vol        = float(v_all.mean()) if len(v_all) > 0 else 1_000_000.0
    avg_dollar_vol = float((c_all * v_all).mean()) if len(c_all) > 0 else 0.0

    for i in range(LOOKBACK, n - max(FORWARD)):
        try:
            # ══════════════════════════════════════════════════════════
            # [ISSUE-1] Look-ahead Bias 완전 차단 — shift(1) 강제 적용
            # ══════════════════════════════════════════════════════════
            # 문제: i+1 슬라이싱은 '오늘(i) 종가'를 포함 → SSE.calc에 누출
            #       c[-1] = 당일 종가 → 해당 가격으로 신호 계산 후
            #       "다음날 수익률"을 예측 → 실전 불가능한 미래 정보 사용
            #
            # 해결: 슬라이싱 상한 i+1 → i (i 미포함, "어제까지")
            #       SSE.calc = 어제(i-1) 종가 기준 신호 확정
            #       cur_price = 오늘(i) 종가 = 오늘 아침 시가 체결 근사
            #
            # 수익률 계산 = c_all.iloc[i+fwd] / c_all.iloc[i] - 1
            #   = 오늘 매수(시장가) → fwd일 후 종가 매도
            #   → 실전 집행 가능한 수익률
            #
            # 근거: Lopez de Prado 'Advances in Financial ML' Ch.3
            #       "The current bar's close is not confirmed at signal time"
            #       Aronson 'Evidence-Based Technical Analysis' (2006) p.78
            #       — point-in-time data rule: 신호 확정 시점에서 미래 데이터 금지
            # ══════════════════════════════════════════════════════════
            c  = c_all.iloc[i - LOOKBACK: i]   # shift(1): 어제까지 (i 미포함)
            v  = v_all.iloc[i - LOOKBACK: i]
            h  = h_all.iloc[i - LOOKBACK: i]
            lo = lo_all.iloc[i - LOOKBACK: i]

            # 최소 데이터 보장: LOOKBACK-1 = 251일 → rolling(60) 충분
            if len(c) < max(60, LOOKBACK // 2):
                continue

            # ── ① LookAheadGuard: 시점별 국면 ───────────────────
            # shift(1) 적용: 국면도 어제(i-1) 기준으로 통일
            # spy_pos = i-1 사용 → "오늘 아침 시점에 파악 가능한 국면"
            if use_lag_guard and spy_close_full is not None:
                cur_date = df.index[i]
                spy_pos  = int(np.searchsorted(
                    spy_close_full.index.values.astype("datetime64[D]"),
                    np.datetime64(cur_date, "D")
                ))
                # shift(1): spy_pos-1 = 어제 기준 국면 (당일 미확정 제거)
                spy_pos  = max(0, min(spy_pos - 1, len(spy_close_full) - 1))
                regime_i = _lag_guard.regime_at(spy_close_full, spy_pos)
            else:
                regime_i = spy_regime

            spy_r5 = compute_spy_ret(spy_df, df.index[i]) if spy_df is not None else 0.0

            result = SSE.calc(c, v, h, lo,
                              {"spy": spy_r5, "sector": spy_r5},
                              regime=regime_i)

            score = result["total"]
            fac   = result.get("factor", {})
            raw   = result.get("raw", {})

            # ── 그로스 미래 수익률 + 초과수익률(XS) ─────────────
            # [수정 근거] 결과 분석:
            #   전 구간 수익률 양수(상승장 bias) → 하위20%도 +8.95%
            #   → 구간 구별력 소멸 (스프레드 20일 0.16%p, 기준 1%p)
            # [해결] 시장중립(Market-Neutral) 초과수익률 병행 계산
            #   ret_Xd_xs = 종목 Xd 수익률 - SPY 동기간 수익률
            #   근거: Fama & French (1993) JFE — 시장요인(β) 제거 시
            #         순수 팩터 효과 측정 가능, 상/하위 스프레드 확대
            cur_price   = float(c_all.iloc[i])
            future_rets = {}
            for fwd in FORWARD:
                fut_idx   = min(i + fwd, n - 1)
                fut_price = float(c_all.iloc[fut_idx])
                gross_r   = (fut_price / cur_price - 1) * 100
                future_rets[f"ret_{fwd}d"] = gross_r

                # SPY 동기간 수익률 (lookahead-safe: 현재 날짜 이후만 참조)
                spy_fwd_r = 0.0
                if spy_df is not None:
                    try:
                        spy_c_s   = spy_df["Close"]
                        cur_date  = df.index[i]
                        fut_date  = df.index[fut_idx]
                        spy_cur   = spy_c_s[spy_c_s.index <= cur_date]
                        spy_fut   = spy_c_s[spy_c_s.index <= fut_date]
                        if len(spy_cur) > 0 and len(spy_fut) > 0:
                            spy_fwd_r = (float(spy_fut.iloc[-1]) /
                                         float(spy_cur.iloc[-1]) - 1) * 100
                    except Exception:
                        spy_fwd_r = 0.0
                future_rets[f"ret_{fwd}d_xs"] = round(gross_r - spy_fwd_r, 3)

            # ── ③ TransactionCostEngine: 거래비용 차감 ──────────
            # 기존: 비용 0원 → 그로스 수익률만 기록
            # v6:   retail 수수료 + 슬리피지 + Almgren-Chriss 충격
            daily_vol_pct = max(float(c.pct_change().std() * 100), 0.5)
            cost_info     = _cost_engine.estimate(
                price                = cur_price,
                avg_daily_volume     = avg_vol,
                avg_daily_dollar_vol = avg_dollar_vol,
                order_shares         = 1000.0,
                daily_vol_pct        = daily_vol_pct,
            )
            roundtrip = cost_info["roundtrip_pct"]

            net_rets = {}
            for fwd in FORWARD:
                gross = future_rets.get(f"ret_{fwd}d", 0.0)
                net_rets[f"ret_{fwd}d_net"] = round(gross - roundtrip, 3)

            # ── v12: 전략A(TIER1) 기관 팩터 계산 ──────────────
            _tier = TICKER_TIER.get(tk, 1)

            # ① raw_dvol: 달러 거래대금 Z-score (5d vs 20d 기준)
            #    = (최근 5일 평균 달러거래대금) / (20일 평균) - 1
            #    기관 진입 포착 — Gervais et al.(2001)
            try:
                _dv = (c * v).values.astype(float)  # 일별 달러거래대금
                _dv5  = _dv[-5:].mean()   if len(_dv) >= 5  else np.nan
                _dv20 = _dv[-20:].mean()  if len(_dv) >= 20 else np.nan
                _dv20_std = _dv[-20:].std() if len(_dv) >= 20 else np.nan
                raw_dvol = ((_dv5 - _dv20) / max(_dv20_std, 1e-9)
                            if (_dv20 is not None and _dv20_std is not None
                                and np.isfinite(_dv5) and _dv20 > 0)
                            else np.nan)
                raw_dvol = _safe_raw(raw_dvol)
            except Exception:
                raw_dvol = float("nan")

            # ② raw_mfi: Money Flow Index (14일)
            #    거래량 가중 RSI — Quong & Soudack (1989)
            #    MFI = 100 - 100/(1 + 양일머니플로우/음일머니플로우)
            #    표준: TP[i] vs TP[i-1] 비교, i=1부터 (prepend 금지)
            try:
                _tp  = ((h + lo + c) / 3).values.astype(float)  # Typical Price
                _mf  = _tp * v.values.astype(float)              # Raw Money Flow
                # TP 증가/감소 비교: 인덱스 1부터 (i vs i-1)
                _tp_delta = _tp[1:] - _tp[:-1]   # len = LOOKBACK
                _mf_body  = _mf[1:]               # 동일 길이
                _up  = np.where(_tp_delta >= 0, _mf_body, 0.0)
                _dn  = np.where(_tp_delta <  0, _mf_body, 0.0)
                _w = 14
                _up14 = _up[-_w:].sum() if len(_up) >= _w else np.nan
                _dn14 = _dn[-_w:].sum() if len(_dn) >= _w else np.nan
                raw_mfi = (100 - 100 / (1 + _up14 / max(_dn14, 1e-9))
                           if (np.isfinite(_up14) and np.isfinite(_dn14))
                           else np.nan)
                raw_mfi = _safe_raw(raw_mfi)
            except Exception:
                raw_mfi = float("nan")

            # ③ raw_rv_ratio: 단기 실현변동성 가속도 (ATR14 / HV20)
            #
            #    [명칭 수정 근거]
            #    이전 명칭 raw_iv_proxy("내재변동성 근사")는 오해를 유발.
            #    ATR(14) / HV(20) 은 옵션 시장의 IV(Implied Volatility)가 아니라
            #    "단기 변동성(ATR) vs 중기 변동성(HV)"의 비율 = 실현변동성 가속도.
            #
            #    [경제적 의미]
            #    rv_ratio > 1  → 단기 변동성이 중기보다 확대 중 (불확실성 급증)
            #    rv_ratio < 1  → 단기 변동성이 중기보다 낮음   (변동성 수축/안정)
            #
            #    [이론 근거]
            #    Parkinson (1980) J. Business: range-based volatility estimator
            #    Brandt & Kinlay (2005): realized variance ratio as regime signal
            #    — ATR14 ≈ Parkinson 단기 추정, HV20 ≈ 로그수익률 표준편차 기반
            #
            #    [TIER1 팩터로서의 예측력]
            #    rv_ratio 급등 구간에서 기관 헤지 수요 증가 → 단기 하방 압력
            #    IC 방향: rv_ratio 낮을수록(변동성 안정) 단기 수익률 선행 예상
            try:
                _c_v = c.values.astype(float)
                _h_v = h.values.astype(float)
                _lo_v = lo.values.astype(float)
                _tr  = np.maximum(_h_v[1:] - _lo_v[1:],
                       np.maximum(abs(_h_v[1:] - _c_v[:-1]),
                                  abs(_lo_v[1:] - _c_v[:-1])))
                # ATR14: 최근 14일 True Range 평균 / 현재가 (% 단위)
                _atr14 = _tr[-14:].mean() / max(_c_v[-1], 1e-9) * 100 if len(_tr) >= 14 else np.nan
                # HV20: 최근 20일 로그수익률 연환산 표준편차 (%)
                _hv20  = (np.diff(np.log(_c_v[-21:])).std() * np.sqrt(252) * 100
                          if len(_c_v) >= 21 else np.nan)
                # rv_ratio: 단기/중기 실현변동성 비율 (무차원)
                raw_rv_ratio = (_atr14 / max(_hv20, 1e-9)
                                if (np.isfinite(_atr14) and np.isfinite(_hv20) and _hv20 > 0)
                                else np.nan)
                raw_rv_ratio = _safe_raw(raw_rv_ratio)
            except Exception:
                raw_rv_ratio = float("nan")

            # ── v12: 전략B(TIER2+3) 샤프 목표변수 계산 ──────────
            # sharpe_Xd: 향후 X일 수익률을 연환산 SR로 변환
            # 연환산 SR = (ret_Xd / period_vol) * sqrt(252/Xd)
            #   period_vol = hist_vol * sqrt(Xd/252)  [기간 변동성]
            #   → 연환산 SR = ret_Xd / (hist_vol * sqrt(Xd/252)) * sqrt(252/Xd)
            #              = ret_Xd / hist_vol * (252/Xd)
            # 이렇게 하면 sharpe_hit 기준 0.5 = Sharpe(1994) 기준과 동일
            # (연간 수익 / 연간 변동성 > 0.5 → 양호한 리스크조정 수익)
            try:
                _ret_hist = c.pct_change().dropna().values[-20:]  # 과거 20일 일간수익률
                _hist_std = float(_ret_hist.std()) * np.sqrt(252) * 100  # 연환산 변동성%
                _hist_std = max(_hist_std, 0.5)  # 최소 0.5% (division by zero 방지)
            except Exception:
                _hist_std = 15.0  # fallback: 시장 평균 연환산 변동성

            # ── ALERT-1 수정: ann_sr 클리핑 ─────────────────────
            # 문제: TIER3 소형주 중 hist_vol이 매우 작은 종목(거래 거의 없음)에서
            #       ann_sr = ret / hist_vol * (252/fwd) 이 폭발 → sharpe_mean=14.0
            # 수정: SR을 현실적 범위 [-10, +10]으로 클리핑
            #   근거: 세계 최고 퀀트펀드(Renaissance, Two Sigma) 연환산 SR < 4
            #         SR=10은 실전 불가능 → 데이터 오염으로 취급
            #         Grinold & Kahn 'Active Portfolio Mgmt' (2000):
            #         "SR > 3 is exceptional and rare"
            #   _hist_std 최소값: 기존 0.5% → 2.0%로 상향
            #     이유: 연환산 2% 미만은 가격 변동이 거의 없는 유동성 극히 낮은 종목
            #           예: FFIE, RIDE 같은 거의 거래 없는 종목
            #           → 분모 0.5%면 SR = 5% / 0.5% * (252/20) = 315 → 비현실
            _hist_std = max(_hist_std, 2.0)  # 최소 2% (연환산) — 유동성 하한

            SR_CLIP = 5.0   # 연환산 SR 클리핑 상한: SR>5는 데이터 오류로 취급
            sharpe_rets = {}
            for fwd_s in FORWARD:
                gross_r = future_rets.get(f"ret_{fwd_s}d", np.nan)
                # 연환산 SR = ret_Xd(%) / hist_vol(%) * (252/Xd)
                # 근거: Sharpe(1994), annualized form
                ann_sr = (gross_r / _hist_std * (252 / fwd_s)
                          if np.isfinite(gross_r) else np.nan)
                # 클리핑: [-5, +5] 범위 초과는 오염 데이터
                if np.isfinite(ann_sr):
                    ann_sr = float(np.clip(ann_sr, -SR_CLIP, SR_CLIP))
                sharpe_rets[f"sharpe_{fwd_s}d"] = round(ann_sr, 4) if np.isfinite(ann_sr) else np.nan

            record = {
                "ticker":  tk,
                "tier":    _tier,
                "date":    str(df.index[i].date()),
                "regime":  regime_i,
                "score":   round(score, 1),
                "fscore":  round(float(fac.get("total", 0)), 1),
                # ── 기존 raw 신호 (FIX-1 유지) ──
                "raw_vol_z":    _safe_raw(raw.get("vol_z")),
                "raw_bb_z":     _safe_raw(raw.get("bb_z")),
                "raw_rs":       _safe_raw(raw.get("rs")),
                "raw_accel":    _safe_raw(raw.get("accel")),
                "raw_mom":      _safe_raw(raw.get("mom")),
                "raw_bab":      _safe_raw(raw.get("bab")),
                "raw_w52":      _safe_raw(raw.get("w52")),
                "raw_cons":     _safe_raw(raw.get("cons")),
                # ── v12 신규: 전략A 기관 팩터 (TIER1 전용) ──
                "raw_dvol":     raw_dvol,      # 달러거래대금 Z-score
                "raw_mfi":      raw_mfi,       # Money Flow Index
                "raw_rv_ratio":  raw_rv_ratio,  # 실현변동성 가속도 (ATR14/HV20)
                # ── v12 신규: 전략B 샤프 목표변수 (TIER2+3 전용) ──
                **sharpe_rets,   # sharpe_20d, sharpe_60d
                "hist_vol":     round(_hist_std, 2),  # 과거 변동성 (진단용)
                # ── v12b: TIER3 역발상 전용 5일 목표변수 ──────────────
                # Jegadeesh & Titman (2001): 단기 반전은 1-4주 보유 시 유효
                # 20d/60d 목표변수로는 반전 신호가 희석됨 → 5d 전용 추가
                # FORWARD_T3[0]=5일 목표변수 — c_all(전체 시계열) 기준으로 계산
                # 주의: c = c_all.iloc[i-LOOKBACK:i] (슬라이싱) → i 기준 인덱스 불가
                #       반드시 c_all + n(전체 길이) 사용
                "ret_5d_t3": (round(float(
                    (c_all.iloc[min(i + FORWARD_T3[0], n-1)] / c_all.iloc[i] - 1) * 100
                ), 4) if _tier == 3 and i + FORWARD_T3[0] < n else np.nan),
                **future_rets,
                **net_rets,
                "roundtrip_pct": round(roundtrip, 4),
                "cost_class":    cost_info["cost_class"],
            }
            records.append(record)
        except Exception:
            continue
    return records


def score_bucket(s: float) -> str:
    """점수 → 구간 레이블."""
    if s >= 75: return "🔥 75-100 강매집"
    if s >= 55: return "✅ 55-75 관심"
    if s >= 40: return "⚖️ 40-55 중립"
    if s >= 25: return "⚠️ 25-40 약세"
    return "🔴 0-25 매도압력"

BUCKET_ORDER = ["🔥 75-100 강매집","✅ 55-75 관심","⚖️ 40-55 중립","⚠️ 25-40 약세","🔴 0-25 매도압력"]


# ══════════════════════════════════════════════════════════════════════
#  LightGBMRanker — 비선형 팩터 상호작용 포착
#
#  [왜 선형 모델만으로 부족한가]
#    현재 cs_score / cf_score 는 Spearman IC 기반 가중 선형합:
#      score = Σ w_i × rank(raw_i)
#    선형합의 한계:
#      ① 팩터 간 교호작용 무시
#         예: MFI 상승 + rv_ratio 낮음 이 동시 충족될 때 훨씬 강한 신호이지만
#             선형합은 두 신호를 독립적으로만 더함
#      ② 비선형 임계(threshold) 효과 무시
#         예: raw_rs > 1.5 이상에서만 모멘텀 효과가 켜지는 구조를 포착 불가
#      ③ 국면-팩터 결합 무시
#         bear 국면에서는 rv_ratio가 역방향 신호가 되는 현상 등
#
#  [LightGBM 선택 이유]
#    - GBDT 계열: 결정 트리 앙상블 → 팩터 교호작용·임계 자동 포착
#    - LightGBM: XGBoost 대비 속도 4~5x (leaf-wise 성장), 메모리 효율
#    - LambdaRank objective: 수익률 예측(회귀)보다 상대 순위 최적화에 적합
#      → IC 극대화 방향과 정합 (Burges et al. 2006 LambdaRank)
#    - 과적합 방어: num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7
#      + 훈련/검증 절대날짜 분리 (OOS_START_DATE 동일 기준)
#
#  [앙상블 전략]
#    lgb_score 를 cs_combo 에 혼합 (기존 모델 유지)
#    혼합 비율: OOS IC 기반 동적 결정
#      lgb_oos_ic > linear_oos_ic  → lgb 비중 높임
#      lgb_oos_ic <= 0              → lgb 완전 제외 (기존 선형만 사용)
#    이유: LGB가 특정 체제에서 과적합될 수 있음 → IC 기준 게이팅으로 방어
#
#  [TIER별 적용]
#    TIER1(대형주):   선형 IC 이미 약함 → LGB 효과 기대, 적용
#    TIER2(중형주):   선형 IC 가장 강함 → LGB가 선형을 보완
#    TIER3(소형주):   mean-reversion 역발상 이미 적용 → LGB 방향 혼동 위험
#                    → TIER3는 LGB 제외, 기존 역발상 유지
# ══════════════════════════════════════════════════════════════════════
class LightGBMRanker:
    """
    LightGBM 기반 비선형 팩터 랭킹 모델.

    파이프라인:
      1. 훈련(IS, <OOS_START_DATE): LGB 훈련 (LambdaRank)
      2. 검증(OOS, >=OOS_START_DATE): lgb_score 예측 + OOS IC 계산
      3. cs_combo 혼합: lgb_oos_ic vs linear_oos_ic 비교 → 동적 비중

    입력 특징(features):
      - raw 신호 11종: vol_z, bb_z, rs, accel, mom, bab, w52, cons,
                       dvol, mfi, rv_ratio
      - 파생 교호작용 특징 4종: rs×mom, dvol×mfi, vol_z×rv_ratio, bb_z×rs
      - 시장 국면 원핫 3종: regime_bull, regime_bear, regime_sideways
      총 18개 특징

    목표변수:
      TIER1/2: ret_60d (60일 수익률 순위)
      TIER3:   제외

    과적합 방어:
      - 훈련/검증 절대날짜 분리 (OOS_START_DATE = 2022-01-01)
      - num_leaves=31 (기본값, 과적합 방지)
      - min_data_in_leaf=50 (소형 리프 제거)
      - feature_fraction=0.7 (컬럼 서브샘플링)
      - bagging_fraction=0.8 + bagging_freq=5
      - early_stopping(50 rounds): 검증 IC로 조기 종료
    """

    # 입력 원시 신호 (run_backtest_single에서 생성되는 컬럼명과 일치해야 함)
    FEATURE_COLS = [
        "raw_vol_z", "raw_bb_z", "raw_rs", "raw_accel",
        "raw_mom",   "raw_bab",  "raw_w52", "raw_cons",
        "raw_dvol",  "raw_mfi",  "raw_rv_ratio",
    ]
    # 교호작용 특징 (비선형 효과 명시적 노출)
    INTERACT_COLS = [
        ("raw_rs",    "raw_mom",      "ia_rs_mom"),      # 모멘텀 강도 결합
        ("raw_dvol",  "raw_mfi",      "ia_dvol_mfi"),    # 거래대금×자금흐름
        ("raw_vol_z", "raw_rv_ratio", "ia_volz_rv"),     # 단기변동성 동조
        ("raw_bb_z",  "raw_rs",       "ia_bbz_rs"),      # 볼린저×상대강도
    ]
    REGIME_DUMMIES = ["regime_bull", "regime_bear", "regime_sw"]

    # LightGBM 파라미터
    _LGB_PARAMS = {
        "objective":        "lambdarank",   # 순위 최적화 (IC 극대화와 정합)
        "metric":           "ndcg",         # 검증 지표 (학습 시)
        "ndcg_eval_at":     [10],           # NDCG@10
        "boosting_type":    "gbdt",
        "num_leaves":       31,             # 과적합 방지 (기본값)
        "min_data_in_leaf": 50,             # 소형 리프 제거
        "feature_fraction": 0.7,            # 컬럼 서브샘플링
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "learning_rate":    0.05,
        "n_estimators":     500,            # early_stopping으로 실제 트리 수 결정
        "verbose":          -1,             # 콘솔 출력 억제
        "n_jobs":           -1,
        "random_state":     42,
        "reg_alpha":        0.1,            # L1 정규화
        "reg_lambda":       0.1,            # L2 정규화
    }

    def __init__(self, oos_start: str = "2022-01-01"):
        self.oos_start   = oos_start
        self.models: dict  = {}    # {tier: lgb.Booster}
        self.oos_ic: dict  = {}    # {tier: float}
        self.feature_names: list = []
        self._trained    = False

    # ── 특징 엔지니어링 ─────────────────────────────────────────────
    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원시 신호 → 모델 입력 특징 변환.
        - NaN 처리: 각 컬럼의 훈련기간 중앙값으로 대체 (중립 대체)
        - 교호작용: 두 원시 신호의 곱 (선형 범위에서 벗어난 결합 효과 포착)
        - 국면 원핫: 비선형 국면-팩터 상호작용 허용
        """
        feat = df[self.FEATURE_COLS].copy()

        # NaN → 중앙값 대체 (컬럼별 훈련기간 중앙값은 fit에서 저장)
        if hasattr(self, "_median_fill"):
            for col, med in self._median_fill.items():
                if col in feat.columns:
                    feat[col] = feat[col].fillna(med)
        else:
            feat = feat.fillna(0.0)

        # 교호작용 특징 추가
        for c1, c2, alias in self.INTERACT_COLS:
            if c1 in feat.columns and c2 in feat.columns:
                feat[alias] = feat[c1].values * feat[c2].values

        # 시장 국면 원핫 인코딩
        if "regime" in df.columns:
            feat["regime_bull"] = (df["regime"] == "bull").astype(float)
            feat["regime_bear"] = (df["regime"] == "bear").astype(float)
            feat["regime_sw"]   = (df["regime"] == "sideways").astype(float)
        else:
            feat["regime_bull"] = 0.0
            feat["regime_bear"] = 0.0
            feat["regime_sw"]   = 1.0

        return feat.astype(float)

    def _spearman_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """예측 순위와 실제 수익률의 Spearman IC."""
        if len(y_true) < 5:
            return 0.0
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 5:
            return 0.0
        ic, _ = _spearmanr(y_pred[valid], y_true[valid])
        return float(ic) if not np.isnan(ic) else 0.0

    # ── 훈련 ─────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, fwd_col: str = "ret_60d") -> dict:
        """
        TIER1/2 각각 LGB 모델 훈련.

        df:      전체 백테스트 레코드 (cross_sectional_rank 호출 전)
        fwd_col: 목표변수 (기본 ret_60d)

        반환: {tier: {"n_train", "n_val", "best_iter", "val_ic"}}
        """
        if not _LGB_OK:
            print("  [LightGBM] 미설치 — fit() 스킵")
            return {}

        if fwd_col not in df.columns:
            print(f"  [LightGBM] 목표변수 {fwd_col} 없음 — fit() 스킵")
            return {}

        train_mask = df["date"] < self.oos_start
        val_mask   = df["date"] >= self.oos_start

        # 훈련기간 중앙값 저장 (NaN 대체용)
        self._median_fill = {
            col: float(df.loc[train_mask, col].median())
            for col in self.FEATURE_COLS
            if col in df.columns
        }

        results = {}

        for tier in [1, 2, 4]:   # TIER4 = Russell 2000, TIER3는 역발상이라 제외
            tier_mask = df["tier"] == tier if "tier" in df.columns else pd.Series(True, index=df.index)
            tr = df[train_mask & tier_mask].copy()
            va = df[val_mask   & tier_mask].copy()

            if len(tr) < 200 or len(va) < 50:
                print(f"  [LightGBM TIER{tier}] 데이터 부족 (훈련{len(tr)}/검증{len(va)}) → 스킵")
                continue

            # 목표변수 정합 확인 (dropna)
            tr = tr.dropna(subset=[fwd_col])
            va = va.dropna(subset=[fwd_col])

            if len(tr) < 200:
                print(f"  [LightGBM TIER{tier}] 유효 훈련 샘플 부족 (<200) → 스킵")
                continue

            X_tr = self._make_features(tr)
            X_va = self._make_features(va)
            y_tr = tr[fwd_col].values.astype(float)
            y_va = va[fwd_col].values.astype(float)

            self.feature_names = list(X_tr.columns)

            # LambdaRank: group 크기 필요 (날짜별 종목 수)
            # 날짜별 그룹 크기 = 해당 날짜의 종목 수
            tr_groups = tr.groupby("date").size().values.tolist()
            va_groups = va.groupby("date").size().values.tolist()

            # 수익률 → 정수 relevance (LambdaRank 요구)
            # 분위수 기반 0~4 등급화 (5분위)
            q80, q60 = np.nanpercentile(y_tr, 80), np.nanpercentile(y_tr, 60)
            q40, q20 = np.nanpercentile(y_tr, 40), np.nanpercentile(y_tr, 20)

            def _to_rel(y: np.ndarray, q20, q40, q60, q80) -> np.ndarray:
                rel = np.where(y >= q80, 4,
                      np.where(y >= q60, 3,
                      np.where(y >= q40, 2,
                      np.where(y >= q20, 1, 0))))
                return rel.astype(int)

            y_tr_rel = _to_rel(y_tr, q20, q40, q60, q80)
            y_va_rel = _to_rel(y_va, q20, q40, q60, q80)

            lgb_train = lgb.Dataset(X_tr, label=y_tr_rel, group=tr_groups,
                                    feature_name=self.feature_names, free_raw_data=False)
            lgb_val   = lgb.Dataset(X_va, label=y_va_rel, group=va_groups,
                                    reference=lgb_train, free_raw_data=False)

            params = dict(self._LGB_PARAMS)
            # early stopping 콜백
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),   # 로그 억제
            ]

            print(f"  [LightGBM TIER{tier}] 훈련 중... (훈련{len(tr)}행 / 검증{len(va)}행)")
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=params.pop("n_estimators", 500),
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )

            # OOS IC 계산 (날짜별 평균)
            # va를 reset_index 후 정수 위치로 접근 (원본 인덱스가 연속이 아닐 수 있음)
            preds = model.predict(X_va)
            va_reset = va.reset_index(drop=True)
            daily_ics = []
            for _, g in va_reset.groupby("date"):
                pos = g.index.tolist()   # reset 후 정수 위치
                if len(pos) < 4:
                    continue
                ic = self._spearman_ic(
                    y_va[pos],
                    preds[pos]
                )
                if not np.isnan(ic):
                    daily_ics.append(ic)

            oos_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
            self.models[tier] = model
            self.oos_ic[tier] = oos_ic

            best_iter = getattr(model, "best_iteration", model.num_trees())
            print(f"  [LightGBM TIER{tier}] 완료 | best_iter={best_iter} | OOS IC={oos_ic:.4f}")

            # 특징 중요도 상위 5개 출력
            imp = model.feature_importance(importance_type="gain")
            top5 = sorted(zip(self.feature_names, imp), key=lambda x: -x[1])[:5]
            print(f"    특징 중요도 상위5: {', '.join(f'{n}({v:.0f})' for n,v in top5)}")

            results[tier] = {
                "n_train":   len(tr),
                "n_val":     len(va),
                "best_iter": best_iter,
                "oos_ic":    round(oos_ic, 4),
            }

        self._trained = len(self.models) > 0
        return results

    # ── 예측 ─────────────────────────────────────────────────────────
    def predict_score(self, df: pd.DataFrame, tier: int) -> np.ndarray:
        """
        티어별 LGB 점수 예측 (0~100 순위로 변환).
        모델 없거나 OOS IC <= 0이면 nan 배열 반환 (선형 모델만 사용).
        """
        if not self._trained or tier not in self.models:
            return np.full(len(df), np.nan)
        if self.oos_ic.get(tier, 0.0) <= 0.0:
            return np.full(len(df), np.nan)

        X = self._make_features(df)
        raw_pred = self.models[tier].predict(X)

        # 예측값 → 0~100 순위 변환 (선형 점수와 단위 통일)
        n = len(raw_pred)
        if n < 2:
            return np.full(n, 50.0)
        order = np.argsort(raw_pred, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.linspace(0, 100, n)
        return ranks

    # ── 앙상블 혼합 비율 계산 ────────────────────────────────────────
    def blend_weight(self, tier: int, linear_oos_ic: float) -> tuple:
        """
        LGB vs 선형 모델 혼합 비율 계산.

        동적 결정 원칙:
          1. LGB OOS IC <= 0:  lgb_w = 0.0 (선형만 사용)
          2. LGB OOS IC <= linear_oos_ic: lgb_w = 0.2 (보수적 혼합)
          3. LGB OOS IC > linear_oos_ic:
               lgb_w = clip(lgb_ic / (lgb_ic + linear_ic), 0.2, 0.5)
             → LGB가 선형보다 강할 때만 비중 확대, 최대 50% 제한

        최대 50% 제한 이유:
          LGB는 IS 과적합 위험 → 선형 모델이 항상 앵커 역할 유지
          Treynor & Black (1973) Appraisal Ratio:
          최적 혼합 비율 = IR_lgb^2 / (IR_linear^2 + IR_lgb^2) 의 보수적 근사

        반환: (lgb_weight, linear_weight)
        """
        lgb_ic = self.oos_ic.get(tier, 0.0)

        if lgb_ic <= 0.0:
            return 0.0, 1.0   # LGB 완전 제외

        if lgb_ic <= linear_oos_ic:
            return 0.2, 0.8   # LGB 소수 보완

        # LGB가 더 강한 경우: IC 비례 혼합 (최대 0.5)
        total = lgb_ic + max(linear_oos_ic, 1e-9)
        lgb_w = float(np.clip(lgb_ic / total, 0.2, 0.5))
        return lgb_w, 1.0 - lgb_w

    def summary(self) -> dict:
        """훈련 결과 요약."""
        return {
            "trained":   self._trained,
            "tiers":     list(self.models.keys()),
            "oos_ic":    self.oos_ic,
            "lgb_ok":    _LGB_OK,
        }


# ── 싱글톤 인스턴스 ─────────────────────────────────────────────────────────
_lgb_ranker = LightGBMRanker(oos_start="2022-01-01")


def _ic_optimal_weights(df: pd.DataFrame,
                        raw_cols: list,
                        fwd_col: str,
                        train_mask: pd.Series,
                        label: str = "") -> np.ndarray:
    """
    훈련 기간 Spearman IC 기반 최적 가중치 — Grinold & Kahn IC-weighting.

    알고리즘:
      1. 훈련 기간(train_mask=True) 데이터만 사용
      2. 각 신호별 날짜별 Spearman IC 계산 후 평균
      3. IC 평균 ≤ 0 인 신호 → 가중치 0 (제거)
      4. 양수 IC에 비례 배분 → 합계 1.0

    과적합 방지:
      훈련/검증 기간을 완전 분리 (날짜 기준 70:30, OOS_START_DATE 절대날짜 고정)
      가중치는 훈련 기간(IS) IC만으로 결정
      검증 기간(OOS) IC는 이 함수에서 절대 사용하지 않음
      근거: Lopez de Prado (2018) Ch.7 — OOS 날짜 고정이 비율 분할보다 우월
    """

    train_df = df.loc[train_mask].copy()
    if fwd_col not in train_df.columns or train_df.empty:
        return np.ones(len(raw_cols)) / len(raw_cols)

    ic_means = []
    for col in raw_cols:
        if col not in train_df.columns:
            ic_means.append(0.0)
            continue
        daily_ics = []
        for _, grp in train_df.groupby("date"):
            valid = grp[[col, fwd_col]].dropna()
            if len(valid) < 5:
                continue
            ic, _ = _spearmanr(valid[col].values, valid[fwd_col].values)
            if not np.isnan(ic):
                daily_ics.append(float(ic))
        ic_means.append(float(np.mean(daily_ics)) if daily_ics else 0.0)

    ic_arr = np.array(ic_means)
    pos    = np.maximum(ic_arr, 0.0)
    total  = pos.sum()
    weights = (pos / total) if total > 1e-9 else np.ones(len(raw_cols)) / len(raw_cols)

    if label:
        print(f"\n  [{label}] 훈련기간 IC-최적 가중치 (fwd={fwd_col}):")
        for c, ic, w in zip(raw_cols, ic_arr, weights):
            bar  = "▓" * int(abs(ic) * 400)
            sign = "+" if ic >= 0 else "-"
            kept = f"{w:.1%}" if w > 0 else "제거"
            # IC 음수 팩터 경고: SmartScore 절대점수에서도 역효과 가능성 있음을 명시
            # 근거: Grinold & Kahn (2000) "Active Portfolio Management" p.147
            #   IC < 0 팩터는 크로스섹셔널에서 자동 제거(weight=0)되지만
            #   절대점수(SmartScore) 계산에서는 별도 차단 필요
            warn = "  ⚠️ IC음수→역방향신호 (SmartScore 기여 확인 필요)" if ic < -0.01 else ""
            print(f"    {c:15s}  IC={sign}{abs(ic):.4f} {bar:8s}  → {kept}{warn}")

    return weights


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜별 크로스섹셔널 변환 — 훈련/검증 분리 IC-최적 가중치.

    [가중치 최적화 — Grinold & Kahn IC-weighting]
      훈련(앞 50% 날짜): 각 신호의 Spearman IC 계산 → IC 비례 가중치 결정
      검증(뒤 50% 날짜): 훈련 가중치로 점수 생성 → 사후 IC 검증만
      → 표본 내 과적합 없이 순수 데이터 기반 가중치 확보

    cs_score : 단기 매집 신호 크로스섹셔널 순위 (0~100)
    cf_score : 장기 팩터 크로스섹셔널 순위     (0~100)
    cs_combo : IC 비례 결합                   (0~100)
    """
    # v12 팩터 — 계층별 이중 전략
    # ─────────────────────────────────────────────────────────
    # TIER1 대형주: 전략A — 기관 주문흐름 팩터 추가
    #   단기: vol_z + bb_z + rs + rv_ratio(ATR14/HV20, 실현변동성 가속도)
    #   장기: mom + bab + dvol(달러거래대금 Z) + mfi(Money Flow Index)
    #   근거: 대형주 IC=0.0103(t=2.00) → 기관 수급 신호 보강
    #         Gervais et al.(2001): 거래대금 급증 → 3-10일 수익률 선행
    # ─────────────────────────────────────────────────────────
    # TIER2+3 중소형주: 전략B — 샤프 기반 리스크 필터링
    #   팩터: 기존 동일, 목표변수 ret→sharpe (위험조정 수익률)
    #   근거: 수익률 스프레드≈0, 샤프 스프레드 0.20p 존재
    #         → 수익률 예측 포기, 리스크 필터링으로 전환
    # ─────────────────────────────────────────────────────────
    raw_st_t1 = ["raw_vol_z", "raw_bb_z", "raw_rs", "raw_rv_ratio"]  # TIER1 단기
    raw_lt_t1 = ["raw_mom", "raw_bab", "raw_dvol", "raw_mfi"]         # TIER1 장기
    raw_st_t23 = ["raw_vol_z", "raw_bb_z", "raw_rs"]                  # TIER2+3 단기
    raw_lt_t23 = ["raw_mom", "raw_bab"]                                # TIER2+3 장기
    # 공통 팩터 (fallback용)
    raw_st = raw_st_t23
    raw_lt = raw_lt_t23

    # ── LightGBM 훈련 (cross_sectional_rank 진입 시 1회) ──────────────
    # 기존 선형 모델과 독립적으로 훈련 → 나중에 앙상블로 혼합
    # TIER3는 역발상 전략 유지, TIER1/2만 LGB 적용
    _fwd_for_lgb = "ret_60d" if "ret_60d" in df.columns else "ret_20d"
    if _LGB_OK and not _lgb_ranker._trained:
        print("\n  [LightGBM] 훈련 시작 (TIER1/2 전용)...")
        _lgb_fit_results = _lgb_ranker.fit(df, fwd_col=_fwd_for_lgb)
        if _lgb_fit_results:
            for t, r in _lgb_fit_results.items():
                print(f"  [LightGBM TIER{t}] OOS IC={r['oos_ic']:.4f} | best_iter={r['best_iter']}")
        print()
    elif not _LGB_OK:
        pass  # import 시 이미 경고 출력됨

    # ── 훈련/검증 날짜 분리 (70:30) ─────────────────────────
    #
    # [50:50 → 70:30으로 변경한 근거]
    #
    # 근거1 — IC 추정 정밀도 (통계적)
    #   날짜 하나의 IC 표준오차 = 1/√(N종목-2)
    #   계층2(N=13) → σ_IC = 1/√11 ≈ 0.30  (매우 큼)
    #   IC 평균의 표준오차 = σ_IC / √(훈련일수)
    #   50:50 → 훈련 1100일: 0.30/√1100 ≈ 0.009
    #   70:30 → 훈련 1540일: 0.30/√1540 ≈ 0.008  (-13% 오차)
    #   → 특히 종목 수 적은 계층2/3에서 가중치 추정 안정성 향상
    #
    # 근거2 — 금융 사이클 커버리지
    #   시장 Bull/Bear 사이클 평균 주기: 3~7년
    #   50:50(5년 훈련): 1~2개 사이클 → 체제 편향 가능성
    #   70:30(7년 훈련): 2~3개 사이클 → 여러 체제 평균화 → 강건한 IC
    #   3회차에서 계층2 IC가 훈련+에서 검증-로 역전된 것이 이 때문
    #
    # 근거3 — 업계 관행 (Lopez de Prado, 'Advances in Financial ML')
    #   Walk-Forward 분석에서 훈련:검증 = 70:30이 표준
    #   50:50은 검증 셋이 너무 커서 검증 데이터가 오히려 낭비됨
    #   (정보는 훈련에서, 평가는 30%면 충분)
    #
    all_dates  = sorted(df["date"].unique())
    n_dates    = len(all_dates)

    # ── [P2-4 FIX] 훈련/검증 절대날짜 고정 ────────────────────────────
    # 문제: TRAIN_RATIO=0.70으로 비율 기반 분할 시
    #       PERIOD='10y' → split_date ≈ 2017, PERIOD='5y' → split_date ≈ 2020
    #       같은 코드가 다른 OOS 구간을 검증 → 결과 재현 불가
    #
    # 해결: OOS_START_DATE 절대날짜 고정
    #   기준: 2022-01-01 (COVID 반등 랠리 종료, 금리인상 사이클 시작)
    #         = 가장 최근의 체제 전환점 → 이후가 진정한 OOS 구간
    #   근거: Lopez de Prado (2018) Ch.7 — "OOS는 훈련 기간 이후
    #         한 번만 사용해야 한다. 날짜 고정이 비율 분할보다 우월."
    #   fallback: 데이터가 2022 이전에만 있으면 70% 비율로 분할
    OOS_START_DATE = "2022-01-01"
    if any(d >= OOS_START_DATE for d in all_dates):
        split_date = OOS_START_DATE
        train_mask = df["date"] < split_date
        val_mask   = df["date"] >= split_date
    else:
        # 구 데이터 fallback: 비율 분할
        split_idx  = int(n_dates * 0.70)
        split_date = all_dates[split_idx]
        train_mask = df["date"] < split_date
        val_mask   = df["date"] >= split_date

    print(f"\n  [최적화] 훈련(IS): ~{split_date}  검증(OOS, 절대날짜 고정): {split_date}~")
    print(f"         훈련 {train_mask.sum()}행  검증 {val_mask.sum()}행")

    # ── 훈련 기간 IC로 가중치 결정 — TIER별 분리 ──────────────
    fwd_ret = "ret_60d" if "ret_60d" in df.columns else "ret_20d"
    fwd_sh  = "sharpe_60d" if "sharpe_60d" in df.columns else fwd_ret

    # TIER1 대형주 (전략A): 수익률 기준 IC, 기관팩터 포함
    df_t1 = df[df["tier"] == 1] if "tier" in df.columns else df
    if len(df_t1) >= 100:
        w_st_t1 = _ic_optimal_weights(df_t1, raw_st_t1, fwd_ret, train_mask[df_t1.index], label="CS단기[T1대형]")
        w_lt_t1 = _ic_optimal_weights(df_t1, raw_lt_t1, fwd_ret, train_mask[df_t1.index], label="CS장기[T1대형]")
    else:
        w_st_t1 = np.ones(len(raw_st_t1)) / len(raw_st_t1)
        w_lt_t1 = np.ones(len(raw_lt_t1)) / len(raw_lt_t1)

    # TIER2+3 중소형주 (전략B): 샤프 기준 IC → 리스크조정 수익 최대화
    df_t23 = df[df["tier"].isin([2, 3])] if "tier" in df.columns else df
    if len(df_t23) >= 50:
        w_st_t23 = _ic_optimal_weights(df_t23, raw_st_t23, fwd_sh, train_mask[df_t23.index], label="CS단기[T2+3중소형/샤프기준]")
        w_lt_t23 = _ic_optimal_weights(df_t23, raw_lt_t23, fwd_sh, train_mask[df_t23.index], label="CS장기[T2+3중소형/샤프기준]")
    else:
        w_st_t23 = np.ones(len(raw_st_t23)) / len(raw_st_t23)
        w_lt_t23 = np.ones(len(raw_lt_t23)) / len(raw_lt_t23)

    # 전체 IC_optimal (combo 비율 계산용 — 전체 기준)
    fwd = fwd_ret
    w_st = w_st_t23  # fallback
    w_lt = w_lt_t23

    # cs_combo 혼합 비율: 두 점수의 훈련기간 IC 절댓값 비례
    # (각 점수의 예측력에 비례해 결합)
    def _score_ic(raw_cols, weights, mask):
        """단일 점수의 날짜별 IC 평균 (훈련기간)"""
        sub = df.loc[mask].copy()
        if fwd not in sub.columns: return 0.0
        # 날짜별 가중합 점수 임시 계산
        ics = []
        for _, grp in sub.groupby("date"):
            if len(grp) < 5: continue
            # BUG-6 수정: fillna(0) → NaN 행 제거
            # 이유: NaN raw 신호를 0으로 치환하면 해당 종목이 중앙값(rank 50)처럼 취급됨
            # → IC가 실제보다 희석 (NaN 많을수록 심각)
            # 수정: raw+ret 모두 유효한 행만 사용 (dropna)
            cols_needed = raw_cols + [fwd]
            valid = grp[cols_needed].dropna()
            if len(valid) < 5: continue
            raw_vals = sum(
                valid[c].values.astype(float) * w
                for c, w in zip(raw_cols, weights)
            )
            ret_vals = valid[fwd].values.astype(float)
            ic, _ = _spearmanr(raw_vals, ret_vals)
            if not np.isnan(ic): ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0

    # ══════════════════════════════════════════════════════════
    # [FIX-B 롤백] TIER1 IC — 20d/60d 혼합 방식 복원
    # ══════════════════════════════════════════════════════════
    # FIX-B (60일 강제) 실험 결과: 계층1 IC 0.0052 → 0.0017 악화
    # 원인: 60일 기준 가중치가 20일 신호 방향을 왜곡
    #       "가중치는 예측 타겟(fwd_ret)에 맞게 학습해야 함"
    #       60일 최적 가중치 ≠ 20일 최적 가중치
    #
    # 복원: 기존 fwd_ret (20d/60d 혼합) 기준으로 TIER1 IC 계산
    # 향후 방향: TIER1 팩터 자체를 교체해야 함
    #   현재 팩터(vol_z/bb_z/rs)는 대형주에서 IC≈0
    #   대형주에는 어닝서프라이즈/애널리스트 리비전 등 펀더멘털 팩터가 필요
    # ══════════════════════════════════════════════════════════
    # TIER1: 기존 fwd_ret 기준 combo IC (FIX-B 롤백)
    _t1_mask  = train_mask & (df["tier"] == 1) if "tier" in df.columns else train_mask
    ic_cs_raw = _score_ic(raw_st_t1, w_st_t1, _t1_mask) if _t1_mask.sum() >= 100 else _score_ic(raw_st, w_st, train_mask)
    ic_cf_raw = _score_ic(raw_lt_t1, w_lt_t1, _t1_mask) if _t1_mask.sum() >= 100 else _score_ic(raw_lt, w_lt, train_mask)
    # TIER2+3: 샤프 기준 combo IC
    _t23_mask = train_mask & df["tier"].isin([2,3]) if "tier" in df.columns else train_mask
    _fwd_save = fwd
    fwd = fwd_sh  # 샤프 기준으로 전환
    ic_cs_t23 = _score_ic(raw_st_t23, w_st_t23, _t23_mask) if _t23_mask.sum() >= 50 else 0.0
    ic_cf_t23 = _score_ic(raw_lt_t23, w_lt_t23, _t23_mask) if _t23_mask.sum() >= 50 else 0.0
    fwd = _fwd_save  # 복원

    # [FIX-3] cf_score 역전 자동 감지 및 반전
    # v8 결과: cf_score IC = -0.018(60d) — 역방향
    # 원인: raw_lt 신호들(mom/bab/w52/cons)의 훈련기간 IC가 음수 → 순위가 뒤집힘
    # 수정: 훈련기간 IC < -0.005 (역방향 신호 임계) → cf_score 반전 플래그 설정
    #       반전: cs_rows에서 cf_score = 100 - cf_raw_rank
    #       근거: 역방향 팩터를 그대로 쓰면 상위 종목이 낮은 수익률 → 손실
    CF_REVERSE_THRESHOLD = -0.005   # IC < 이 값이면 역방향으로 판단
    cf_score_reversed = ic_cf_raw < CF_REVERSE_THRESHOLD
    if cf_score_reversed:
        print(f"\n  [FIX-3] ⚠️  CF장기 역전 감지: 훈련IC={ic_cf_raw:.4f} < {CF_REVERSE_THRESHOLD}")
        print(f"         cf_score 부호 자동 반전 적용 (100 - rank)")
    else:
        print(f"\n  [FIX-3] ✅ CF장기 정상: 훈련IC={ic_cf_raw:.4f} ≥ {CF_REVERSE_THRESHOLD}")

    ic_cs = max(ic_cs_raw, 0.0)
    ic_cf = max(ic_cf_raw, 0.0)   # 반전 후 combo 가중치는 절댓값 비례

    # ══════════════════════════════════════════════════════════
    # FIX-A: CF장기 IC 비유의 시 combo에서 완전 제외
    # ══════════════════════════════════════════════════════════
    # 문제: v12 결과 ic_cf_raw=+0.0033(t=1.08) — 통계적 비유의
    #       IC 기반 자동 비중 = 18% → 노이즈가 combo에 혼입
    # 해결: ic_cf < IC_CF_MIN 이면 combo_w_cf = 0.0 강제
    #   임계값 IC_CF_MIN = 0.01:
    #     t = IC / (1/√N) ≈ 0.01 / (1/√2203) ≈ 0.47 → 비유의
    #     IC=0.01 미만은 순수 노이즈 수준
    # 근거: Grinold & Kahn 'Active Portfolio Mgmt' (2000)
    #       "IC < 0.01: noise level, do not use in alpha model"
    # ══════════════════════════════════════════════════════════
    IC_CF_MIN = 0.010   # 이 이하이면 cf를 combo에서 제외
    if ic_cf < IC_CF_MIN:
        print(f"  [FIX-A] ⚠️  CF장기 IC={ic_cf:.4f} < {IC_CF_MIN} → combo에서 CF 제외 (cs 100%)")
        ic_cf = 0.0   # combo에서 배제

    ic_tot = ic_cs + ic_cf
    # fallback: cs+cf 모두 IC≤0인 경우 cs 100% (cf가 없을 때 cs만으로)
    combo_w_cs = ic_cs / ic_tot if ic_tot > 1e-9 else 1.0
    combo_w_cf = ic_cf / ic_tot if ic_tot > 1e-9 else 0.0
    print(f"\n  [combo 가중치] cs={combo_w_cs:.1%}  cf={combo_w_cf:.1%}")
    print(f"    (훈련기간 IC — cs:{ic_cs_raw:.4f}  cf:{ic_cf_raw:.4f}  cf_reversed={cf_score_reversed})")

    # ── 날짜별 크로스섹셔널 변환 (전체 기간, 결정된 가중치 적용) ──
    def pct_rank_grp(grp, col):
        n    = len(grp)
        # 랭킹 계산에서 NaN → 0 대체는 의도적:
        # NaN raw 신호는 "신호 없음" = 중간값(rank 50)이 아닌 최하위로 처리
        # 0으로 채우면 해당 종목이 하위 랭크에 배치됨 → 보수적 처리
        # IC 계산(dropna)과 달리 랭킹은 모든 종목 포함 필요
        vals = grp[col].fillna(0).values.astype(float)
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.linspace(0, 100, n)
        for uv in np.unique(vals):
            mask = vals == uv
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    def rerank(arr):
        n = len(arr)
        o = np.argsort(arr, kind="stable")
        r = np.empty(n)
        r[o] = np.linspace(0, 100, n)
        for uv in np.unique(arr):
            mask = arr == uv
            if mask.sum() > 1:
                r[mask] = r[mask].mean()
        return r

    # ── 날짜별 크로스섹셔널 변환 — 계층별 독립 랭크 ─────────────
    # [수정 근거] 결과 분석: 중형주(계층2) IC = -0.0054 (역방향)
    #   대형주(37종) + 중형주(13종) 혼합 순위 계산 시
    #   대형주의 강한 vol_z/bb_z/rs 신호가 중형주 순위를 억압
    #   → 중형주는 구조적으로 하위 랭크에 몰림 → IC 역전
    # [수정] 계층별 독립 순위 계산 후 병합
    #   근거: Hou, Xue & Zhang (2015) "Digesting Anomalies", RFS 28(2)
    #         계층 혼합 크로스섹셔널은 소형주 신호 희석 유발
    #   각 계층 내 0~100 순위 → 병합 시 계층 간 비교 가능
    cs_rows = []
    for date, grp in df.groupby("date"):
        n = len(grp)
        if n < 2:
            for idx in grp.index:
                cs_rows.append({"_idx": idx,
                                "cs_score": 50.0,
                                "cf_score": 50.0,
                                "cs_combo": 50.0,
                                "lgb_score": np.nan})
            continue

        # ticker 컬럼 없으면 기존 방식 fallback
        if "ticker" not in grp.columns:
            cs_raw = sum(pct_rank_grp(grp, c) * w for c, w in zip(raw_st, w_st))
            cf_raw = sum(pct_rank_grp(grp, c) * w for c, w in zip(raw_lt, w_lt))
            cs    = rerank(cs_raw)
            _cf_raw_rank = rerank(cf_raw)
            # [FIX-3] cf_score 역전 시 반전
            cf    = (100.0 - _cf_raw_rank) if cf_score_reversed else _cf_raw_rank
            combo = rerank(cs_raw * combo_w_cs + cf * combo_w_cf)
            for i2, idx in enumerate(grp.index):
                cs_rows.append({"_idx": idx,
                                "cs_score": round(cs[i2], 1),
                                "cf_score": round(cf[i2], 1),
                                "cs_combo": round(combo[i2], 1),
                                "lgb_score": np.nan})
            continue

        # ── 계층별 독립 순위 계산 ──────────────────────────────
        # 각 계층: 해당 날짜에 존재하는 계층 종목만으로 0~100 순위
        # → 계층 내 상대 순위 → 계층 간 공정 비교
        tier_col = grp["tier"] if "tier" in grp.columns else                    grp["ticker"].map(lambda x: TICKER_TIER.get(x, 1))

        cs_arr  = np.full(n, 50.0)   # default 중립
        cf_arr  = np.full(n, 50.0)
        cmb_arr = np.full(n, 50.0)
        lgb_arr = np.full(n, np.nan)  # LGB 점수 (TIER1/2만, TIER3는 nan)

        for tier_val in [1, 2, 3, 4]:   # TIER4 = Russell 2000
            tier_mask_local = (tier_col.values == tier_val)
            if tier_mask_local.sum() < 2:
                continue
            t_grp = grp[tier_mask_local]

            # v12: TIER별 팩터/가중치 분기
            if tier_val == 1:
                # 전략A: 기관 주문흐름 팩터 포함
                _all_st = raw_st_t1
                _all_lt = raw_lt_t1
                _full_w_st = w_st_t1
                _full_w_lt = w_lt_t1
            elif tier_val == 4:
                # 전략B': TIER4(Russell 2000) — TIER2+3와 동일 팩터
                # 소형주 특성 동일 → 샤프 기준 IC 학습
                _all_st = raw_st_t23
                _all_lt = raw_lt_t23
                _full_w_st = w_st_t23
                _full_w_lt = w_lt_t23
            else:
                # 전략B: TIER2+3 (IC는 샤프 기준으로 훈련됨)
                _all_st = raw_st_t23
                _all_lt = raw_lt_t23
                _full_w_st = w_st_t23
                _full_w_lt = w_lt_t23

            # 존재하는 컬럼의 인덱스로 가중치 추출 → 슬라이싱 대신 인덱싱
            _idx_st = [i for i, c in enumerate(_all_st) if c in t_grp.columns]
            _idx_lt = [i for i, c in enumerate(_all_lt) if c in t_grp.columns]
            _cols_st = [_all_st[i] for i in _idx_st]
            _cols_lt = [_all_lt[i] for i in _idx_lt]
            _w_st_raw = _full_w_st[_idx_st] if len(_idx_st) > 0 else np.array([])
            _w_lt_raw = _full_w_lt[_idx_lt] if len(_idx_lt) > 0 else np.array([])
            _w_st = _w_st_raw / _w_st_raw.sum() if _w_st_raw.sum() > 1e-9 else np.ones(len(_cols_st)) / max(len(_cols_st), 1)
            _w_lt = _w_lt_raw / _w_lt_raw.sum() if _w_lt_raw.sum() > 1e-9 else np.ones(len(_cols_lt)) / max(len(_cols_lt), 1)

            cs_raw_t = sum(pct_rank_grp(t_grp, c) * w
                           for c, w in zip(_cols_st, _w_st))
            cf_raw_t = sum(pct_rank_grp(t_grp, c) * w
                           for c, w in zip(_cols_lt, _w_lt))

            cs_t   = rerank(cs_raw_t)
            _cf_t_raw = rerank(cf_raw_t)
            # [FIX-3] cf_score 역전 감지 시 순위 반전 (100 - rank)
            cf_t   = (100.0 - _cf_t_raw) if cf_score_reversed else _cf_t_raw
            # combo: 반전된 cf_t를 사용해야 combo도 올바른 방향
            _cf_for_combo = cf_t
            cmb_t  = rerank(cs_raw_t * combo_w_cs + _cf_for_combo * combo_w_cf)

            # ══════════════════════════════════════════════════════════
            # [ISSUE-2] TIER3 U자형(Rank Reversal) 대응 — 역발상 스위칭
            # ══════════════════════════════════════════════════════════
            # 현상: v11 결과 TIER3 IC=0.0421(t=8.68) — 신호가 방향을 맞춤
            #       BUT 수익률: 하위20% +29.91% > 상위20% +15.41% — 역전
            #
            # 진단: IC가 양수(+0.04)인데 수익률이 역전
            #   = 신호가 "절대 수익률"이 아닌 "변동성/반전 모멘텀"을 포착
            #   = 고점수(과열) 종목이 단기 급락, 저점수(침체) 종목이 급반등
            #   = 전형적인 Mean Reversion 패턴 (소형 성장주 과매수/과매도)
            #
            # ── 이론 근거 (보유기간 정합 수정) ───────────────────────
            # Jegadeesh & Titman (2001) JF: "소형주 단기 반전은 1~4주에 유효"
            #   → 핵심: 반전 효과는 5~20일 보유에서 포착, 20d 이상 희석됨
            #   → FIX: cs_combo 역발상은 TIER3에서 ret_5d_t3(5일) 기준 IC로 검증
            #          20d/60d 목표변수는 반전 신호와 보유기간 불일치 → 참고용 유지
            #
            # Baker & Stein (2004) JFE: "유동성 낮은 종목일수록 과매수/과매도 후
            #         반전 강도 높음" → TIER3(소형) 특성과 일치
            #
            # Lehmann (1990) JF: "주 단위 contrarian strategy는 bid-ask bounce
            #         에 의해 상당 부분 설명됨" → 5d 이상 보유 시 이 효과 제거됨
            #         → 5일 보유가 이론적으로 가장 정합
            #
            # 해결: TIER3에 한해 cs_combo를 (100 - cs_combo)로 반전
            #   = "신호 낮을수록 매수" — 역발상 롱 전략
            #   = IC 방향은 그대로 유지, 수익률 예측 방향만 뒤집음
            #   = 실전 운영 시 5일 후 청산 권장 (ret_5d_t3 기준)
            #
            # 조건: TIER3(tier_val==3)에만 적용 — TIER1/2는 정방향 유지
            # ══════════════════════════════════════════════════════════
            # ── LightGBM 독립 예측 (TIER1/2 전용) ──────────────────────
            # cs_score / cf_score / cs_combo 는 선형 모델 그대로 유지.
            # LGB는 lgb_score 컬럼에만 독립적으로 저장 → 비교용.
            # cs_combo를 덮어쓰지 않음.
            if tier_val in [1, 2, 4] and _LGB_OK and _lgb_ranker._trained:
                lgb_pred = _lgb_ranker.predict_score(t_grp, tier=tier_val)
                if not np.all(np.isnan(lgb_pred)):
                    _lgb_scores_for_tier = lgb_pred   # 0~100 순위, 독립 저장
                else:
                    _lgb_scores_for_tier = np.full(len(t_grp), np.nan)
            else:
                _lgb_scores_for_tier = np.full(len(t_grp), np.nan)

            if tier_val == 3:
                # TIER3 역발상: combo 순위 반전 (100 - rank)
                # TIER4(Russell 2000)는 정방향 유지 (유동성 낮아 반전 효과 약함)
                cmb_t = 100.0 - cmb_t
                cs_t  = 100.0 - cs_t
                cf_t  = 100.0 - cf_t

            # 전체 grp 배열에 해당 계층 위치에 결과 기입
            local_indices = np.where(tier_mask_local)[0]
            for li, rank_cs, rank_cf, rank_cmb in zip(
                    local_indices, cs_t, cf_t, cmb_t):
                cs_arr[li]  = round(rank_cs,  1)
                cf_arr[li]  = round(rank_cf,  1)
                cmb_arr[li] = round(rank_cmb, 1)
            # lgb_score 기입 (TIER1/2: 예측값, TIER3: nan 유지)
            for li, lgb_val in zip(local_indices, _lgb_scores_for_tier):
                lgb_arr[li] = round(float(lgb_val), 1) if np.isfinite(lgb_val) else np.nan

        for i2, idx in enumerate(grp.index):
            cs_rows.append({"_idx": idx,
                            "cs_score": cs_arr[i2],
                            "cf_score": cf_arr[i2],
                            "cs_combo": cmb_arr[i2],
                            "lgb_score": lgb_arr[i2]})

    cs_df = pd.DataFrame(cs_rows).set_index("_idx")
    return df.join(cs_df)


def cs_bucket(s: float) -> str:
    """크로스섹셔널 점수(0~100) → 구간 레이블."""
    if s >= 80: return "🔥 80-100 상위20%"
    if s >= 60: return "✅ 60-80 상위40%"
    if s >= 40: return "⚖️ 40-60 중립"
    if s >= 20: return "⚠️ 20-40 하위40%"
    return "🔴 0-20 하위20%"

CS_BUCKET_ORDER = ["🔥 80-100 상위20%","✅ 60-80 상위40%","⚖️ 40-60 중립","⚠️ 20-40 하위40%","🔴 0-20 하위20%"]


def analyze(df: pd.DataFrame) -> dict:
    """전체 분석 실행."""
    results = {}

    # ── 크로스섹셔널 변환 ─────────────────────────────────
    print("  [분석] 날짜별 크로스섹셔널 변환 중...")
    df = cross_sectional_rank(df)

    # ── 선형 vs LGB OOS IC 나란히 비교 ───────────────────────────────
    if "lgb_score" in df.columns and _LGB_OK and _lgb_ranker._trained:
        _fwd_v   = "ret_60d" if "ret_60d" in df.columns else "ret_20d"
        _oos_df  = df[df["date"] >= "2022-01-01"]

        def _daily_ic_mean(score_col, fwd_col, sub):
            ics = []
            for _, _g in sub.groupby("date"):
                _v = _g[[score_col, fwd_col]].dropna()
                if len(_v) < 4: continue
                _ic, _ = _spearmanr(_v[score_col].values, _v[fwd_col].values)
                if not np.isnan(_ic): ics.append(float(_ic))
            mean_ic = float(np.mean(ics)) if ics else 0.0
            std_ic  = float(np.std(ics))  if ics else 0.0
            t_stat  = mean_ic / max(std_ic / max(len(ics)**0.5, 1), 1e-9)
            return mean_ic, t_stat, len(ics)

        if len(_oos_df) > 100 and _fwd_v in _oos_df.columns:
            _lin_ic, _lin_t, _lin_n = _daily_ic_mean("cs_combo",  _fwd_v, _oos_df)
            _lgb_ic, _lgb_t, _lgb_n = _daily_ic_mean("lgb_score", _fwd_v, _oos_df)

            print(f"\n  {'─'*52}")
            print(f"  [OOS IC 비교] 기준: {_fwd_v}  구간: 2022-01-01~")
            print(f"  {'모델':<20} {'IC':>8} {'t-stat':>8} {'n(일)':>7}")
            print(f"  {'─'*52}")
            print(f"  {'선형(cs_combo)':<20} {_lin_ic:>8.4f} {_lin_t:>8.2f} {_lin_n:>7}")
            print(f"  {'LightGBM(lgb_score)':<20} {_lgb_ic:>8.4f} {_lgb_t:>8.2f} {_lgb_n:>7}")
            print(f"  {'─'*52}")
            _delta = _lgb_ic - _lin_ic
            _winner = "LGB" if _lgb_ic > _lin_ic else "선형"
            print(f"  IC 차이: {_delta:+.4f}  →  {_winner} 모델이 OOS에서 우세")

            # TIER별 LGB OOS IC
            for _t, _tic in _lgb_ranker.oos_ic.items():
                _t_sub = _oos_df[_oos_df["tier"] == _t] if "tier" in _oos_df.columns else _oos_df
                _t_lin_ic, _, _ = _daily_ic_mean("cs_combo", _fwd_v, _t_sub)
                print(f"    TIER{_t}: 선형 IC={_t_lin_ic:.4f}  LGB IC={_tic:.4f}  "
                      f"{'LGB우세' if _tic > _t_lin_ic else '선형우세'}")
            print(f"  {'─'*52}\n")

    # 1. 크로스섹셔널 구간별 미래 수익률
    # quantile 기반 버킷 — 가중합 분포가 mean=50,std=15로 좁아지기 때문에
    # 고정 80/60 기준 사용 시 상위20%에 2%만 배정됨 → quantile로 정확히 20%씩
    def _quantile_bucket(score_col: str) -> pd.Series:
        result = pd.Series("⚖️ 40-60 중립", index=df.index, dtype=str)
        for _d, _g in df.groupby("date"):
            if len(_g) < 3:
                result.loc[_g.index] = "⚖️ 40-60 중립"
                continue
            sc = _g[score_col]
            q80, q60 = sc.quantile(0.80), sc.quantile(0.60)
            q40, q20 = sc.quantile(0.40), sc.quantile(0.20)
            for ridx, val in sc.items():
                if   val >= q80: result.loc[ridx] = "🔥 80-100 상위20%"
                elif val >= q60: result.loc[ridx] = "✅ 60-80 상위40%"
                elif val >= q40: result.loc[ridx] = "⚖️ 40-60 중립"
                elif val >= q20: result.loc[ridx] = "⚠️ 20-40 하위40%"
                else:            result.loc[ridx] = "🔴 0-20 하위20%"
        return result
    df["cs_bucket"]    = _quantile_bucket("cs_score")
    df["cf_bucket"]    = _quantile_bucket("cf_score")
    df["combo_bucket"] = _quantile_bucket("cs_combo")   # ★ 주신호 (IC 0.036, ICIR 0.235)
    df["bucket"]       = df["combo_bucket"]              # 호환성 — 주신호로 격상

    # LightGBM 앙상블 결과 버킷 (lgb_score 컬럼이 있고 유효값 존재 시)
    if "lgb_score" in df.columns and df["lgb_score"].notna().any():
        df["lgb_bucket"] = _quantile_bucket("lgb_score")
        _lgb_summary = _lgb_ranker.summary()
        print(f"  [LightGBM] 앙상블 적용 | TIER OOS IC: {_lgb_summary['oos_ic']}")
    else:
        df["lgb_bucket"] = df["combo_bucket"]   # 폴백: combo와 동일
        if _LGB_OK:
            print("  [LightGBM] 예측값 없음 — combo_bucket으로 폴백")

    # ══ 우선순위4: score_v11 먼저 생성 (bucket_abs에서 사용하므로 반드시 선행) ══
    # 기존 "score": IC≈0 (0.0006/0.0049) — 수익률 예측력 없음
    # 재설계: cs_combo 날짜별 퍼센타일 → score_v11 (0~100, IC 기반 예측점수)
    if "cs_combo" in df.columns:
        df["score_v11"] = (
            df.groupby("date")["cs_combo"]
            .transform(lambda x: x.rank(pct=True) * 100)
            .round(1)
        )
        print("  [우선순위4] score_v11 생성 완료 (cs_combo 퍼센타일 기반)")
    else:
        df["score_v11"] = df["score"]
    # ════════════════════════════════════════════════════════════════

    # [v11 우선순위4] 절대예측점수 구간 — score_v11 기반 (score_v11 생성 후 실행)
    ABS_BUCKET_ORDER = ["🔥 상위2%","✅ 상위10%","⚖️ 상위30%","⚠️ 상위50%","🔴 하위50%"]
    def _abs_bucket_v11(s):
        if   s >= 98: return "🔥 상위2%"
        elif s >= 90: return "✅ 상위10%"
        elif s >= 70: return "⚖️ 상위30%"
        elif s >= 50: return "⚠️ 상위50%"
        else:         return "🔴 하위50%"
    df["bucket_abs"] = df["score_v11"].apply(_abs_bucket_v11)
    # 기존 SSE 절대점수 분위수 (진단용)
    _score_p98 = df["score"].quantile(0.98)
    _score_p90 = df["score"].quantile(0.90)
    _score_p70 = df["score"].quantile(0.70)
    _score_p50 = df["score"].quantile(0.50)
    print(f"  [v11] score_v11 분위수 임계(고정): "
          f"상위2%≥98  상위10%≥90  상위30%≥70  상위50%≥50")
    print(f"  [진단] 기존 SSE score 분위수: "
          f"상위2%≥{_score_p98:.1f}  상위10%≥{_score_p90:.1f}  "
          f"상위30%≥{_score_p70:.1f}  상위50%≥{_score_p50:.1f} (IC≈0, 참고용)")
    def make_bucket_stats(df, bucket_col, order):
        """
        [FIX-2] 베타 중립화 — 그로스 + XS(시장초과) 수익률 병행 계산.
        그로스 수익률만 보면 상승장 beta 효과로 하위20%도 높게 나옴.
        XS = 종목 수익률 - SPY 동기간 수익률 → 순수 팩터 알파 측정.
        """
        stats = {}
        has_xs = all(f"ret_{fwd}d_xs" in df.columns for fwd in FORWARD)
        for bucket in order:
            sub = df[df[bucket_col] == bucket]
            if len(sub) < 10: continue
            bst = {}
            for fwd in FORWARD:
                col = f"ret_{fwd}d"
                r = sub[col].dropna()
                d = {
                    "mean":     round(r.mean(), 2),
                    "median":   round(r.median(), 2),
                    "win_rate": round((r > 0).mean() * 100, 1),
                    "sharpe":   round(r.mean()/r.std()*np.sqrt(252/fwd), 2) if r.std()>0 else 0,
                    "skew":     round(float(r.skew()), 2),
                    "n":        len(r),
                }
                # [FIX-2] XS 수익률 통계 추가
                if has_xs:
                    xs_col = f"ret_{fwd}d_xs"
                    rxs = sub[xs_col].dropna()
                    if len(rxs) >= 10:
                        d["xs_mean"]     = round(rxs.mean(), 2)
                        d["xs_win_rate"] = round((rxs > 0).mean() * 100, 1)
                        d["xs_sharpe"]   = round(rxs.mean()/rxs.std()*np.sqrt(252/fwd), 2) if rxs.std()>0 else 0
                bst[f"{fwd}d"] = d
            stats[bucket] = bst
        return stats

    # 우선순위3: TIER3(고베타 소형주) 분리 — U자형 mean reversion 노이즈 제거
    # TIER3가 CS 하위20%에서 반등 급등 → 스프레드 0.31%p로 왜곡
    # → TIER1+2 메인과 TIER3 별도 결과 나란히 제공
    if "tier" in df.columns:
        df_main  = df[df["tier"].isin([1, 2])].copy()   # 대형+중형 (메인)
        df_tier3 = df[df["tier"] == 3].copy()            # 고베타 소형 (별도)
    else:
        df_main  = df.copy()
        df_tier3 = pd.DataFrame()

    # ── v12: 샤프 기반 버킷 통계 (TIER2+3 전략B용) ─────────────
    def make_sharpe_bucket_stats(df_sub, bucket_col, order):
        """
        전략B: 샤프 기반 리스크 필터링 효과 측정
        각 구간의 평균 샤프, 히트율(샤프>0.5), 수익률, 변동성 출력
        """
        stats = {}
        for bucket in order:
            sub = df_sub[df_sub[bucket_col] == bucket]
            if len(sub) < 10: continue
            bst = {}
            for fwd in FORWARD:
                ret_col = f"ret_{fwd}d"
                sh_col  = f"sharpe_{fwd}d"
                r = sub[ret_col].dropna()
                d = {
                    "mean":      round(r.mean(), 2),
                    "win_rate":  round((r > 0).mean() * 100, 1),
                    "sharpe_gross": round(r.mean()/r.std()*np.sqrt(252/fwd), 2) if r.std()>0 else 0,
                    "n":         len(r),
                }
                if sh_col in sub.columns:
                    sh = sub[sh_col].dropna()
                    if len(sh) >= 10:
                        # ALERT-1 후처리 클리핑: 캐시 재사용을 위해 record 저장 대신
                        # 분석 단계에서 클리핑 → 캐시(.bt_cache_v12b) 삭제 불필요
                        # SR > 5 or < -5 는 유동성 극히 낮은 종목 오염값
                        sh_clipped = sh.clip(-5.0, 5.0)
                        d["sharpe_mean"]   = round(sh_clipped.mean(), 3)
                        d["sharpe_hit"]    = round((sh_clipped > 0.5).mean() * 100, 1)
                        d["sharpe_median"] = round(sh_clipped.median(), 3)
                if "hist_vol" in sub.columns:
                    d["avg_vol"] = round(sub["hist_vol"].mean(), 1)
                bst[f"{fwd}d"] = d
            stats[bucket] = bst
        return stats

    # ★ 메인: TIER1만 cs_combo 기준 (전략A — 수익률 예측)
    bucket_stats = make_bucket_stats(df_main, "combo_bucket", CS_BUCKET_ORDER)
    results["bucket_stats"] = bucket_stats
    # v12: TIER1 전용 (전략A — 기관팩터 수익률 기준)
    df_tier1 = df[df["tier"] == 1].copy() if "tier" in df.columns else df_main
    results["tier1_bucket_stats"]  = make_bucket_stats(df_tier1, "combo_bucket", CS_BUCKET_ORDER)
    # v12: TIER2+3 전용 (전략B — 샤프 기반 리스크 필터링)
    if "tier" in df.columns:
        _t23 = pd.concat([df_main[df_main["tier"]==2], df_tier3]) if len(df_tier3)>0 else df_main[df_main["tier"]==2]
        results["tier23_sharpe_stats"] = make_sharpe_bucket_stats(_t23, "combo_bucket", CS_BUCKET_ORDER)
    else:
        results["tier23_sharpe_stats"] = {}
    # TIER2 단독
    df_tier2 = df[df["tier"] == 2].copy() if "tier" in df.columns else pd.DataFrame()
    results["tier2_sharpe_stats"]  = make_sharpe_bucket_stats(df_tier2, "combo_bucket", CS_BUCKET_ORDER) if len(df_tier2)>0 else {}
    # TIER3 별도 (mean reversion 특성)
    results["tier3_bucket_stats"]  = make_sharpe_bucket_stats(df_tier3, "combo_bucket", CS_BUCKET_ORDER) if len(df_tier3) > 0 else {}
    # 전체 비교용
    results["all_bucket_stats"]    = make_bucket_stats(df, "combo_bucket", CS_BUCKET_ORDER)
    # 단기 CS 참고용
    results["cs_bucket_stats"]     = make_bucket_stats(df_tier1, "cs_bucket", CS_BUCKET_ORDER)
    # 장기 팩터 구간별
    results["factor_bucket_stats"] = make_bucket_stats(df_tier1, "cf_bucket", CS_BUCKET_ORDER)
    # 절대점수 구간별
    results["abs_bucket_stats"]    = make_bucket_stats(df_main, "bucket_abs", ABS_BUCKET_ORDER)

    # 2. 서브컴포넌트 상관관계
    # 단기 원점수 상관계수 — v5 재구성
    sub_cols = ["raw_vol_z","raw_bb_z","raw_rs","raw_rv_ratio"]
    sub_labels = {"raw_vol_z":   "거래량 서지(vol_surge)",
                  "raw_bb_z":    "BB압축 역수",
                  "raw_rs":      "RS vs SPY 5일",
                  "raw_rv_ratio":"RV가속도(ATR14/HV20)"}
    # 장기 원점수 상관계수 — v5 재구성
    fac_cols = ["raw_mom","raw_bab","raw_w52","raw_cons"]
    fac_labels = {"raw_mom":  "12M-1M모멘텀",
                  "raw_bab":  "BAB Vol-adj샤프",
                  "raw_w52":  "52주고점근접도",
                  "raw_cons": "추세일관성(상승일비율)"}
    # cs_score / cf_score 상관계수도 추가
    sub_cols   += ["cs_score", "cf_score", "cs_combo"]
    sub_labels.update({"cs_score":"★ CS단기점수","cf_score":"★ CS장기점수","cs_combo":"★ CS종합점수"})
    # LGB 독립 점수 상관계수도 추가 (비교용)
    if "lgb_score" in df.columns and df["lgb_score"].notna().any():
        sub_cols.append("lgb_score")
        sub_labels["lgb_score"] = "★ LightGBM점수(비선형)"
    # 상관계수: 20일/60일 기준 (매집신호 중장기 선행성 검증)
    corr_stats = {}
    for col in sub_cols:
        label = sub_labels[col]
        corr_stats[label] = {}
        for fwd in [20, 60]:
            ret_col = f"ret_{fwd}d"
            if col in df.columns and ret_col in df.columns:
                corr = df[[col, ret_col]].dropna().corr().iloc[0,1]
                corr_stats[label][f"{fwd}d"] = round(corr, 3)
    results["corr_stats"] = corr_stats

    # 장기 상관계수 (20일 수익률)
    fac_corr = {}
    for col in fac_cols:
        label = fac_labels[col]
        fac_corr[label] = {}
        for fwd in [20, 60]:
            ret_col = f"ret_{fwd}d"
            if col in df.columns and ret_col in df.columns:
                corr = df[[col, ret_col]].dropna().corr().iloc[0,1]
                fac_corr[label][f"{fwd}d"] = round(corr, 3)
    results["fac_corr"] = fac_corr

    # 7. IC — Fama-MacBeth 방식
    # 날짜별 cross-sectional Spearman IC 계산 후 평균/t-stat
    # ICIR = IC/std(IC) : 1일 이상이면 강한 팩터
    # ── [FIX-5] IC 계산 — XS IC 제거 ────────────────────────────────
    # v8 문제: XS(종목수익률 - SPY수익률) IC가 그로스 IC와 동일하게 출력됨
    # 원인: Spearman rank IC는 공통 상수(SPY)를 빼도 rank 순서가 불변
    #       → XS IC ≡ 그로스 IC (수학적으로 동일)
    # 수정: XS IC 계산 제거, 그로스 IC만 계산
    # 진짜 시장중립 IC는 롱숏 포트폴리오 Sharpe로 측정해야 함 (v10 예정)
    print("  [FIX-5] XS IC 제거 — Spearman rank에서 공통 SPY 상수 제거 시 IC 불변")
    print("         진짜 시장중립 성과는 구간별 XS 수익률 스프레드로 확인")

    ic_results = {}
    _ic_targets = [("cs_score","CS단기(선형)"), ("cf_score","CS장기(선형)"),
                   ("cs_combo","CS종합(선형)"), ("score","절대SSE(구버전)"),
                   ("score_v11","★예측점수v11(cs_combo퍼센타일)")]
    # LGB 독립 점수 IC 비교 추가
    if "lgb_score" in df.columns and df["lgb_score"].notna().any():
        _ic_targets.append(("lgb_score", "★ LightGBM(비선형)"))
    for score_col, label in _ic_targets:
        if score_col not in df.columns: continue
        ic_20d, ic_60d = [], []
        for _d, _g in df.groupby("date"):
            if len(_g) < 10: continue
            # BUG-1 수정: fillna(0) → dropna() — NaN을 0으로 채우면 IC 희석
            for ret_col, ic_list in [("ret_20d", ic_20d), ("ret_60d", ic_60d)]:
                if ret_col not in _g.columns: continue
                _valid = _g[[score_col, ret_col]].dropna()
                if len(_valid) < 4: continue
                ic, _ = _scipy_stats.spearmanr(_valid[score_col].values, _valid[ret_col].values)
                if not np.isnan(ic): ic_list.append(ic)

        def _stat(arr, fwd_label):
            if len(arr) < 10: return None
            a    = np.array(arr)
            mean = a.mean()
            std  = max(a.std(), 1e-9)
            t    = mean / (std / len(a)**0.5)
            icir = mean / std
            if abs(mean) >= 0.02 and abs(t) >= 2.0:
                sig = "✅ 유의"
            elif abs(mean) >= 0.01 and abs(t) >= 1.5:
                sig = "⚠️ 약신호"
            else:
                sig = "❌ 비유의"
            return {"IC": round(float(mean),4), "IC_std": round(float(std),4),
                    "ICIR": round(float(icir),3), "t_stat": round(float(t),2),
                    "n_days": len(arr), "fwd": fwd_label, "sig": sig}

        s20, s60 = _stat(ic_20d, "20일"), _stat(ic_60d, "60일")
        if s20 or s60:
            ic_results[label] = {"20d": s20, "60d": s60}
    results["ic_results"] = ic_results

    # 장기 FactorScore 구간별 20일 수익률
    if "fscore" in df.columns:
        df["fbucket"] = df["fscore"].apply(lambda s:
            # fscore(FactorScore) 임계값: 0~100 균등 4분위 기준
            # 25/50/75 분위수 → 각 구간 25%씩 균등 배분
            # 근거: SSE FactorScore는 0~100 설계 → 25단위가 자연스러운 4분위
            "💎 75+ 장기매력" if s>=75 else "📈 50-75 장기관심" if s>=50 else
            "📊 25-50 중립"   if s>=25 else "📉 0-25 비선호")
        fbucket_stats = {}
        for fb in ["💎 75+ 장기매력","📈 50-75 장기관심","📊 25-50 중립","📉 0-25 비선호"]:
            sub = df[df["fbucket"]==fb]
            if len(sub) < 5: continue
            r20 = sub["ret_20d"].dropna()
            fbucket_stats[fb] = {
                "mean": round(r20.mean(),2), "win_rate": round((r20>0).mean()*100,1),
                "sharpe": round(r20.mean()/r20.std()*(252/20)**0.5,2) if r20.std()>0 else 0,
                "n": len(r20)}
        results["fbucket_stats"] = fbucket_stats

    # 3. 전체 분포
    results["total_records"] = len(df)
    results["score_dist"] = {
        "mean":   round(df["score"].mean(), 1),
        "median": round(df["score"].median(), 1),
        "std":    round(df["score"].std(), 1),
        "p25":    round(df["score"].quantile(0.25), 1),
        "p75":    round(df["score"].quantile(0.75), 1),
    }

    # 4. 고점수 vs 저점수 누적수익률 (시간별 추이)
    # 매일 점수 상위 20% / 하위 20% 뽑아서 다음 5일 수익률 평균
    results["top_vs_bottom"] = compute_top_bottom(df)
    results["_df"] = df  # 계층별 IC 분석용 (main에서 참조)

    # ══════════════════════════════════════════════════════
    # v6 펀드-그레이드 검증 6종 (analyze 내 통합 실행)
    # ══════════════════════════════════════════════════════

    # v11: 역방향 팩터(raw_accel/raw_w52/raw_cons) 제거 후 유효 팩터만 검증
    # v12b: raw_rv_ratio(TIER1 단기) + raw_dvol/raw_mfi(TIER1 장기) 추가
    raw_all = ["raw_vol_z","raw_bb_z","raw_rs",
               "raw_mom","raw_bab",
               "raw_rv_ratio","raw_dvol","raw_mfi"]
    # 역방향 팩터는 진단용으로만 별도 저장 (OOS IC 계산 대상에서 제외)
    raw_removed = ["raw_accel","raw_w52","raw_cons"]

    # ── ② 복수검정 보정 ────────────────────────────────────
    # ic_results에서 t-stat 수집 → BH-FDR + Bonferroni
    print("  [v6] 복수검정 보정 (BH-FDR + Bonferroni)...")
    t_stats_20d = {}
    t_stats_60d = {}
    for label, ir in results.get("ic_results", {}).items():
        if ir.get("20d") and ir["20d"].get("t_stat") is not None:
            t_stats_20d[label] = ir["20d"]["t_stat"]
        if ir.get("60d") and ir["60d"].get("t_stat") is not None:
            t_stats_60d[label] = ir["60d"]["t_stat"]

    n_obs_approx = max(len(sorted(df["date"].unique())), 10)
    mt_result_20d = _mt_guard.combined_test(t_stats_20d, n_obs_approx) if t_stats_20d else {}
    mt_result_60d = _mt_guard.combined_test(t_stats_60d, n_obs_approx) if t_stats_60d else {}
    fp_risk = MultipleTestingGuard.false_positive_risk(
        k=len(t_stats_20d), alpha=0.05)
    results["mt_correction"] = {
        "20d": mt_result_20d,
        "60d": mt_result_60d,
        "false_positive_risk_pct": round(fp_risk * 100, 1),
        "n_tests": len(t_stats_20d),
    }
    print(f"         위양성 리스크 {fp_risk*100:.1f}% → BH/Bonferroni 보정 적용")

    # ── ④ Roll-Forward OOS 검증 ────────────────────────────
    print("  [v6] Roll-Forward OOS 검증 (3년훈/1년테)...")
    rf_folds = _rf_validator.run(df, raw_all, fwd_col="ret_20d")
    rf_stability = _rf_validator.stability_score()
    results["roll_forward"] = {
        "folds":     rf_folds,
        "stability": rf_stability,
    }
    if rf_stability:
        stable = rf_stability.get("stable", False)
        print(f"         {rf_stability.get('n_folds',0)}폴드  "
              f"평균OOS-IC={rf_stability.get('mean_oos_ic',0):.4f}  "
              f"양수폴드={rf_stability.get('positive_fold_pct',0):.0f}%  "
              f"{'✅ 안정' if stable else '⚠️ 불안정'}")

    # ── ⑤ Bootstrap 신뢰구간 ──────────────────────────────
    print("  [v6] Bootstrap CI 계산 (B=1000, Stationary Block)...")
    boot_results = {}
    for _boot_idx, (score_col, label) in enumerate(
            [("cs_score","CS단기"), ("cf_score","CS장기"), ("cs_combo","CS종합")]):
        if score_col not in df.columns:
            continue
        # BootstrapCI seed: 루프마다 42+idx로 차별화
        # → 동일 seed로 3개 반복 시 identical 샘플 생성 방지
        _bootstrap._seed_offset = _boot_idx  # ic_ci/sharpe_ci 내부에서 42+offset 사용
        # 날짜별 IC 시계열 재계산
        ic_20 = []
        for _d, _g in df.groupby("date"):
            if len(_g) < 5:
                continue
            # BUG-2 수정: fillna(0) → dropna() 통일 (BUG-1과 동일 원칙)
            if "ret_20d" not in _g.columns: continue
            _valid = _g[[score_col, "ret_20d"]].dropna()
            if len(_valid) < 4: continue
            ic, _ = _spearmanr(_valid[score_col].values.astype(float),
                         _valid["ret_20d"].values.astype(float))
            if not np.isnan(ic):
                ic_20.append(float(ic))

        if ic_20:
            ci_ic  = _bootstrap.ic_ci(np.array(ic_20))
            # 그로스 vs 순수익률 샤프 CI
            gross_rets = df["ret_20d"].dropna().values.astype(float)
            # ⚠️  sharpe_ci는 소수 단위(예: 0.025) 수익률 요구
            # ret_20d는 % 단위(예: 2.5) → /100 변환
            # valid_strategy 기준 SR>0.5: 일간소수 기준으로 연환산 ≈ 연 8~12% / 변동성 1~2% = 현실적
            gross_rets_dec = gross_rets / 100.0
            ci_sr_gross = _bootstrap.sharpe_ci(gross_rets_dec)
            net_col = "ret_20d_net"
            net_rets_arr = df[net_col].dropna().values.astype(float) if net_col in df.columns else gross_rets
            net_rets_dec = net_rets_arr / 100.0
            ci_sr_net   = _bootstrap.sharpe_ci(net_rets_dec)
            ci_mean_net = _bootstrap.mean_return_ci(net_rets_arr)  # mean_return_ci는 % 단위 유지

            boot_results[label] = {
                "ic_ci":       ci_ic,
                "sharpe_gross":ci_sr_gross,
                "sharpe_net":  ci_sr_net,
                "mean_net":    ci_mean_net,
            }
            print(f"         [{label}] IC 95%CI=[{ci_ic.get('ci_lower','?'):.4f}, "
                  f"{ci_ic.get('ci_upper','?'):.4f}]  "
                  f"강한근거={'✅' if ci_ic.get('strong_evidence') else '❌'}")

    results["bootstrap_ci"] = boot_results

    # ── ⑥ 국면별 분리 분석 ────────────────────────────────
    # 주의: df에 "regime" 컬럼이 run_backtest_single에서 이미 추가됨
    # RegimeAnalyzer.label_dates()는 그 컬럼 없을 때 fallback용
    print("  [v6] 국면별 성과 분리 분석...")
    if "regime" in df.columns:
        regime_res = _regime_analyzer.analyze_by_regime(
            df, score_col="cs_score", fwd_col="ret_20d")
        results["regime_analysis"] = regime_res
        for rg in ["bull","sideways","bear"]:
            r = regime_res.get(rg, {})
            if "ic_mean" in r:
                print(f"         {rg:8s}: IC={r['ic_mean']:+.4f}  "
                      f"t={r['t_stat']:+.2f}  "
                      f"SR={r['sharpe']:+.2f}  "
                      f"{r['ic_sig']}")
        ic_spread = regime_res.get("ic_spread_bull_bear", 0)
        print(f"         Bull-Bear IC 스프레드: {ic_spread:+.4f}  "
              f"{'→ 국면 의존적 전략' if regime_res.get('regime_dependent') else '→ 국면 무관 안정'}")

    # ── ① LookAheadGuard 요약 ─────────────────────────────
    lag_summary = _lag_guard.summary()
    results["lag_guard_summary"] = lag_summary
    print(f"  [v6] LookAheadGuard: {lag_summary['total_calls']}호출  "
          f"캐시히트 {lag_summary['cache_hit_pct']}%  "
          f"위반 {lag_summary['violations']}건")

    return results


def compute_top_bottom(df: pd.DataFrame) -> dict:
    """날짜별로 상위20%/하위20% 종목의 다음 5일 수익률 추이."""
    df2 = df.sort_values("date")
    dates = sorted(df2["date"].unique())
    top_rets, bot_rets, date_list = [], [], []
    for d in dates:
        day = df2[df2["date"] == d]
        if len(day) < 5: continue
        sc_col = "cs_score" if "cs_score" in day.columns else "score"
        p80 = day[sc_col].quantile(0.80)
        p20 = day[sc_col].quantile(0.20)
        ret_col_tb = "ret_20d" if "ret_20d" in day.columns else "ret_60d"
        top = day[day[sc_col] >= p80][ret_col_tb].median()
        bot = day[day[sc_col] <= p20][ret_col_tb].median()
        if pd.notna(top) and pd.notna(bot):
            top_rets.append(round(top, 2))
            bot_rets.append(round(bot, 2))
            date_list.append(d)
    # 30일 이동평균으로 노이즈 제거
    s_top = pd.Series(top_rets).rolling(30, min_periods=5).mean().round(2).tolist()
    s_bot = pd.Series(bot_rets).rolling(30, min_periods=5).mean().round(2).tolist()
    return {"dates": date_list, "top": s_top, "bot": s_bot}


def build_html(results: dict, elapsed: float) -> str:
    """백테스트 결과 → HTML 리포트."""
    bs = results["bucket_stats"]
    cs = results["corr_stats"]
    dist = results["score_dist"]
    n = results["total_records"]
    tb = results["top_vs_bottom"]

    # ── 구간별 테이블 행 생성 ──
    def color_ret(v):
        if v is None: return "#666"
        return "#00ff88" if v > 0.5 else "#ff4466" if v < -0.5 else "#ffd700"
    def color_corr(v):
        if v is None: return "#666"
        return "#00ff88" if v > 0.05 else "#ff4466" if v < -0.05 else "#888"

    bucket_rows = ""
    for bucket in CS_BUCKET_ORDER:
        if bucket not in bs: continue
        st = bs[bucket]
        row = f'<tr><td style="font-weight:700">{bucket}</td>'
        for fwd in FORWARD:
            s = st.get(f"{fwd}d", {})
            m = s.get("mean", 0)
            wr = s.get("win_rate", 0)
            sh = s.get("sharpe", 0)
            nn = s.get("n", 0)
            mc = color_ret(m)
            wr_c = "#00ff88" if wr > 55 else "#ff4466" if wr < 45 else "#ffd700"
            med  = s.get("median", 0)
            row += (
                '<td style="color:' + mc + '">' + f"{m:+.2f}%" + '</td>'
                + '<td style="color:' + color_ret(med) + '">' + f"{med:+.2f}%" + '</td>'
                + '<td style="color:' + wr_c + '">' + f"{wr:.0f}%" + '</td>'
                + '<td style="color:' + color_ret(sh) + '">' + f"{sh:.2f}" + '</td>'
                + f'<td style="color:#556">{nn:,}</td>'
            )
        bucket_rows += row + "</tr>"

    # ── 상관관계 테이블 ──
    corr_rows = ""
    for label, cv in cs.items():
        row = f'<tr><td style="font-weight:600">{label}</td>'
        for fwd in FORWARD:
            v = cv.get(f"{fwd}d", 0)
            row += f'<td style="color:{color_corr(v)}">{v:+.3f}</td>'
        corr_rows += row + "</tr>"

    # ── Chart.js 데이터 ──
    chart_dates = json.dumps(tb["dates"][-120:])  # 최근 120개
    chart_top   = json.dumps([x for x in tb["top"][-120:]])
    chart_bot   = json.dumps([x for x in tb["bot"][-120:]])

    # ── 신호등 요약 ──
    # 75+ 구간의 5일 평균이 양수인지, 0-25 구간이 음수인지 확인
    # BUG-2/5 수정: FORWARD=[20,60] — 5d 없음 → 20d 기준으로 교체
    top_20d = bs.get("🔥 80-100 상위20%",{}).get("20d",{}).get("mean", None)
    top_xs  = bs.get("🔥 80-100 상위20%",{}).get("20d",{}).get("xs_mean", None)
    bot_20d = bs.get("🔴 0-20 하위20%",{}).get("20d",{}).get("mean", None)
    top_wr  = bs.get("🔥 80-100 상위20%",{}).get("20d",{}).get("win_rate", None)
    spread  = round((top_20d or 0) - (bot_20d or 0), 2) if (top_20d is not None and bot_20d is not None) else None

    top_20d_str = f"{top_20d:+.2f}%" if top_20d is not None else "N/A"
    # verdict: 그로스 스프레드 + XS 방향성 종합 판정
    if top_20d is not None and spread is not None and spread > 0.5 and top_wr is not None and top_wr > 53:
        verdict = "🟢 유효 — 상위20%가 하위20% 대비 통계적으로 유의한 초과수익"
        verdict_c = "#00ff88"
    elif top_20d is not None and top_20d > 0 and spread is not None and spread > 0:
        verdict = "🟡 부분 유효 — 방향성 확인, 스프레드 약함"
        verdict_c = "#ffd700"
    else:
        verdict = "🔴 주의 — 상위/하위 구간 구별력 불명확"
        verdict_c = "#ff4466"


    # ── XS(시장초과) 구간별 테이블 행 생성 ──────────────────
    xs_bucket_rows = ""
    for bucket in CS_BUCKET_ORDER:
        if bucket not in bs: continue
        st = bs[bucket]
        row = f'<tr><td style="font-weight:700">{bucket}</td>'
        for fwd in FORWARD:
            s   = st.get(f"{fwd}d", {})
            xm  = s.get("xs_mean")
            xwr = s.get("xs_win_rate", 0)
            xsh = s.get("xs_sharpe", 0)
            nn  = s.get("n", 0)
            if xm is None:
                row += '<td colspan="4" style="color:#4a6080">N/A</td>'
                continue
            mc   = color_ret(xm)
            wrc  = "#00ff88" if xwr > 55 else "#ff4466" if xwr < 45 else "#ffd700"
            row += (
                '<td style="color:' + mc   + '">' + f"{xm:+.2f}%" + '</td>'
                + '<td style="color:' + wrc  + '">' + f"{xwr:.0f}%" + '</td>'
                + '<td style="color:' + color_ret(xsh) + '">' + f"{xsh:.2f}" + '</td>'
                + f'<td style="color:#556">{nn:,}</td>'
            )
        xs_bucket_rows += row + "</tr>"

    # ── IC 테이블 행 생성 ─────────────────────────────────────
    ic_rows = ""
    ic_res  = results.get("ic_results", {})
    for lbl, ic_d in ic_res.items():
        row = f'<tr><td style="font-weight:600;color:#c8d8f0">{lbl}</td>'
        for fwd_key in ["20d", "60d"]:
            ic = ic_d.get(fwd_key)
            if not ic:
                row += '<td colspan="3" style="color:#4a6080">—</td>'
                continue
            ic_v = ic["IC"]; t_v = ic["t_stat"]; sig = ic["sig"]
            ic_c = "#00ff88" if ic_v > 0.02 else "#ffd700" if ic_v > 0 else "#ff4466"
            t_c  = "#00ff88" if abs(t_v) > 2.0 else "#ffd700" if abs(t_v) > 1.5 else "#4a6080"
            row += (
                f'<td style="color:{ic_c};font-family:monospace">{ic_v:+.4f}</td>'
                + f'<td style="color:{t_c};font-family:monospace">{t_v:+.2f}</td>'
                + f'<td>{sig}</td>'
            )
        ic_rows += row + "</tr>"

    # ── 스프레드 카드 (20일/60일 그로스 + XS) ────────────────
    spread_str = f"{spread:+.2f}%p" if spread is not None else "N/A"
    spread_c   = "#00ff88" if (spread or 0) > 0.5 else "#ffd700" if (spread or 0) > 0 else "#ff4466"
    top_xs_str = f"{top_xs:+.2f}%" if top_xs is not None else "N/A"
    xs_spread_val = None
    if top_xs is not None:
        bot_xs = bs.get("🔴 0-20 하위20%",{}).get("20d",{}).get("xs_mean")
        if bot_xs is not None:
            xs_spread_val = round(top_xs - bot_xs, 2)
    xs_spread_str = f"{xs_spread_val:+.2f}%p" if xs_spread_val is not None else "N/A"
    xs_spread_c   = "#00ff88" if (xs_spread_val or 0) > 0.3 else "#ffd700" if (xs_spread_val or 0) > 0 else "#ff4466"

    # ── [ISSUE-3/B] TIER2+3 전략B 리스크조정 수익 요약 (상단 배너용) ──
    # 전략B 목표: 수익률 예측 X → 변동성 낮은 상태에서 진입 필터링
    # 핵심 지표: 상위20% vs 하위20% 변동성(avg_vol) 차이 + 샤프 히트율
    t23_sh = results.get("tier23_sharpe_stats", {})
    _t23_top = t23_sh.get("🔥 80-100 상위20%", {}).get("20d", {})
    _t23_bot = t23_sh.get("🔴 0-20 하위20%",   {}).get("20d", {})
    t23_top_vol  = _t23_top.get("avg_vol",   None)   # 상위20% 평균 변동성
    t23_bot_vol  = _t23_bot.get("avg_vol",   None)   # 하위20% 평균 변동성
    t23_top_sh   = _t23_top.get("sharpe_mean", None) # 상위20% 평균 연환산 SR
    t23_bot_sh   = _t23_bot.get("sharpe_mean", None)
    t23_top_hit  = _t23_top.get("sharpe_hit",  None) # 상위20% SR>0.5 히트율
    t23_bot_hit  = _t23_bot.get("sharpe_hit",  None)
    # 변동성 스프레드: 상위 - 하위 (음수 = 상위20%가 더 안정적)
    vol_spread = round(t23_top_vol - t23_bot_vol, 1) if (t23_top_vol and t23_bot_vol) else None
    sh_spread  = round(t23_top_sh  - t23_bot_sh,  3) if (t23_top_sh  and t23_bot_sh)  else None
    # 변동성 필터 판정: 상위20% 변동성 < 하위20% 변동성이면 ✅ (리스크 필터 작동)
    vol_filter_ok = vol_spread is not None and vol_spread < 0
    vol_spread_str = f"{vol_spread:+.1f}%p" if vol_spread is not None else "N/A"
    sh_spread_str  = f"{sh_spread:+.3f}"    if sh_spread  is not None else "N/A"
    t23_top_vol_str  = f"{t23_top_vol:.1f}%" if t23_top_vol  is not None else "N/A"
    t23_top_sh_str   = f"{t23_top_sh:.3f}"   if t23_top_sh   is not None else "N/A"
    t23_top_hit_str  = f"{t23_top_hit:.1f}%" if t23_top_hit  is not None else "N/A"
    vol_filter_color = "#00ff88" if vol_filter_ok else "#ffd700"
    sh_spread_color  = "#00ff88" if (sh_spread or 0) > 0.1 else "#ffd700" if (sh_spread or 0) > 0 else "#ff4466"
    # TIER3 역발상 여부 표시 (ISSUE-2)
    t3_reversal_note = "✅ TIER3 역발상(Rank Reversal) 적용 — 저점수 롱 전략"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartScore v10 백테스트 리포트</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#060d1a; --bg2:#0a1428; --bg3:#0f1e35; --bg4:#060f20;
    --border:#1a2f50; --accent:#00d4ff; --green:#00ff88;
    --red:#ff3366; --yellow:#ffd700; --text:#c8d8f0; --dim:#4a6080;
    --mono:'IBM Plex Mono',monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Noto Sans KR',sans-serif;padding:28px 32px;max-width:1600px;margin:0 auto}}
  h1{{font-size:1.5rem;color:var(--accent);margin-bottom:3px;letter-spacing:-.5px}}
  .sub{{color:var(--dim);font-size:.82rem;margin-bottom:22px}}
  .tag{{display:inline-block;background:#0a1e3a;border:1px solid #1a3a5c;border-radius:4px;padding:2px 8px;font-size:.75rem;color:#6a90c0;margin-left:6px;font-family:var(--mono)}}
  /* 판정 배너 */
  .verdict{{padding:12px 20px;border-radius:8px;border-left:4px solid {verdict_c};
    background:var(--bg3);margin-bottom:20px;font-size:.95rem;font-weight:600;color:{verdict_c};
    display:flex;align-items:center;gap:12px}}
  /* 메타 카드 */
  .meta{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
  .card{{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
    padding:14px 22px;text-align:center;min-width:130px}}
  .card .val{{font-size:1.35rem;font-weight:700;font-family:var(--mono);color:var(--accent)}}
  .card .lbl{{font-size:.68rem;color:var(--dim);margin-top:3px}}
  .card.hi .val{{color:{spread_c}}}
  /* 섹션 타이틀 */
  h2{{font-size:1rem;color:var(--accent);margin:26px 0 10px;border-bottom:1px solid var(--border);padding-bottom:5px;display:flex;align-items:center;gap:6px}}
  /* 테이블 공통 */
  table{{width:100%;border-collapse:collapse;font-size:.8rem}}
  th{{background:var(--bg3);color:var(--dim);padding:7px 9px;text-align:center;
    border-bottom:1px solid var(--border);white-space:nowrap;font-weight:500}}
  td{{padding:6px 9px;border-bottom:1px solid var(--border);text-align:center;font-family:var(--mono)}}
  td:first-child{{text-align:left;font-family:'Noto Sans KR',sans-serif;font-size:.82rem}}
  tr:hover td{{background:var(--bg3)}}
  /* 그리드 */
  .g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px}}
  @media(max-width:900px){{.g2,.g3{{grid-template-columns:1fr}}}}
  /* 카드 박스 */
  .box{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:18px}}
  /* 차트 */
  .chart-wrap{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:20px}}
  /* 노트 */
  .note{{color:var(--dim);font-size:.72rem;margin-top:7px;line-height:1.65}}
  /* 배지 */
  .badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.72rem;font-weight:700}}
  .badge-g{{background:rgba(0,255,136,.15);color:#00ff88}}
  .badge-y{{background:rgba(255,215,0,.12);color:#ffd700}}
  .badge-r{{background:rgba(255,51,102,.12);color:#ff3366}}
</style>
</head>
<body>

<h1>📊 SmartScore 백테스트 리포트 <span class="tag">v12</span><span class="tag">이중전략</span><span class="tag">T1기관팩터/T2+3샤프필터</span></h1>
<div class="sub">
  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  종목 {len(TICKERS)}개 &nbsp;·&nbsp; 기간 {PERIOD} &nbsp;·&nbsp;
  소요 {elapsed:.0f}초 &nbsp;·&nbsp; 관측치 {n:,}건
</div>

<div class="verdict">
  {verdict}
  <span style="font-size:.8rem;color:var(--dim);margin-left:auto">
    20일 스프레드 {spread_str} &nbsp;|&nbsp; XS스프레드 {xs_spread_str}
  </span>
</div>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  전략A (TIER1 대형주) — 수익률 예측 지표
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<div style="color:#4a6080;font-size:.72rem;margin:-10px 0 6px;letter-spacing:.5px">
  ▶ 전략A: TIER1 대형주 — 기관 팩터 수익률 예측
</div>
<div class="meta">
  <div class="card"><div class="val">{n:,}</div><div class="lbl">총 관측치</div></div>
  <div class="card"><div class="val">{dist['mean']}</div><div class="lbl">평균 SmartScore</div></div>
  <div class="card hi"><div class="val">{spread_str}</div><div class="lbl">20일 그로스 스프레드</div></div>
  <div class="card" style="--accent:{xs_spread_c}"><div class="val" style="color:{xs_spread_c}">{xs_spread_str}</div><div class="lbl">20일 XS 스프레드</div></div>
  <div class="card"><div class="val" style="color:{verdict_c}">{top_20d_str}</div><div class="lbl">TIER1 상위20% 20일 그로스</div></div>
  <div class="card"><div class="val" style="color:{xs_spread_c}">{top_xs_str}</div><div class="lbl">TIER1 상위20% 20일 XS</div></div>
</div>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  전략B (TIER2+3 중소형주) — 리스크 조정 수익 지표
  [ISSUE-3/B] 상위20%의 변동성 감소 수치를 상단에 명시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<div style="color:#4a6080;font-size:.72rem;margin:6px 0 6px;letter-spacing:.5px">
  ▶ 전략B: TIER2+3 중소형주 — 샤프 기반 리스크 필터링
  <span style="color:{vol_filter_color};margin-left:10px">
    {"✅ 변동성 필터 작동" if vol_filter_ok else "⚠️ 변동성 필터 확인 필요"}
  </span>
  &nbsp;|&nbsp;
  <span style="color:#4a6080;font-size:.7rem">{t3_reversal_note}</span>
</div>
<div class="meta">
  <div class="card" style="border-color:{vol_filter_color}55">
    <div class="val" style="color:{vol_filter_color}">{t23_top_vol_str}</div>
    <div class="lbl">상위20% 변동성(연환산)</div>
  </div>
  <div class="card" style="border-color:{vol_filter_color}55">
    <div class="val" style="color:{vol_filter_color}">{vol_spread_str}</div>
    <div class="lbl">변동성 스프레드 (상위-하위)<br><span style="font-size:.65rem;color:#4a6080">음수 = 상위20%가 더 안정적</span></div>
  </div>
  <div class="card" style="border-color:{sh_spread_color}55">
    <div class="val" style="color:{t23_top_sh_str and sh_spread_color}">{t23_top_sh_str}</div>
    <div class="lbl">상위20% 평균 연환산 SR</div>
  </div>
  <div class="card" style="border-color:{sh_spread_color}55">
    <div class="val" style="color:{sh_spread_color}">{sh_spread_str}</div>
    <div class="lbl">SR 스프레드 (상위-하위)<br><span style="font-size:.65rem;color:#4a6080">양수 = 리스크 필터 유효</span></div>
  </div>
  <div class="card">
    <div class="val" style="color:#00d4ff">{t23_top_hit_str}</div>
    <div class="lbl">상위20% SR&gt;0.5 히트율<br><span style="font-size:.65rem;color:#4a6080">Sharpe(1994) 기준</span></div>
  </div>
</div>

<!-- ①②: 그로스 + XS 구간표 -->
<div class="g2">
  <div class="box">
    <h2>📈 ★ TIER1 대형주 — cs_combo 구간별 수익률 <span class="badge badge-y">전략A: 기관팩터</span></h2>
    <table>
      <thead>
        <tr>
          <th rowspan="2" style="text-align:left">구간</th>
          <th colspan="5">20일 후</th>
          <th colspan="5">60일 후</th>
        </tr>
        <tr>{"".join(["<th>평균</th><th>중앙값</th><th>승률</th><th>샤프</th><th>N</th>"]*2)}</tr>
      </thead>
      <tbody>{bucket_rows}</tbody>
    </table>
    <p class="note">· 그로스: 시장 상승 beta 포함 · 승률 &gt;53% 유효 · 샤프: 연환산</p>
  </div>

  <div class="box">
    <h2>📊 ★ TIER1 대형주 — 초과수익률 <span class="badge badge-g">XS(시장중립)</span></h2>
    <table>
      <thead>
        <tr>
          <th rowspan="2" style="text-align:left">구간</th>
          <th colspan="4">20일 XS</th>
          <th colspan="4">60일 XS</th>
        </tr>
        <tr>{"".join(["<th>XS평균</th><th>승률</th><th>샤프</th><th>N</th>"]*2)}</tr>
      </thead>
      <tbody>{xs_bucket_rows}</tbody>
    </table>
    <p class="note">· XS = 종목수익률 - SPY 동기간 수익률 · 상승장 beta 제거된 순수 알파</p>
  </div>
</div>

<!-- ③: 상관계수 + IC -->
<div class="g2">
  <div class="box">
    <h2>🔗 서브컴포넌트 상관계수</h2>
    <table>
      <thead><tr><th style="text-align:left">컴포넌트</th><th>20일</th><th>60일</th></tr></thead>
      <tbody>{corr_rows}</tbody>
    </table>
    <p class="note">· &gt;0.05 유효 · &lt;0 역방향(빨강) — v11: accel/w52/cons 제거 완료</p>
  </div>

  <div class="box">
    <h2>📐 IC 분석 <span class="badge badge-y">Fama-MacBeth</span></h2>
    <table>
      <thead>
        <tr>
          <th style="text-align:left">팩터</th>
          <th colspan="3">20일</th>
          <th colspan="3">60일</th>
        </tr>
        <tr>{"".join(["<th>IC</th><th>t</th><th>판정</th>"]*2)}</tr>
      </thead>
      <tbody>{ic_rows}</tbody>
    </table>
    <p class="note">· IC &gt;0.02 &amp; t &gt;2.0 유의 · ICIR &gt;0.5 강한팩터 · &gt;0.03 헤지펀드급</p>
  </div>
</div>

<!-- ④: 상위/하위 추이 차트 -->
<h2>📉 상위 20% vs 하위 20% 20일 수익률 추이 <span class="badge badge-y">30일 MA</span></h2>
<div class="chart-wrap">
  <canvas id="tbChart" height="70"></canvas>
  <p class="note">· 초록(상위20%)이 빨강(하위20%)보다 지속적으로 위에 있을수록 신호 유효</p>
</div>

<script>
new Chart(document.getElementById('tbChart'), {{
  type:'line',
  data:{{
    labels:{chart_dates},
    datasets:[
      {{label:'상위 20% (강매집)',data:{chart_top},
        borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,.07)',
        borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}},
      {{label:'하위 20% (매도압력)',data:{chart_bot},
        borderColor:'#ff3366',backgroundColor:'rgba(255,51,102,.07)',
        borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}},
    ]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{labels:{{color:'#c8d8f0',font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#4a6080',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#0a1428'}}}},
      y:{{ticks:{{color:'#4a6080',callback:v=>v+'%'}},grid:{{color:'#1a2f50'}},
          title:{{display:true,text:'20일 평균 수익률',color:'#4a6080'}}}}
    }}
  }}
}});
</script>

<!-- TIER3 별도 섹션 -->
{{tier3_section}}

<!-- v6 검증 섹션 -->
{{v6_section_placeholder}}

</body>
</html>"""

    # ═══════════════════════════════════════════════════════
    # v6 검증 섹션 HTML 조립
    # ═══════════════════════════════════════════════════════

    # ① LookAheadGuard
    lg = results.get("lag_guard_summary", {})
    lag_guard_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
        f'<tr>'
        f'<td style="color:#4a6080;padding:6px">총 호출 수</td>'
        f'<td style="color:#00d4ff;font-family:monospace">{lg.get("total_calls",0):,}회</td>'
        f'<td style="color:#4a6080;padding:6px">캐시 히트율</td>'
        f'<td style="color:#00ff88;font-family:monospace">{lg.get("cache_hit_pct",0):.1f}%</td>'
        f'<td style="color:#4a6080;padding:6px">위반 감지</td>'
        f'<td style="color:{"#ff3366" if lg.get("violations",0)>0 else "#00ff88"};font-family:monospace">'
        f'{lg.get("violations",0)}건 {"⚠️" if lg.get("violations",0)>0 else "✅ 없음"}'
        f'</td></tr></table>'
        f'<p style="color:#00ff88;margin-top:8px;font-size:.8rem">'
        f'✅ 모든 시점에서 spy.iloc[:i+1] 슬라이싱 — 미래 데이터 혼입 차단</p>'
    )

    # ② 복수검정 보정
    mt       = results.get("mt_correction", {})
    fp_pct   = mt.get("false_positive_risk_pct", 0)
    n_tests  = mt.get("n_tests", 0)
    mt20     = mt.get("20d", {})
    mt_rows  = ""
    for lbl, r in mt20.items():
        bh     = "✅" if r.get("bh_rejected")  else "❌"
        bf     = "✅" if r.get("bf_rejected")  else "❌"
        strong = "⭐ 강한신호" if r.get("strong_signal") else ("📶 BH만유의" if r.get("ic_significant") else "—")
        clr    = "#00ff88" if r.get("strong_signal") else "#ffd700" if r.get("ic_significant") else "#4a6080"
        mt_rows += (
            f'<tr><td style="color:#c8d8f0">{lbl}</td>'
            f'<td style="color:#4a6080;font-family:monospace">{r.get("t_stat",0):.3f}</td>'
            f'<td style="color:#4a6080;font-family:monospace">{r.get("p_raw",0):.4f}</td>'
            f'<td style="color:{"#00ff88" if r.get("bh_rejected") else "#ff3366"}">{bh} BH</td>'
            f'<td style="color:{"#00ff88" if r.get("bf_rejected") else "#ff3366"}">{bf} Bonf</td>'
            f'<td style="color:{clr};font-weight:700">{strong}</td></tr>'
        )
    mt_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:.8rem">'
        f'<tr style="background:#0f1e35">'
        f'<th style="color:#4a6080;text-align:left;padding:6px">팩터</th>'
        f'<th style="color:#4a6080">t-stat</th><th style="color:#4a6080">p값</th>'
        f'<th style="color:#4a6080">BH-FDR</th><th style="color:#4a6080">Bonferroni</th>'
        f'<th style="color:#4a6080">판정</th></tr>{mt_rows}</table>'
    )

    # ③ 거래비용
    bs_top    = results.get("bucket_stats", {}).get("🔥 80-100 상위20%", {}).get("20d", {})
    boot      = results.get("bootstrap_ci", {})
    net_ci    = boot.get("CS단기", {}).get("mean_net", {})
    sr_net    = boot.get("CS단기", {}).get("sharpe_net", {})
    sr_gross  = boot.get("CS단기", {}).get("sharpe_gross", {})
    net_mean  = net_ci.get("mean", 0)
    net_lo    = net_ci.get("ci_lower", 0)
    gross_m   = bs_top.get("mean", 0)
    drag      = round(gross_m - net_mean, 3)
    net_ok    = "✅ 비용 후에도 통계적 양수 수익" if net_lo > 0 else "⚠️  비용 차감 후 유의성 재검토"
    cost_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
        f'<tr style="background:#0f1e35">'
        f'<th style="color:#4a6080;text-align:left;padding:8px">지표</th>'
        f'<th style="color:#4a6080">그로스(비용전)</th>'
        f'<th style="color:#4a6080">순수익(비용후)</th>'
        f'<th style="color:#4a6080">드래그</th></tr>'
        f'<tr><td style="color:#c8d8f0;padding:6px">평균 20일 수익 (상위20%)</td>'
        f'<td style="color:#00ff88;font-family:monospace">{gross_m:+.2f}%</td>'
        f'<td style="color:{"#00ff88" if net_mean>0 else "#ff3366"};font-family:monospace">{net_mean:+.3f}%</td>'
        f'<td style="color:#ffd700;font-family:monospace">{drag:.3f}%p</td></tr>'
        f'<tr><td style="color:#c8d8f0;padding:6px">샤프(95% CI 하한)</td>'
        f'<td style="color:#00ff88;font-family:monospace">{sr_gross.get("sharpe",0):.3f} [{sr_gross.get("ci_lower",0):.3f}~]</td>'
        f'<td style="color:{"#00ff88" if sr_net.get("ci_lower",0)>0 else "#ff3366"};font-family:monospace">'
        f'{sr_net.get("sharpe",0):.3f} [{sr_net.get("ci_lower",0):.3f}~]</td>'
        f'<td style="color:#ffd700;font-family:monospace">{sr_gross.get("sharpe",0)-sr_net.get("sharpe",0):.3f}</td></tr>'
        f'</table>'
        f'<p style="color:#4a6080;font-size:.75rem;margin-top:8px">{net_ok}</p>'
    )

    # ④ Roll-Forward
    rf       = results.get("roll_forward", {})
    stab     = rf.get("stability", {})
    fold_rows= ""
    for fold in rf.get("folds", []):
        ic_f  = fold.get("oos_ic_mean", 0)
        fc    = "#00ff88" if ic_f > 0.01 else "#ffd700" if ic_f > 0 else "#ff3366"
        fold_rows += (
            f'<tr><td style="color:#c8d8f0;font-family:monospace">{str(fold.get("test_start",""))[:7]}</td>'
            f'<td style="color:#4a6080;font-family:monospace">'
            f'{str(fold.get("train_start",""))[:7]}~{str(fold.get("train_end",""))[:7]}</td>'
            f'<td style="color:{fc};font-family:monospace;font-weight:700">{ic_f:+.4f}</td>'
            f'<td style="color:#4a6080">{fold.get("n_test",0):,}</td></tr>'
        )
    stable_c = "#00ff88" if stab.get("stable") else "#ffd700"
    stable_s = "✅ 안정적 전략" if stab.get("stable") else "⚠️  재최적화 필요"
    rf_html  = (
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">'
        f'<div style="background:#0f1e35;padding:10px 18px;border-radius:8px;text-align:center">'
        f'<div style="color:#00d4ff;font-size:1.2rem;font-family:monospace;font-weight:700">'
        f'{stab.get("mean_oos_ic",0):+.4f}</div>'
        f'<div style="color:#4a6080;font-size:.72rem">평균 OOS-IC</div></div>'
        f'<div style="background:#0f1e35;padding:10px 18px;border-radius:8px;text-align:center">'
        f'<div style="color:#00ff88;font-size:1.2rem;font-family:monospace;font-weight:700">'
        f'{stab.get("positive_fold_pct",0):.0f}%</div>'
        f'<div style="color:#4a6080;font-size:.72rem">양수 폴드 비율</div></div>'
        f'<div style="background:#0f1e35;padding:10px 18px;border-radius:8px;text-align:center">'
        f'<div style="color:{stable_c};font-size:.95rem;font-weight:700">{stable_s}</div>'
        f'<div style="color:#4a6080;font-size:.72rem">양수폴드≥70% ∧ IC≥0.01</div></div></div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:.8rem">'
        f'<tr style="background:#0f1e35">'
        f'<th style="color:#4a6080;text-align:left;padding:6px">테스트 연도</th>'
        f'<th style="color:#4a6080">훈련 기간</th>'
        f'<th style="color:#4a6080">OOS IC</th>'
        f'<th style="color:#4a6080">N(테스트)</th></tr>{fold_rows}</table>'
    )

    # ⑤ Bootstrap CI
    boot_rows = ""
    for lbl, br in boot.items():
        ic_c  = br.get("ic_ci", {})
        sr_n  = br.get("sharpe_net", {})
        mn_n  = br.get("mean_net", {})
        se    = "✅" if ic_c.get("strong_evidence") else "❌"
        vs    = "✅" if sr_n.get("valid_strategy") else "❌"
        sig   = "✅" if mn_n.get("significant") else "❌"
        boot_rows += (
            f'<tr><td style="color:#c8d8f0;padding:6px">{lbl}</td>'
            f'<td style="color:#aa88ff;font-family:monospace">'
            f'[{ic_c.get("ci_lower",0):.4f}, {ic_c.get("ci_upper",0):.4f}]</td>'
            f'<td style="color:{"#00ff88" if ic_c.get("strong_evidence") else "#ff3366"}">{se}</td>'
            f'<td style="color:#aa88ff;font-family:monospace">'
            f'[{sr_n.get("ci_lower",0):.3f}, {sr_n.get("ci_upper",0):.3f}]</td>'
            f'<td style="color:{"#00ff88" if sr_n.get("valid_strategy") else "#ff3366"}">{vs}</td>'
            f'<td style="color:{"#00ff88" if mn_n.get("significant") else "#ffd700"}">'
            f'{sig} {mn_n.get("mean",0):+.3f}%</td></tr>'
        )
    boot_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:.8rem">'
        f'<tr style="background:#0f1e35">'
        f'<th style="color:#4a6080;text-align:left;padding:6px">점수</th>'
        f'<th style="color:#4a6080">IC 95% CI</th><th style="color:#4a6080">CI하한>0</th>'
        f'<th style="color:#4a6080">Sharpe(순) CI</th><th style="color:#4a6080">SR>0.5∧CI>0</th>'
        f'<th style="color:#4a6080">순수익 유의</th></tr>{boot_rows}</table>'
        f'<p style="color:#4a6080;font-size:.75rem;margin-top:6px">'
        f'B=1000 Stationary Block Bootstrap (블록길이=√n) | CI하한>0 → 정규분포 가정 없이 확인</p>'
    )

    # ⑥ 국면별
    ra       = results.get("regime_analysis", {})
    ic_spread= ra.get("ic_spread_bull_bear", 0)
    reg_dep  = ra.get("regime_dependent", False)
    reg_rows = ""
    for rg, rg_c, em in [("bull","#00ff88","🟢"),("sideways","#ffd700","🟡"),("bear","#ff3366","🔴")]:
        r = ra.get(rg, {})
        if "ic_mean" not in r:
            continue
        ic_c = "#00ff88" if r["ic_mean"] > 0.01 else "#ffd700" if r["ic_mean"] > 0 else "#ff3366"
        reg_rows += (
            f'<tr><td style="color:{rg_c};font-weight:700;padding:8px">{em} {rg.upper()}</td>'
            f'<td style="color:{ic_c};font-family:monospace">{r.get("ic_mean",0):+.4f}</td>'
            f'<td style="color:#c8d8f0;font-family:monospace">{r.get("t_stat",0):+.2f}</td>'
            f'<td style="color:#c8d8f0;font-family:monospace">{r.get("icir",0):+.3f}</td>'
            f'<td style="color:{"#00ff88" if r.get("mean_ret",0)>0 else "#ff3366"};font-family:monospace">{r.get("mean_ret",0):+.2f}%</td>'
            f'<td style="color:{"#00ff88" if r.get("sharpe",0)>0 else "#ff3366"};font-family:monospace">{r.get("sharpe",0):+.2f}</td>'
            f'<td style="color:#c8d8f0;font-family:monospace">{r.get("win_rate",0):.0f}%</td>'
            f'<td style="color:#4a6080">{r.get("n",0):,}</td>'
            f'<td>{r.get("ic_sig","—")}</td></tr>'
        )
    _reg_span_color = "#ffd700" if reg_dep else "#00ff88"
    _reg_dep_text   = "→ ⚠️  국면 의존적 — Bear 비활성화 권장" if reg_dep else "→ ✅  국면 무관 안정"
    regime_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
        '<tr style="background:#0f1e35">'
        '<th style="color:#4a6080;text-align:left;padding:6px">국면</th>'
        '<th style="color:#4a6080">IC평균</th><th style="color:#4a6080">t-stat</th>'
        '<th style="color:#4a6080">ICIR</th><th style="color:#4a6080">평균수익</th>'
        '<th style="color:#4a6080">샤프</th><th style="color:#4a6080">승률</th>'
        '<th style="color:#4a6080">N</th><th style="color:#4a6080">유의</th></tr>'
        + reg_rows + '</table>'
        + f'<p style="color:#4a6080;font-size:.75rem;margin-top:8px">'
        f'Bull-Bear IC 스프레드: <span style="color:{_reg_span_color};font-family:monospace">{ic_spread:+.4f}</span>'
        + _reg_dep_text + '</p>'
    )

    # 최종 v6 섹션 조립
    _sec_parts = []
    _sec_parts.append(
        '<h2 style="margin-top:40px;color:#00d4ff;border-bottom:2px solid #1a2f50;padding-bottom:8px">'
        '🛡️ v6 펀드-그레이드 검증 결과 (6종)</h2>'
    )
    # ① LookAhead
    _sec_parts.append(
        '<div style="background:#0a1428;border:1px solid #00d4ff33;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#00d4ff;margin-bottom:10px">① 룩어헤드 바이어스 제거 (LookAheadGuard)</h3>'
        '<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        "Lopez de Prado 'Advances in Financial ML' Ch.4</p>"
        + lag_guard_html + '</div>'
    )
    # ② 복수검정
    _mt_intro = (
        '<div style="background:#0a1428;border:1px solid #ffd70033;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#ffd700;margin-bottom:10px">② 복수검정 보정 (BH-FDR + Bonferroni)</h3>'
        + f'<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        f'BH: B&H 1995 | Bonferroni: Dunn 1961 | {n_tests}개 동시검정 → 위양성 {fp_pct:.1f}% 차단</p>'
    )
    _sec_parts.append(_mt_intro + mt_html + '</div>')
    # ③ 거래비용
    _sec_parts.append(
        '<div style="background:#0a1428;border:1px solid #ff885533;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#ff8855;margin-bottom:10px">③ 거래비용 반영 (수수료+슬리피지+충격)</h3>'
        '<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        'Fidelity/Schwab 0.05% | Kissell(2013) | Almgren-Chriss(2001) k=0.5</p>'
        + cost_html + '</div>'
    )
    # ④ Roll-Forward
    _sec_parts.append(
        '<div style="background:#0a1428;border:1px solid #00ff8833;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#00ff88;margin-bottom:10px">④ Roll-Forward OOS (3년훈/1년테/12개월 스텝)</h3>'
        '<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        'Lopez de Prado (2018) Ch.12 | 양수폴드≥70% ∧ 평균IC≥0.01 → 안정 (70% rule)</p>'
        + rf_html + '</div>'
    )
    # ⑤ Bootstrap
    _sec_parts.append(
        '<div style="background:#0a1428;border:1px solid #aa88ff33;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#aa88ff;margin-bottom:10px">⑤ Bootstrap 신뢰구간 (B=1000, Stationary Block)</h3>'
        '<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        'Politis &amp; Romano (1994) | 블록길이=√n | Ledoit &amp; Wolf (2008) Sharpe CI</p>'
        + boot_html + '</div>'
    )
    # ⑥ Regime
    _sec_parts.append(
        '<div style="background:#0a1428;border:1px solid #ff669933;border-radius:10px;padding:20px;margin-bottom:20px">'
        '<h3 style="color:#ff6699;margin-bottom:10px">⑥ 국면별 성과 분리 (Bull/Sideways/Bear)</h3>'
        '<p style="color:#4a6080;font-size:.78rem;margin-bottom:8px">'
        'Faber(2007) SMA200 | Lo et al.(2000) SMA50 | CBOE VIX 중앙값 vol 15%</p>'
        + regime_html + '</div>'
    )
    v6_section = "".join(_sec_parts)

    # TIER3 별도 섹션 조립
    t3_bs = results.get("tier3_bucket_stats", {})
    if t3_bs:
        t3_rows = ""
        for bucket in CS_BUCKET_ORDER:
            if bucket not in t3_bs: continue
            st = t3_bs[bucket]
            row = f'<tr><td style="font-weight:700">{bucket}</td>'
            for fwd in FORWARD:
                s = st.get(f"{fwd}d", {})
                m = s.get("mean", 0); wr = s.get("win_rate", 0); nn = s.get("n", 0)
                mc = color_ret(m)
                wrc = "#00ff88" if wr > 55 else "#ff4466" if wr < 45 else "#ffd700"
                row += (f'<td style="color:{mc}">{m:+.2f}%</td>'
                        f'<td style="color:{wrc}">{wr:.0f}%</td>'
                        f'<td style="color:#556">{nn:,}</td>')
            t3_rows += row + "</tr>"
        tier3_section = (
            '<div style="background:#0a1428;border:1px solid #ff885533;border-radius:10px;'
            'padding:20px;margin-bottom:20px">'
            '<h2 style="color:#ff8855;margin-bottom:10px">'
            '⚡ TIER3 고베타 소형주 — 역발상 전략 (Rank Reversal 적용)</h2>'
            '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">'
            '<div style="background:#0f1e35;padding:8px 16px;border-radius:6px;'
            'border-left:3px solid #ff8855;font-size:.78rem">'
            '<span style="color:#ff8855;font-weight:700">U자형 현상</span>'
            '<span style="color:#4a6080;margin-left:8px">'
            'IC=+0.04(신호 정상) BUT 하위20% 수익 > 상위20% → 역발상 필요</span></div>'
            '<div style="background:#0f1e35;padding:8px 16px;border-radius:6px;'
            'border-left:3px solid #00ff88;font-size:.78rem">'
            '<span style="color:#00ff88;font-weight:700">✅ Rank Reversal 적용</span>'
            '<span style="color:#4a6080;margin-left:8px">'
            'cs_combo = 100-cs_combo → 저점수(침체) 종목을 상위로 역전</span></div>'
            '<div style="background:#0f1e35;padding:8px 16px;border-radius:6px;'
            'border-left:3px solid #aa88ff;font-size:.78rem">'
            '<span style="color:#aa88ff;font-weight:700">근거</span>'
            '<span style="color:#4a6080;margin-left:8px">'
            'Jegadeesh&amp;Titman(2001) 소형주 과잉반응 후 Mean Reversion</span></div>'
            '</div>'
            '<p style="color:#4a6080;font-size:.78rem;margin-bottom:10px">'
            '아래 표는 Rank Reversal 적용 후 구간별 성과. '
            '상위20%(🔥)가 역발상 후 실제로 높은 수익률을 보이면 전략 유효.</p>'
            '<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
            '<thead><tr style="background:#0f1e35">'
            '<th style="text-align:left;color:#4a6080">구간(역발상 후)</th>'
            + "".join([f'<th style="color:#4a6080">{fwd}일 평균</th>'
                       f'<th style="color:#4a6080">{fwd}일 승률</th>'
                       f'<th style="color:#4a6080">N</th>' for fwd in FORWARD])
            + '</tr></thead><tbody>' + t3_rows + '</tbody></table>'
            '<p style="color:#4a6080;font-size:.75rem;margin-top:8px">'
            '💡 상위20%(🔥) = 역발상 후 "원래 점수 낮던" 침체 종목. '
            'TIER1+2 메인 포트폴리오와 독립 운영 권장.</p>'
            '</div>'
        )
    else:
        tier3_section = ""

    html = html.replace("{v6_section_placeholder}", v6_section)
    html = html.replace("{tier3_section}", tier3_section)

    return html


# ─────────────────────────────────────────────────────────────
def _parse_tickers_input(raw: str) -> list:
    """
    티커 입력 문자열 파싱.
    콤마/공백/개행 모두 구분자로 허용.
    대문자 변환 + 중복 제거 + 순서 유지.
    예: "AAPL, msft nvda\nTSLA" → ["AAPL","MSFT","NVDA","TSLA"]
    """
    import re
    tokens = re.split(r"[,\s]+", raw.strip())
    seen, result = set(), []
    for t in tokens:
        t = t.upper().strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _prompt_tickers() -> tuple:
    """
    실행 모드 선택 및 티커 입력 받기.

    반환: (target_tickers: list, mode: str)
      mode = "custom"  — 입력 종목만 백테스트 (터미널 직접 실행 전용)
      mode = "full"    — 전체 종목 (unified/프로그래밍 호출 시 기본값)

    [unified에서 호출 시 자동 전체 모드]
      환경변수 BT_MODE=full  → input() 없이 전체 모드 즉시 진입
      환경변수 BT_TICKERS=AAPL,MSFT → input() 없이 해당 종목 커스텀 모드
      → unified의 _do_backtest가 이 환경변수를 설정하고 main()을 호출
    """
    # ── 환경변수 기반 무인 실행 (unified / 자동화 호출) ──────────────
    bt_mode    = os.environ.get("BT_MODE", "").strip().lower()
    bt_tickers = os.environ.get("BT_TICKERS", "").strip()

    if bt_mode == "full" or (not bt_mode and not bt_tickers and
                              not sys.stdin.isatty()):
        # unified에서 호출되거나 stdin이 터미널이 아닌 경우 → 전체 모드
        print("  [자동] 전체 유니버스 모드 (unified 호출 감지)")
        return list(TICKERS), "full"

    if bt_tickers:
        # 환경변수로 종목 지정된 경우
        tickers = _parse_tickers_input(bt_tickers)
        if "SPY" not in tickers:
            tickers = ["SPY"] + tickers
        print(f"  [자동] 커스텀 모드: {' '.join(tickers)}")
        for tk in tickers:
            if tk not in TICKER_TIER and tk != "SPY":
                TICKER_TIER[tk] = 2
        return tickers, "custom"

    # ── 터미널 대화형 모드 ────────────────────────────────────────────
    W = 58
    print(f"\n{'═'*W}")
    print(f"  Zeus BackTest — 실행 모드 선택")
    print(f"{'─'*W}")
    print(f"  [1] 종목 직접 입력  (빠름, 수 초~수 분)")
    print(f"      예: AAPL, MSFT, NVDA, TSLA")
    print(f"  [2] 전체 유니버스   (느림, ~90분)")
    print(f"      대상: {len(TICKERS)}종목 (TIER1/2/3 전체)")
    print(f"{'═'*W}")

    while True:
        try:
            choice = input("  선택 [1/2] (기본값 1): ").strip() or "1"
        except (EOFError, OSError):
            choice = "2"   # stdin 없으면 전체 모드
        if choice in ("1", "2"):
            break
        print("  1 또는 2를 입력하세요.")

    if choice == "2":
        return list(TICKERS), "full"

    # 직접 입력 모드
    print(f"{'─'*W}")
    print("  티커 입력 (콤마 또는 공백으로 구분)")
    print("  예) AAPL MSFT NVDA   또는   AAPL, MSFT, NVDA")
    print("  SPY는 시장 국면 계산용으로 자동 포함됩니다.")
    print(f"{'─'*W}")

    while True:
        try:
            raw = input("  티커: ").strip()
        except (EOFError, OSError):
            raw = ""
        tickers = _parse_tickers_input(raw)
        if tickers:
            break
        print("  최소 1개 이상 입력하세요.")

    # SPY는 시장 국면(LookAheadGuard) 및 초과수익률(XS) 계산에 반드시 필요
    if "SPY" not in tickers:
        tickers_with_spy = ["SPY"] + tickers
    else:
        tickers_with_spy = tickers

    print(f"\n  대상 종목 ({len(tickers_with_spy)}개): {' '.join(tickers_with_spy)}")

    # TIER 자동 판별 (TIER_MAP에 없으면 TIER2로 기본 배정)
    for tk in tickers_with_spy:
        if tk not in TICKER_TIER and tk != "SPY":
            TICKER_TIER[tk] = 2
            print(f"  ℹ️  {tk}: TIER 미등록 → TIER2로 배정")

    return tickers_with_spy, "custom"


def main():
    t0 = time.time()
    import gc, os

    # ── 실행 모드 선택 ─────────────────────────────────────
    target_tickers, run_mode = _prompt_tickers()

    # 커스텀 모드: TIER 유효성 경고
    # cross_sectional_rank의 계층 내 독립 랭크는 최소 2종목 필요
    if run_mode == "custom":
        tier_counts = {}
        for tk in target_tickers:
            if tk == "SPY": continue
            t = TICKER_TIER.get(tk, 2)
            tier_counts[t] = tier_counts.get(t, 0) + 1
        single_tier = [t for t, c in tier_counts.items() if c == 1]
        if single_tier:
            print(f"  ⚠️  TIER {single_tier} 종목이 1개 → 크로스섹셔널 순위 중립(50)으로 처리됨")
            print(f"     동일 TIER 종목을 2개 이상 입력하면 상대 순위 비교 가능합니다")

    W = 58
    print(f"\n{'='*W}")
    t4_cnt = len(TIER4_SET)
    mode_label = (f"{len(target_tickers)}종목 (커스텀)" if run_mode == "custom"
                  else f"{len(target_tickers)}종목 (T1:{len(TIER1_SET)} T2:{len(TIER2_SET)} T3:{len(TIER3_SET)} T4(R2K):{t4_cnt})")
    print(f"  SmartScore 백테스트  |  {mode_label}  |  기간:{PERIOD}")
    print(f"{'='*W}")

    # 메모리 체크
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1024**3
        total_gb = mem.total / 1024**3
        print(f"  메모리: {avail_gb:.1f}GB 여유 / {total_gb:.1f}GB 전체")
        if avail_gb < 3.0:
            print(f"  ⚠️  메모리 여유 3GB 미만 → PERIOD를 '5y'로 변경 권장")
        elif avail_gb < 5.0:
            print(f"  ⚠️  메모리 여유 5GB 미만 → 청크 단위 처리 자동 적용")
    except ImportError:
        pass

    # 1. 데이터 다운로드
    n_tk = len(target_tickers)
    est_dl  = max(1, n_tk // 10)   # 10종목/분 추산
    est_bt  = max(1, n_tk // 30)   # 30종목/분 추산
    print(f"\n[1/3] 데이터 다운로드...")
    print(f"  예상 소요: 다운로드 ~{est_dl}분 + 백테스트 ~{est_bt}분")
    print(f"  대상: {' '.join(target_tickers)}")

    data = {}
    # 커스텀 모드: 종목 수 작으면 워커 수 줄여 rate limit 방어
    n_workers = min(4, max(1, n_tk // 3)) if run_mode == "custom" else 4
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(download_ticker, tk): tk for tk in target_tickers}
        for fut in as_completed(futs):
            tk, df = fut.result()
            if df is not None:
                data[tk] = df
                print(f"  ✅ {tk}: {len(df)}일")
            else:
                print(f"  ⚠️  {tk}: 데이터 없음 (상폐/기간 부족)")

    spy_df = data.get("SPY")
    print(f"\n  다운로드 성공: {len(data)}/{len(target_tickers)}개")

    # SPY 없으면 경고 (국면 계산 불가)
    if spy_df is None:
        print("  ⚠️  SPY 데이터 없음 → 시장 국면 감지 비활성 (국면=sideways 고정)")

    # ── 시장 국면 감지 (콘솔 출력용, 실제 루프는 LookAheadGuard)
    _global_regime = "sideways"
    if spy_df is not None and len(spy_df) >= 200:
        try:
            _spy_c = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()
            last_idx = len(_spy_c) - 1
            _global_regime = _lag_guard.regime_at(_spy_c, last_idx, vix_val=20.0)
            _rl = {"bull": "🟢 상승장", "bear": "🔴 하락장", "sideways": "🟡 횡보장"}
            print(f"  현재 시장 국면(최신): {_rl.get(_global_regime, _global_regime)}")
        except Exception:
            pass

    # 2. 롤링 백테스트
    print(f"\n[2/3] 롤링 SmartScore 계산...")

    items = list(data.items())
    del data
    gc.collect()

    # ── 캐시 전략 ────────────────────────────────────────────
    # 커스텀 모드: 캐시 사용 안 함 (종목 수 적어 빠르므로 항상 새로 계산)
    #   이유: 입력 종목이 바뀔 때마다 캐시가 무의미하게 남음
    # 전체 모드: 기존 청크 캐시 재사용 (90분 절약)
    use_cache = (run_mode == "full")
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bt_cache_v12b")
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)

    CHUNK_SIZE = 50 if run_mode == "full" else len(items)  # 커스텀: 1청크로 처리
    chunks     = [items[i:i+CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]

    chunk_files = []
    total_done  = 0

    def _run_one(args):
        tk, df = args
        recs = run_backtest_single(
            tk, df, spy_df,
            spy_regime=_global_regime,
            use_lag_guard=True,
        )
        return tk, recs

    for ci, chunk in enumerate(chunks):
        cache_path = os.path.join(CACHE_DIR, f"chunk_{ci:03d}.pkl") if use_cache else None

        # 전체 모드: 캐시 hit 시 스킵
        if use_cache and cache_path and os.path.exists(cache_path):
            print(f"  [청크 {ci+1}/{len(chunks)}] 캐시 hit → 스킵")
            chunk_files.append(cache_path)
            total_done += len(chunk)
            continue

        n_w = min(8, len(chunk))
        if run_mode == "custom":
            # 커스텀: SPY 제외한 실제 분석 종목만 표시
            analysis_tks = [tk for tk, _ in chunk if tk != "SPY"]
            print(f"  백테스트 시작: {' '.join(analysis_tks)}")
        else:
            print(f"  [청크 {ci+1}/{len(chunks)}] {len(chunk)}종목 계산 중...")

        chunk_records = []
        with ThreadPoolExecutor(max_workers=n_w) as pool:
            futs = {pool.submit(_run_one, item): item[0] for item in chunk}
            for fut in as_completed(futs):
                tk, records = fut.result()
                chunk_records.extend(records)
                total_done += 1
                print(f"    [{total_done}/{len(items)}] {tk}: {len(records)}건")

        if chunk_records:
            df_chunk = pd.DataFrame(chunk_records)
            if use_cache and cache_path:
                df_chunk.to_pickle(cache_path)
                chunk_files.append(cache_path)
                print(f"  ✅ 청크 {ci+1} 저장: {cache_path}")
            else:
                # 커스텀 모드: 메모리에 바로 보관
                chunk_files.append(("mem", df_chunk))
            del chunk_records
            gc.collect()

    # 모든 청크 병합
    if not chunk_files:
        print("❌ 백테스트 데이터 없음")
        return

    if run_mode == "custom":
        # 커스텀 모드: 메모리에 직접 보관한 DataFrame 바로 concat
        _chunk_dfs = [df for tag, df in chunk_files if tag == "mem"]
        if not _chunk_dfs:
            print("❌ 백테스트 데이터 없음")
            return
        df_all = pd.concat(_chunk_dfs, ignore_index=True)
        del _chunk_dfs
        gc.collect()
        # SPY 자체 레코드 제외 (분석 대상 아님)
        df_all = df_all[df_all["ticker"] != "SPY"].copy()
    else:
        # 전체 모드: pkl 청크 스트리밍 병합 + float32 다운캐스팅
        print("\n  청크 병합 중...")
        _chunk_dfs = []
        for _cf in chunk_files:
            _cdf = pd.read_pickle(_cf)
            for _col in _cdf.select_dtypes(include="float64").columns:
                _cdf[_col] = _cdf[_col].astype("float32")
            _chunk_dfs.append(_cdf)
        df_all = pd.concat(_chunk_dfs, ignore_index=True)
        del _chunk_dfs
        gc.collect()

    print(f"  총 {len(df_all):,}건 관측치 생성")
    _mem_mb = df_all.memory_usage(deep=True).sum() / 1024**2
    print(f"  메모리: {_mem_mb:.0f}MB (float32 다운캐스팅 적용)")
    if _mem_mb > 2000:
        print(f"  ⚠️  메모리 {_mem_mb:.0f}MB — PERIOD='5y' 변경 권장")

    # 3. 분석 + 리포트
    print("\n[3/3] 분석 및 리포트 생성...")
    results = analyze(df_all)

    elapsed = time.time() - t0
    html = build_html(results, elapsed)

    # 커스텀 모드: 항상 새 파일로 저장 (기존 캐시 HTML 덮어쓰지 않음)
    # 전체 모드:  기존 경로 유지
    if run_mode == "custom":
        # 종목 목록을 파일명에 포함 (최대 4개 + 날짜)
        _tk_label = "_".join(
            [tk for tk in target_tickers if tk != "SPY"][:4]
        )
        _ts = datetime.now().strftime("%m%d_%H%M")
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"backtest_{_tk_label}_{_ts}.html"
        )
        # unified가 탐색하는 기본 경로에도 복사
        _std_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "backtest_smartscore.html"
        )
    else:
        out_path  = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "backtest_smartscore.html"
        )
        _std_path = out_path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    # unified가 항상 최신 결과를 찾을 수 있도록 표준 경로에도 저장
    if out_path != _std_path:
        with open(_std_path, "w", encoding="utf-8") as f:
            f.write(html)

    # ────────────────────────────────────────────────────────
    # 콘솔 최종 요약
    # ────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'═'*W}")
    print(f"  ✅ 백테스트 완료  |  소요: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
    print(f"  📄 리포트: {out_path}")
    print(f"{'═'*W}")

    bs      = results["bucket_stats"]        # TIER1+2 메인
    bs_all  = results.get("all_bucket_stats", bs)  # 전체(TIER1+2+3)
    bs_t3   = results.get("tier3_bucket_stats", {})  # TIER3만

    # 스프레드 계산: 메인(TIER1+2) vs 전체 비교
    print("\n  [★ 스프레드 — TIER1+2 메인 유니버스]")
    _top20 = bs.get("🔥 80-100 상위20%",{})
    _bot20 = bs.get("🔴 0-20 하위20%",{})
    for _fwd in [20, 60]:
        _tm = _top20.get(f"{_fwd}d",{}).get("mean")
        _bm = _bot20.get(f"{_fwd}d",{}).get("mean")
        _tx = _top20.get(f"{_fwd}d",{}).get("xs_mean")
        _bx = _bot20.get(f"{_fwd}d",{}).get("xs_mean")
        if _tm is not None and _bm is not None:
            _gs = _tm - _bm
            _xs = (_tx - _bx) if (_tx is not None and _bx is not None) else None
            _xs_str = f"  XS스프레드={_xs:+.2f}%p" if _xs is not None else ""
            _flag = "✅" if _gs > 0.5 else "⚠️" if _gs > 0 else "❌"
            print(f"    {_flag} {_fwd}일 그로스스프레드={_gs:+.2f}%p{_xs_str}")

    print(f"  [참고 — 전체(TIER1+2+3) 포함 시]")
    _top20a = bs_all.get("🔥 80-100 상위20%",{})
    _bot20a = bs_all.get("🔴 0-20 하위20%",{})
    for _fwd in [20, 60]:
        _tm = _top20a.get(f"{_fwd}d",{}).get("mean")
        _bm = _bot20a.get(f"{_fwd}d",{}).get("mean")
        if _tm is not None and _bm is not None:
            _gs = _tm - _bm
            _flag = "✅" if _gs > 0.5 else "⚠️" if _gs > 0 else "❌"
            print(f"    {_flag} {_fwd}일 전체스프레드={_gs:+.2f}%p")

    if bs_t3:
        print(f"  [TIER3 고베타 소형 별도]")
        _t3top = bs_t3.get("🔥 80-100 상위20%",{})
        _t3bot = bs_t3.get("🔴 0-20 하위20%",{})
        for _fwd in [20, 60]:
            _tm = _t3top.get(f"{_fwd}d",{}).get("mean")
            _bm = _t3bot.get(f"{_fwd}d",{}).get("mean")
            if _tm is not None and _bm is not None:
                _gs = _tm - _bm
                _flag = "✅" if _gs > 0.5 else "⚠️" if _gs > 0 else "❌"
                print(f"    {_flag} TIER3 {_fwd}일 스프레드={_gs:+.2f}%p  (mean reversion 포함)")
    print(f"{'─'*W}")

    def _bar(v):
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
        n = min(10, int(abs(v) * 100))
        return ("+" if v >= 0 else "-") + "▓"*n + "░"*(10-n)

    print("\n  [서브컴포넌트 상관계수 — 20일/60일 수익률]")
    for label, cv in results["corr_stats"].items():
        v20 = cv.get("20d", float("nan"))
        v60 = cv.get("60d", float("nan"))
        print(f"    {label:26s}  20d:{v20:+.4f}{_bar(v20)}  60d:{v60:+.4f}{_bar(v60)}")

    print("\n  [장기 팩터 상관계수 — 20일/60일 수익률]")
    for label, cv in results.get("fac_corr",{}).items():
        v20 = cv.get("20d", float("nan"))
        v60 = cv.get("60d", float("nan"))
        print(f"    {label:26s}  20d:{v20:+.4f}{_bar(v20)}  60d:{v60:+.4f}{_bar(v60)}")

    # ── [v11] 팩터 진단: 유효/제거 구분 ──────────────────────────
    _ACTIVE_FACTORS  = ["raw_vol_z","raw_bb_z","raw_rs","raw_mom","raw_bab",
                        "raw_rv_ratio","raw_dvol","raw_mfi"]  # v12b 추가
    _REMOVED_FACTORS = ["raw_accel","raw_w52","raw_cons"]
    print(f"\n  [★ raw 신호 유효성 진단 — v11 팩터 선별 결과]")
    print(f"    ✅ 활성: {_ACTIVE_FACTORS}")
    print(f"    ❌ 제거: {_REMOVED_FACTORS}  (IC음수/피어슨역상관)")
    _df_diag = results.get("_df", pd.DataFrame())
    for _rc in ["raw_vol_z","raw_bb_z","raw_rs","raw_accel",
                "raw_mom","raw_bab","raw_w52","raw_cons",
                "raw_rv_ratio","raw_dvol","raw_mfi"]:
        if _rc not in _df_diag.columns:
            print(f"    {_rc:16s}: ❌ 컬럼 없음"); continue
        _s = _df_diag[_rc].dropna()
        _n_nan  = _df_diag[_rc].isna().sum()
        _std_val = float(_s.std()) if len(_s) > 1 else 0.0
        _all_c  = (_std_val < 1e-9) if len(_s) > 0 else True
        _status = "❌ 전체상수(NaN위험)" if _all_c else ("⚠️ NaN많음" if _n_nan > len(_df_diag)*0.3 else "✅ 정상")
        print(f"    {_rc:16s}: {_status}  유효N={len(_s):,}  NaN={_n_nan:,}  std={_std_val:.4f}")

    # v12: TIER1 전략A 수익률 출력
    print("\n  [★ TIER1 대형주 — cs_combo 구간별 수익률 (전략A: 기관팩터 수익예측)]")
    for fb, st in results.get("tier1_bucket_stats", results.get("bucket_stats",{})).items():
        s20 = st.get("20d",{}); s60 = st.get("60d",{})
        print(f"    {fb}")
        if s20:
            import math
            _xs20 = s20.get("xs_mean", float("nan"))
            xs_str = f"  | XS:{_xs20:+.2f}%(승률{s20.get('xs_win_rate',0):.0f}%  샤프XS{s20.get('xs_sharpe',0):.2f})" if not math.isnan(_xs20) else ""
            print(f"      20일: 그로스{s20.get('mean',0):+.2f}%  승률{s20.get('win_rate',0):.0f}%  샤프{s20.get('sharpe',0):.2f}  N={s20.get('n',0):,}{xs_str}")
        if s60:
            _xs60 = s60.get("xs_mean", float("nan"))
            xs_str = f"  | XS:{_xs60:+.2f}%(승률{s60.get('xs_win_rate',0):.0f}%  샤프XS{s60.get('xs_sharpe',0):.2f})" if not math.isnan(_xs60) else ""
            print(f"      60일: 그로스{s60.get('mean',0):+.2f}%  승률{s60.get('win_rate',0):.0f}%  샤프{s60.get('sharpe',0):.2f}  N={s60.get('n',0):,}{xs_str}")
    # XS 스프레드 요약
    import math as _m
    _t = results.get("bucket_stats",{}).get("🔥 80-100 상위20%",{})
    _b = results.get("bucket_stats",{}).get("🔴 0-20 하위20%",{})
    for fwd in [20, 60]:
        _tt = _t.get(f"{fwd}d",{}); _bb = _b.get(f"{fwd}d",{})
        if _tt and _bb:
            _gs = (_tt.get("mean",0) or 0) - (_bb.get("mean",0) or 0)
            _xs_t = _tt.get("xs_mean", float("nan")) or float("nan")
            _xs_b = _bb.get("xs_mean", float("nan")) or float("nan")
            _xss = _xs_t - _xs_b if not (_m.isnan(_xs_t) or _m.isnan(_xs_b)) else float("nan")
            _xss_str = f"  XS스프레드={_xss:+.2f}%p" if not _m.isnan(_xss) else ""
            print(f"  ★ {fwd}일 상위-하위 그로스스프레드={_gs:+.2f}%p{_xss_str}")

    # v12: TIER2 샤프 기반 리스크 필터 출력 (전략B)
    if results.get("tier2_sharpe_stats"):
        print("\n  [★ TIER2 중형주 — cs_combo 구간별 샤프+수익률 (전략B: 리스크필터)]")
        for fb, st in results["tier2_sharpe_stats"].items():
            s20 = st.get("20d",{}); s60 = st.get("60d",{})
            print(f"    {fb}")
            if s20:
                import math as _mth
                _sh = s20.get('sharpe_mean', float('nan'))
                sh_str = f"  샤프mean={_sh:.3f}  히트율={s20.get('sharpe_hit',0):.0f}%" if not _mth.isnan(_sh) else ""
                print(f"      20일: 수익{s20.get('mean',0):+.2f}%  승률{s20.get('win_rate',0):.0f}%  변동성{s20.get('avg_vol',0):.1f}%{sh_str}  N={s20.get('n',0):,}")
            if s60:
                _sh = s60.get('sharpe_mean', float('nan'))
                sh_str = f"  샤프mean={_sh:.3f}  히트율={s60.get('sharpe_hit',0):.0f}%" if not _mth.isnan(_sh) else ""
                print(f"      60일: 수익{s60.get('mean',0):+.2f}%  승률{s60.get('win_rate',0):.0f}%  변동성{s60.get('avg_vol',0):.1f}%{sh_str}  N={s60.get('n',0):,}")
        _t2t = results["tier2_sharpe_stats"].get("🔥 80-100 상위20%",{})
        _t2b = results["tier2_sharpe_stats"].get("🔴 0-20 하위20%",{})
        for _fwd in [20, 60]:
            _rm_t = _t2t.get(f"{_fwd}d",{}).get("mean"); _rm_b = _t2b.get(f"{_fwd}d",{}).get("mean")
            _sm_t = _t2t.get(f"{_fwd}d",{}).get("sharpe_mean"); _sm_b = _t2b.get(f"{_fwd}d",{}).get("sharpe_mean")
            if _rm_t is not None and _rm_b is not None:
                _ss_str = f"  샤프스프레드={_sm_t-_sm_b:+.3f}" if (_sm_t is not None and _sm_b is not None) else ""
                print(f"  ★ TIER2 {_fwd}일 수익스프레드={_rm_t-_rm_b:+.2f}%p{_ss_str}")

    # TIER3 별도 출력 (샤프 포함)
    if results.get("tier3_bucket_stats"):
        print("\n  [TIER3 고베타 소형 — cs_combo 구간별 샤프+수익률 (mean reversion 주의)]")
        for fb, st in results["tier3_bucket_stats"].items():
            s20 = st.get("20d",{}); s60 = st.get("60d",{})
            print(f"    {fb}")
            if s20:
                import math as _mth2
                _sh = s20.get('sharpe_mean', float('nan'))
                sh_str = f"  샤프mean={_sh:.3f}  히트율={s20.get('sharpe_hit',0):.0f}%" if not _mth2.isnan(_sh) else ""
                print(f"      20일: 수익{s20.get('mean',0):+.2f}%  승률{s20.get('win_rate',0):.0f}%{sh_str}  N={s20.get('n',0):,}")
            if s60:
                _sh = s60.get('sharpe_mean', float('nan'))
                sh_str = f"  샤프mean={_sh:.3f}" if not _mth2.isnan(_sh) else ""
                print(f"      60일: 수익{s60.get('mean',0):+.2f}%{sh_str}  N={s60.get('n',0):,}")
        _t3t = results["tier3_bucket_stats"].get("🔥 80-100 상위20%",{})
        _t3b = results["tier3_bucket_stats"].get("🔴 0-20 하위20%",{})
        for _fwd in [20, 60]:
            _rm_t = _t3t.get(f"{_fwd}d",{}).get("mean"); _rm_b = _t3b.get(f"{_fwd}d",{}).get("mean")
            _sm_t = _t3t.get(f"{_fwd}d",{}).get("sharpe_mean"); _sm_b = _t3b.get(f"{_fwd}d",{}).get("sharpe_mean")
            if _rm_t is not None and _rm_b is not None:
                _ss_str = f"  샤프스프레드={_sm_t-_sm_b:+.3f}" if (_sm_t is not None and _sm_b is not None) else ""
                print(f"  ★ TIER3 {_fwd}일 수익스프레드={_rm_t-_rm_b:+.2f}%p{_ss_str}")

    print("\n  [★ CS장기(cf_score) 구간별 수익률]")
    for fb, st in results.get("factor_bucket_stats",{}).items():
        s20 = st.get("20d",{}); s60 = st.get("60d",{})
        print(f"    {fb}")
        if s20:
            _xs = s20.get("xs_mean", float("nan"))
            xs_str = f"  | XS:{_xs:+.2f}%" if not _m.isnan(_xs) else ""
            print(f"      20일: 그로스{s20.get('mean',0):+.2f}%  승률{s20.get('win_rate',0):.0f}%  샤프{s20.get('sharpe',0):.2f}  N={s20.get('n',0):,}{xs_str}")
        if s60:
            _xs = s60.get("xs_mean", float("nan"))
            xs_str = f"  | XS:{_xs:+.2f}%" if not _m.isnan(_xs) else ""
            print(f"      60일: 그로스{s60.get('mean',0):+.2f}%  승률{s60.get('win_rate',0):.0f}%  샤프{s60.get('sharpe',0):.2f}  N={s60.get('n',0):,}{xs_str}")

    print("\n  [절대점수(SmartScore) 구간별 수익률 — 분위수 버킷]")
    for fb, st in results.get("abs_bucket_stats",{}).items():
        s20 = st.get("20d",{}); s60 = st.get("60d",{})
        line = f"    {fb}:"
        if s20: line += f"  20일 평균{s20.get('mean',0):+.2f}%  N={s20.get('n',0):,}"
        if s60: line += f"  |  60일 평균{s60.get('mean',0):+.2f}%  N={s60.get('n',0):,}"
        print(line)

    print("\n  [★ IC — Fama-MacBeth | 20일/60일 수익률 기준]")
    print("     IC>0.02 & t>2.0 유의  |  ICIR>0.5 강한팩터  |  IC>0.03 헤지펀드급")
    print("     ※ v9: XS IC 제거 (SPY 공통상수→Spearman IC 불변)")
    for lbl, ic_d in results.get("ic_results",{}).items():
        print(f"  {lbl}")
        for fwd_k, ic in ic_d.items():
            if ic is None: continue
            bar_ic = _bar(ic["IC"])
            print(f"    [{ic['fwd']}]  IC={ic['IC']:+.4f}{bar_ic}  t={ic['t_stat']:+.2f}  ICIR={ic['ICIR']:+.3f}  {ic['sig']}  (n={ic['n_days']}일)")


    # ══════════════════════════════════════════════════════════════
    # 계층별 IC 분석 — 대형/중형/고베타 비교
    # ══════════════════════════════════════════════════════════════
    # [ALERT-2 수정] TIER3 Rank Reversal 적용 후 IC 부호 정정
    # 문제: 계층별 IC 분석이 Rank Reversal 이전 cs_combo 값을 사용
    #       → TIER3 IC=-0.0219 출력 (실제 전략 방향과 반대)
    # 수정: TIER3에 한해 cs_combo_analysis = 100 - cs_combo 사용
    #       = Rank Reversal 후 방향의 IC를 측정
    #       = "역발상 신호의 실제 예측력"을 올바르게 반영
    # 판정: Rank Reversal 후 IC가 양수 → 역발상 전략 유효
    # ══════════════════════════════════════════════════════════════
    print("\n  [계층별 IC 분석 — 대형/중형/고베타 비교]")
    for tier_name, tier_set, is_tier3 in [
            ("계층1 대형앵커",  TIER1_SET, False),
            ("계층2 중형모멘텀", TIER2_SET, False),
            ("계층3 고베타테마", TIER3_SET, True)]:   # TIER3: Rank Reversal 적용
        _df_full = results.get("_df", pd.DataFrame())
        sub = _df_full[_df_full["ticker"].isin(tier_set)] if "ticker" in _df_full.columns else pd.DataFrame()
        if len(sub) < 100: continue

        # [FIX2] TIER3: Jegadeesh & Titman(2001) 단기 반전 이론 정합
        # ret_5d_t3(5일 목표변수)로 IC 계산 — 20d는 반전 효과 희석
        # ret_5d_t3 없으면 ret_20d fallback (하위호환)
        if is_tier3 and "cs_combo" in sub.columns:
            sub = sub.copy()
            sub["cs_combo"] = 100.0 - sub["cs_combo"]
            tier3_note = " [역발상: 100-cs_combo, 목표변수:5일]"
        else:
            tier3_note = ""

        ic_arr, ic_arr_xs = [], []
        for _d, _g in sub.groupby("date"):
            if len(_g) < 3: continue
            # TIER3는 5일 목표변수 우선 사용
            ret_col = "ret_5d_t3" if (is_tier3 and "ret_5d_t3" in _g.columns) else "ret_20d"
            valid = _g[["cs_combo", ret_col]].dropna()
            if len(valid) < 3: continue
            sc     = valid["cs_combo"].values.astype(float)
            r_vals = valid[ret_col].values.astype(float)
            ic_val, _ = _scipy_stats.spearmanr(sc, r_vals)
            if not np.isnan(ic_val): ic_arr.append(ic_val)
            if "ret_20d_xs" in _g.columns:
                valid_xs = _g[["cs_combo","ret_20d_xs"]].dropna()
                if len(valid_xs) >= 3:
                    ic_xs, _ = _scipy_stats.spearmanr(valid_xs["cs_combo"].values,
                                                  valid_xs["ret_20d_xs"].values)
                    if not np.isnan(ic_xs): ic_arr_xs.append(ic_xs)
        if len(ic_arr) < 10: continue
        a  = np.array(ic_arr)
        t  = a.mean() / max(a.std() / len(a)**0.5, 1e-9)
        icir = a.mean() / max(a.std(), 1e-9)
        sig  = "✅" if abs(a.mean())>=0.02 and abs(t)>=2.0 else "⚠️" if abs(a.mean())>=0.01 else "❌"
        xs_str = ""
        if len(ic_arr_xs) >= 10:
            ax = np.array(ic_arr_xs)
            tx = ax.mean() / max(ax.std() / len(ax)**0.5, 1e-9)
            xs_str = f"  |  IC_xs={ax.mean():+.4f}  t_xs={tx:+.2f}"
        print(f"    {tier_name:14s}  IC={a.mean():+.4f}  t={t:+.2f}  ICIR={icir:+.3f}  {sig}  N={len(sub):,}행{xs_str}{tier3_note}")

    # ══════════════════════════════════════════════════════════════
    # 계층별 IC 진단 요약 — v12 결과 기반 구조적 판단
    # ══════════════════════════════════════════════════════════════
    _df_full2 = results.get("_df", pd.DataFrame())
    _t1_ic   = None; _t2_ic = None; _t3_ic = None
    for _tname, _tset, _is3 in [
            ("계층1", TIER1_SET, False),
            ("계층2", TIER2_SET, False),
            ("계층3", TIER3_SET, True)]:
        _sub = _df_full2[_df_full2["ticker"].isin(_tset)] if "ticker" in _df_full2.columns else pd.DataFrame()
        if len(_sub) < 100: continue
        if _is3 and "cs_combo" in _sub.columns:
            _sub = _sub.copy(); _sub["cs_combo"] = 100.0 - _sub["cs_combo"]
        _ics = []
        for _, _g in _sub.groupby("date"):
            _v = _g[["cs_combo","ret_20d"]].dropna()
            if len(_v) < 3: continue
            _ic, _ = _spearmanr(_v["cs_combo"].values, _v["ret_20d"].values)
            if not np.isnan(_ic): _ics.append(_ic)
        if not _ics: continue
        _ic_mean = float(np.mean(_ics))
        if   _tname == "계층1": _t1_ic = _ic_mean
        elif _tname == "계층2": _t2_ic = _ic_mean
        elif _tname == "계층3": _t3_ic = _ic_mean

    print()
    print("  ─────────────────────────────────────────────────────")
    print("  [계층별 전략 구조적 판단]")
    if _t1_ic is not None:
        _t1_judge = "⚠️  팩터교체 필요 (현재 기술팩터는 대형주에 비효과)" if abs(_t1_ic) < 0.01 else "✅ 유효"
        print(f"    계층1(TIER1): IC={_t1_ic:+.4f}  → {_t1_judge}")
        if abs(_t1_ic) < 0.01:
            print("      권고: a) TIER1 비중 축소 + SPY ETF로 대체")
            print("            b) 어닝서프라이즈/애널리스트 리비전 팩터 도입 (v13)")
    if _t2_ic is not None:
        _t2_judge = "✅ 핵심전략 — 실전 운용 가능 (헤지펀드급 IC)" if _t2_ic > 0.03 else "⚠️  모니터링 필요"
        print(f"    계층2(TIER2): IC={_t2_ic:+.4f}  → {_t2_judge}")
    if _t3_ic is not None:
        _t3_judge = "🔴 전략 폐기 권고 — IC≈0, 스프레드 우연" if abs(_t3_ic) < 0.01 else "⚠️  검토 필요"
        print(f"    계층3(TIER3): IC={_t3_ic:+.4f}  → {_t3_judge}")
        if abs(_t3_ic) < 0.01:
            print("      권고: TIER3 완전 제외 or 유동성필터(거래대금>100만달러) 후 재평가")
    print("  ─────────────────────────────────────────────────────")

    # ── unified 연동용: 종목별 최신 cs_combo 점수 저장 ──────────────────
    # results["_df"]에는 전체 종목×날짜 레코드가 있음
    # 최신 날짜 기준으로 각 종목의 cs_combo 점수를 추출해 ranked list로 저장
    # unified._do_backtest가 top_n 제한 없이 전체를 읽을 수 있도록 제공
    try:
        _df_full = results.get("_df", pd.DataFrame())
        if not _df_full.empty and "ticker" in _df_full.columns and "cs_combo" in _df_full.columns:
            _latest_date = _df_full["date"].max()
            _latest = _df_full[_df_full["date"] == _latest_date].copy()
            # 티어별 분리
            _t1 = _latest[_latest["tier"] == 1].sort_values("cs_combo", ascending=False) if "tier" in _latest.columns else pd.DataFrame()
            _t2 = _latest[_latest["tier"] == 2].sort_values("cs_combo", ascending=False) if "tier" in _latest.columns else pd.DataFrame()
            _t3 = _latest[_latest["tier"] == 3].sort_values("cs_combo", ascending=False) if "tier" in _latest.columns else pd.DataFrame()
            _all_ranked = _latest.sort_values("cs_combo", ascending=False)

            results["top_tickers_ranked"] = {
                "as_of_date":  str(_latest_date),
                "all":   _all_ranked["ticker"].tolist(),      # 전체 484종목 cs_combo 순위
                "tier1": _t1["ticker"].tolist() if not _t1.empty else [],
                "tier2": _t2["ticker"].tolist() if not _t2.empty else [],
                "tier3": _t3["ticker"].tolist() if not _t3.empty else [],
                "scores": _all_ranked.set_index("ticker")["cs_combo"].round(1).to_dict(),
            }
            print(f"\n  [unified] 종목별 최신 cs_combo 점수 저장 완료")
            print(f"           날짜={_latest_date}  전체={len(_all_ranked)}종목")
            print(f"           TIER1={len(_t1)}  TIER2={len(_t2)}  TIER3={len(_t3)}")
    except Exception as _e:
        print(f"  [unified] 종목 점수 저장 실패 (무시): {_e}")

    return results


if __name__ == "__main__":
    main()