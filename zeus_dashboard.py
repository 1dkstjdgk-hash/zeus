"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ZEUS MARKET FLOW DASHBOARD v8 — Cross-Sectional Edition                       ║
║                                                                              ║
║  v2 → v3 변경:                                                               ║
║  [1] 자동 새로고침 — Python이 5분마다 HTML 재생성, 브라우저 자동 갱신        ║
║  [2] 섹터 대폭 확장 — 기존 11개 → 20개                                      ║
║       방산(ITA) / 항공우주(XAR) / 데이터센터(DTCR) / 반도체(SOXX)           ║
║       사이버보안(CIBR) / 클라우드(SKYY) / 바이오(XBI) / 리튬(LIT)           ║
║       핀테크(FINX) / 청정에너지(ICLN)                                        ║
║  [3] 장중 실시간 — 5분봉으로 당일 움직임 즉시 반영                           ║
║                                                                              ║
║  사용법:                                                                      ║
║    python market_dashboard_v8.py          → 한 번 실행 후 종료               ║
║    python market_dashboard_v8.py --live   → 5분마다 자동 갱신 (Ctrl+C 종료) ║
║                                                                              ║
║  필요 패키지:                                                                 ║
║    pip install yfinance pandas numpy                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import contextlib
import io
import json  # (아까 추가하신 json도 잘 있는지 확인하시고요!)
import time
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
import os, sys, time, webbrowser, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

# ══════════════════════════════════════════════════════════════
#  v7 A-GRADE MODULES
#  1. AuditLogger          — 신호 생성 근거 감사 추적
#  2. TransactionCostModel — 슬리피지·수수료 현실화
#  3. RiskManager          — 포지션 사이징·손절 기준
#  4. PortfolioConstructor — 상관관계·분산·포트폴리오 구성
#  5. StressTestEngine     — 2008/2020 테일리스크 시나리오
# ══════════════════════════════════════════════════════════════

import pickle, hashlib, threading
from collections import deque

# ──────────────────────────────────────────────────────────────
#  1. AuditLogger — 신호 생성 근거 감사 추적
# ──────────────────────────────────────────────────────────────
class AuditLogger:
    """
    모든 SmartScore 신호 생성 근거를 구조화된 로그로 기록.

    기록 항목:
      - 신호 생성 시각 / 티커 / 점수 / 각 컴포넌트 세부값
      - 시장 국면 (regime) / 크로스섹셔널 랭킹
      - 트리거된 매집 태그
      - 최근 N개 로그 메모리 보관 + JSON 파일 저장
    """
    _instance = None
    _lock = threading.Lock()
    LOG_FILE = "zeus_audit_log.json"
    MAX_MEMORY = 2000   # 메모리 최대 보관 건수

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._records = deque(maxlen=cls.MAX_MEMORY)
                cls._instance._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                cls._instance._log_count = 0
                cls._instance._flush_every = 50   # 50건마다 디스크 플러시
        return cls._instance

    def log_signal(self, ticker: str, score_result: dict,
                   regime: str, cs_score: float = None,
                   price: float = None, atr_pct: float = None):
        """
        SmartScore calc() 결과를 감사 로그로 기록.
        score_result: SmartScoreEngine.calc() 반환값 전체
        """
        ts = datetime.now().isoformat(timespec="seconds")
        details = score_result.get("details", {})
        raw     = score_result.get("raw", {})

        record = {
            "ts":       ts,
            "session":  self._session_id,
            "ticker":   ticker,
            "regime":   regime,
            # ── 총점 ──
            "total_score":  round(float(score_result.get("total", 0)), 1),
            "factor_score": round(float(score_result.get("factor", {}).get("total", 0)), 1),
            "cs_score":     round(float(cs_score), 1) if cs_score is not None else None,
            "price":        round(float(price), 4) if price is not None else None,
            "atr_pct":      round(float(atr_pct), 3) if atr_pct is not None else None,
            # ── 컴포넌트별 점수 ──
            "components": {
                "RS":       round(float(details.get("RS",        {}).get("score", 0)), 1),
                "vol":      round(float(details.get("거래량구조", {}).get("score", 0)), 1),
                "accum":    round(float(details.get("매집탐지",   {}).get("score", 0)), 1),
                "obv":      round(float(details.get("OBV",        {}).get("score", 0)), 1),
                "momentum": round(float(details.get("모멘텀가속", {}).get("score", 0)), 1),
                "bb":       round(float(details.get("BB압축",     {}).get("score", 0)), 1),
                "trend":    round(float(details.get("추세일관성", {}).get("score", 0)), 1),
                "option":   round(float(details.get("옵션신호",   {}).get("score", 0)), 1),
            },
            # ── 핵심 raw 값 ──
            "raw": {k: round(float(v), 4) for k, v in raw.items()
                    if isinstance(v, (int, float)) and not (v != v)},  # NaN 제외
            # ── 매집 태그 ──
            "accum_tags": list(details.get("매집탐지", {}).get("tags", [])),
            # ── RS 세부 ──
            "vs_spy":     round(float(details.get("RS", {}).get("vs_spy", 0)), 2),
            "vs_sector":  round(float(details.get("RS", {}).get("vs_sector", 0)), 2),
            # ── 등급 ──
            "grade":      str(score_result.get("grade", "")),
            "ret_1d":     round(float(score_result.get("ret_1d", 0)), 2),
        }

        with self._lock:
            self._records.append(record)
            self._log_count += 1
            if self._log_count % self._flush_every == 0:
                self._flush_to_disk()

    def log_risk_decision(self, ticker: str, action: str,
                          position_size: float, stop_loss: float,
                          reason: str, score: float = None):
        """포지션 진입/청산 결정 로그."""
        record = {
            "ts":             datetime.now().isoformat(timespec="seconds"),
            "type":           "RISK_DECISION",
            "ticker":         ticker,
            "action":         action,       # BUY / SELL / STOP / SIZE_REDUCE
            "position_size":  round(float(position_size), 4),
            "stop_loss_pct":  round(float(stop_loss), 4),
            "reason":         reason,
            "score":          round(float(score), 1) if score is not None else None,
        }
        with self._lock:
            self._records.append(record)

    def log_stress_test(self, scenario: str, result: dict):
        """스트레스 테스트 결과 로그."""
        record = {
            "ts":       datetime.now().isoformat(timespec="seconds"),
            "type":     "STRESS_TEST",
            "scenario": scenario,
            "result":   result,
        }
        with self._lock:
            self._records.append(record)

    def _flush_to_disk(self):
        """비동기적으로 JSON 파일에 추가 저장."""
        try:
            existing = []
            if os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            all_records = existing + list(self._records)
            # 최대 10,000건 유지
            if len(all_records) > 10000:
                all_records = all_records[-10000:]
            with open(self.LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=None,
                          separators=(",", ":"))
        except Exception as _e:
            pass  # 로그 실패가 메인 로직을 막으면 안 됨

    def get_recent(self, n: int = 100, ticker: str = None) -> list:
        """최근 N건 반환. ticker 지정 시 해당 종목만."""
        with self._lock:
            records = list(self._records)
        if ticker:
            records = [r for r in records if r.get("ticker") == ticker]
        return records[-n:]

    def get_top_signals(self, min_score: float = 70, limit: int = 20) -> list:
        """고점수 신호만 필터링."""
        with self._lock:
            records = list(self._records)
        filtered = [r for r in records
                    if isinstance(r.get("total_score"), (int, float))
                    and r["total_score"] >= min_score]
        return sorted(filtered, key=lambda x: x["total_score"], reverse=True)[:limit]

    def get_summary_html(self) -> str:
        """감사 로그 요약 HTML 섹션 생성."""
        with self._lock:
            records = [r for r in self._records
                       if r.get("type") not in ("RISK_DECISION", "STRESS_TEST")]
        if not records:
            return '<div class="audit-empty">감사 로그 없음 (이번 세션 첫 실행)</div>'

        total   = len(records)
        high_ct = sum(1 for r in records if r.get("total_score", 0) >= 70)
        med_ct  = sum(1 for r in records if 55 <= r.get("total_score", 0) < 70)
        avg_sc  = sum(r.get("total_score", 0) for r in records) / max(total, 1)

        # 최근 고점수 신호 5개
        top5 = sorted(
            [r for r in records if r.get("total_score", 0) >= 55],
            key=lambda x: x["total_score"], reverse=True
        )[:5]

        top_rows = ""
        for r in top5:
            tags = ", ".join(r.get("accum_tags", [])) or "—"
            comp = r.get("components", {})
            top_rows += f"""
            <tr>
              <td style="color:var(--accent);font-weight:700">{r['ticker']}</td>
              <td style="color:var(--green);font-weight:700">{r['total_score']:.0f}</td>
              <td style="color:#8b949e">{r['regime']}</td>
              <td style="color:var(--yellow);font-size:.75rem">{tags}</td>
              <td style="color:#8b949e;font-size:.72rem">{r['ts'][11:19]}</td>
              <td style="color:#bc8cff;font-size:.75rem">
                RS:{comp.get('RS',0):.0f} V:{comp.get('vol',0):.0f}
                A:{comp.get('accum',0):.0f} O:{comp.get('obv',0):.0f}
              </td>
            </tr>"""

        return f"""
        <div class="audit-summary">
          <div class="audit-stats">
            <div class="audit-stat">
              <span class="audit-val">{total}</span>
              <span class="audit-lbl">이번 세션 로그</span>
            </div>
            <div class="audit-stat">
              <span class="audit-val" style="color:var(--green)">{high_ct}</span>
              <span class="audit-lbl">강매집 신호(70+)</span>
            </div>
            <div class="audit-stat">
              <span class="audit-val" style="color:var(--yellow)">{med_ct}</span>
              <span class="audit-lbl">관심 신호(55~70)</span>
            </div>
            <div class="audit-stat">
              <span class="audit-val">{avg_sc:.1f}</span>
              <span class="audit-lbl">평균 점수</span>
            </div>
          </div>
          <table class="audit-table">
            <thead>
              <tr>
                <th>티커</th><th>점수</th><th>국면</th>
                <th>매집태그</th><th>생성시각</th><th>컴포넌트 (RS/V/A/O)</th>
              </tr>
            </thead>
            <tbody>{top_rows}</tbody>
          </table>
          <div style="font-size:.72rem;color:var(--text-dim);margin-top:6px">
            📁 전체 로그: zeus_audit_log.json ({total}건 기록됨)
          </div>
        </div>"""


# ──────────────────────────────────────────────────────────────
#  2. TransactionCostModel — 슬리피지·수수료 현실화
# ──────────────────────────────────────────────────────────────
class TransactionCostModel:
    """
    실거래 비용 현실적 추정.

    수수료 계층:
      - 기관급 DMA:   0.02% 편도 (로빈후드/IB Pro급)
      - 리테일 표준:  0.05% 편도 (기본값)
      - 리테일 고가:  0.10% 편도

    슬리피지 모델 (Almgren-Chriss 단순화):
      - 대형주 (일거래량 > 5M): 0.05%
      - 중형주 (1M ~ 5M):       0.10%
      - 소형주 (< 1M):          0.20~0.50% (변동성 비례)

    시장충격 (Market Impact):
      - 주문 규모가 일거래량 대비 비율에 따라 추가 비용
      - impact = k * sqrt(order_size / avg_daily_volume) * volatility
      - k = 0.5 (경험 상수, Almgren-Chriss 1999)
    """
    # 수수료 (편도 %)
    COMMISSION = {
        "institutional": 0.02,
        "retail":        0.05,
        "retail_high":   0.10,
    }
    # 슬리피지 기준 (편도 %, 거래량 기반)
    SLIPPAGE_BASE = {
        "large":  0.05,   # 일거래량 > 5M
        "mid":    0.10,   # 1M ~ 5M
        "small":  0.20,   # 0.3M ~ 1M
        "micro":  0.50,   # < 0.3M
    }

    def __init__(self, account_type: str = "retail"):
        self.commission_rate = self.COMMISSION.get(account_type, 0.05)

    def estimate_cost(self,
                      price: float,
                      avg_daily_volume: float,
                      order_shares: float = None,
                      order_value: float = None,
                      volatility_pct: float = 2.0,
                      side: str = "buy") -> dict:
        """
        단방향 거래비용 추정.

        Parameters
        ----------
        price            : 현재 주가
        avg_daily_volume : 20일 평균 거래량 (주수)
        order_shares     : 주문 수량 (shares). None이면 order_value 사용
        order_value      : 주문 금액 ($). None이면 소규모 가정
        volatility_pct   : 일간 변동성 % (기본 2%)
        side             : "buy" or "sell"

        Returns
        -------
        dict with: commission_pct, slippage_pct, impact_pct, total_pct,
                   total_dollar (order_value 기준), cost_class
        """
        if order_value is None and order_shares is not None:
            order_value = order_shares * price
        elif order_value is None:
            order_value = price * 100   # 기본 100주

        order_shares = order_value / max(price, 0.01)
        adv_dollar   = avg_daily_volume * price   # 일 거래대금

        # 거래량 계층
        if avg_daily_volume >= 5_000_000:
            slip_base = self.SLIPPAGE_BASE["large"]
        elif avg_daily_volume >= 1_000_000:
            slip_base = self.SLIPPAGE_BASE["mid"]
        elif avg_daily_volume >= 300_000:
            slip_base = self.SLIPPAGE_BASE["small"]
        else:
            slip_base = self.SLIPPAGE_BASE["micro"]

        # 변동성 보정 슬리피지 (고변동 종목은 슬리피지 더 큼)
        vol_adj_slip = slip_base * max(1.0, volatility_pct / 2.0)

        # 시장 충격 — Almgren-Chriss 단순화
        # participation_rate = 주문 / 일거래량
        participation = min(order_shares / max(avg_daily_volume, 1), 0.30)
        impact_pct    = 0.5 * (participation ** 0.5) * volatility_pct

        comm_pct  = self.commission_rate
        slip_pct  = vol_adj_slip
        total_pct = comm_pct + slip_pct + impact_pct

        # 비용 등급
        if total_pct < 0.15:
            cost_class = "LOW"
        elif total_pct < 0.35:
            cost_class = "MEDIUM"
        elif total_pct < 0.70:
            cost_class = "HIGH"
        else:
            cost_class = "VERY_HIGH"

        return {
            "commission_pct": round(comm_pct, 4),
            "slippage_pct":   round(slip_pct, 4),
            "impact_pct":     round(impact_pct, 4),
            "total_pct":      round(total_pct, 4),
            "total_dollar":   round(order_value * total_pct / 100, 2),
            "roundtrip_pct":  round(total_pct * 2, 4),   # 왕복 비용
            "cost_class":     cost_class,
            "breakeven_move": round(total_pct * 2 / 100, 6),  # 손익분기점 이동
        }

    def net_return(self, gross_return_pct: float,
                   cost_dict: dict) -> float:
        """순수익률 = 총수익률 - 왕복거래비용."""
        return gross_return_pct - cost_dict["roundtrip_pct"]

    def min_score_threshold(self, cost_dict: dict,
                             win_rate: float = 0.55,
                             avg_win_pct: float = 5.0,
                             avg_loss_pct: float = 3.0) -> float:
        """
        거래비용을 넘기 위한 최소 기대수익 조건.
        반환: 손익분기 SmartScore 추정 (단순 선형 근사)
        """
        breakeven = cost_dict["roundtrip_pct"]
        expected  = win_rate * avg_win_pct - (1 - win_rate) * avg_loss_pct
        # 기대수익이 거래비용보다 높아야 진입 가치 있음
        # SmartScore 55 = 기대수익 약 2~3% 가정 → 비례 역산
        if expected <= 0:
            return 100.0  # 진입 불가
        # 필요 기대수익 비율
        needed_ratio = breakeven / expected
        # 기준 점수 55에서 비례 상향
        min_score = 55 + needed_ratio * 20
        return round(min(min_score, 85), 1)

    def get_html_badge(self, cost_dict: dict) -> str:
        """비용 수준 HTML 배지."""
        cls_map = {
            "LOW":       ("var(--green)", "저비용"),
            "MEDIUM":    ("var(--yellow)", "중비용"),
            "HIGH":      ("#ff8844", "고비용"),
            "VERY_HIGH": ("#ff4466", "극고비용"),
        }
        color, label = cls_map.get(cost_dict["cost_class"], ("#888", "?"))
        rt = cost_dict["roundtrip_pct"]
        return (f'<span style="font-size:.7rem;color:{color};border:1px solid {color};'
                f'border-radius:3px;padding:1px 5px;margin-left:4px">'
                f'비용 {rt:.2f}% 왕복</span>')


# ──────────────────────────────────────────────────────────────
#  3. RiskManager — 포지션 사이징 & 손절 기준
# ──────────────────────────────────────────────────────────────
class RiskManager:
    """
    기관급 리스크 관리 모듈.

    포지션 사이징 방법론:
      A) ATR 기반 (기본): position = risk_per_trade / (N * ATR)
         - N: ATR 배수 (손절 너비), 기본 2
         - risk_per_trade: 계좌의 1~2% (기본 1%)
         → Turtle Trading / AQR 표준

      B) Kelly Criterion (보조):
         f* = (p*b - q) / b
         - p: 승률, b: 평균이익/평균손실 비율
         - 반켈리(f*/2) 사용 (풀켈리는 변동성 과대)

      C) Volatility Parity (포트폴리오):
         각 종목 포지션 = target_vol / 종목_연변동성
         → 모든 종목이 동일 변동성 기여

    손절 기준:
      - ATR 손절: 진입가 - N * ATR (기본 N=2)
      - 시간 손절: N일 이내 목표 미달성 시 청산
      - 손실 한도: 포지션 최대 -8% (하드 스탑)
    """

    def __init__(self,
                 account_size: float = 100_000,
                 risk_per_trade_pct: float = 1.0,
                 max_position_pct: float = 10.0,
                 atr_multiplier: float = 2.0,
                 hard_stop_pct: float = 8.0):
        """
        account_size       : 계좌 총액 ($)
        risk_per_trade_pct : 거래당 최대 손실 허용 (계좌 대비 %)
        max_position_pct   : 단일 종목 최대 비중 (%)
        atr_multiplier     : 손절 너비 = N * ATR
        hard_stop_pct      : 절대 손실 한도 (%)
        """
        self.account_size        = account_size
        self.risk_per_trade      = account_size * risk_per_trade_pct / 100
        self.max_position_value  = account_size * max_position_pct  / 100
        self.atr_mult            = atr_multiplier
        self.hard_stop_pct       = hard_stop_pct

    def atr_position_size(self,
                           price: float,
                           atr: float,
                           smart_score: float = 55.0,
                           cost_total_pct: float = 0.15) -> dict:
        """
        ATR 기반 포지션 사이징.

        고점수 종목에 더 많은 비중 배분:
          score_mult = 0.5 + (score - 40) / 100   (score=40→0.5배, score=80→0.9배)
          단, 과도한 레버리지 방지: 최대 1.0배

        Parameters
        ----------
        price         : 현재 주가
        atr           : ATR (14일 기본)
        smart_score   : SmartScore (점수 비례 사이징)
        cost_total_pct: 편도 거래비용 % (손절 너비 보정용)

        Returns
        -------
        dict with: shares, position_value, stop_price, stop_pct,
                   risk_amount, score_multiplier, sizing_method
        """
        if atr <= 0 or price <= 0:
            return self._zero_position("ATR/Price 0")

        # 점수 비례 배율 (0.0 ~ 1.0)
        # 근거: _cs_grade 기준 score<40 = '중립/하위' → 진입 기준 미달 → 0배
        #       score=40(중립하한) → 0.5배, score=100 → 1.0배 선형 매핑
        # 공식: 0.5 + (score-40)/60 * 0.5  (score 40~100 → mult 0.5~1.0)
        if smart_score < 40:
            return self._zero_position(f"점수 미달({smart_score:.0f}<40)")
        score_mult = max(0.5, min(1.0, 0.5 + (smart_score - 40) / 60 * 0.5))

        # 손절 너비: ATR × N + 거래비용 (비용도 손실이므로 포함)
        stop_width  = atr * self.atr_mult
        stop_width += price * cost_total_pct / 100  # 거래비용 보정

        # 포지션 수량 = (허용손실 × 점수배율) / 손절너비
        risk_adj    = self.risk_per_trade * score_mult
        raw_shares  = risk_adj / max(stop_width, price * 0.005)
        raw_value   = raw_shares * price

        # 최대 포지션 한도 적용
        capped_value  = min(raw_value, self.max_position_value)
        final_shares  = int(capped_value / price)
        final_value   = final_shares * price

        stop_price    = price - stop_width
        stop_pct      = stop_width / price * 100

        return {
            "shares":           final_shares,
            "position_value":   round(final_value, 2),
            "position_pct":     round(final_value / self.account_size * 100, 2),
            "stop_price":       round(stop_price, 2),
            "stop_pct":         round(stop_pct, 2),
            "hard_stop_pct":    self.hard_stop_pct,
            "risk_amount":      round(final_shares * stop_width, 2),
            "score_multiplier": round(score_mult, 2),
            "sizing_method":    "ATR",
            "atr_used":         round(float(atr), 4),
            "valid":            final_shares > 0,
        }

    def kelly_position_size(self,
                             price: float,
                             win_rate: float = 0.55,
                             avg_win_pct: float = 5.0,
                             avg_loss_pct: float = 3.0,
                             fraction: float = 0.5) -> dict:
        """
        Kelly Criterion 기반 포지션 사이징.
        fraction=0.5: 반켈리 (표준 리스크 관리 관행)
        """
        if avg_loss_pct <= 0:
            return self._zero_position("손실% 0")
        b = avg_win_pct / avg_loss_pct   # 이익/손실 비율
        q = 1 - win_rate
        f_star = (win_rate * b - q) / b  # Kelly f*
        f_used  = f_star * fraction       # 반켈리

        if f_used <= 0:
            return self._zero_position(f"Kelly음수({f_star:.3f})")

        kelly_value  = self.account_size * f_used
        kelly_value  = min(kelly_value, self.max_position_value)
        kelly_shares = int(kelly_value / max(price, 0.01))

        return {
            "shares":         kelly_shares,
            "position_value": round(kelly_shares * price, 2),
            "position_pct":   round(kelly_shares * price / self.account_size * 100, 2),
            "kelly_f_star":   round(f_star, 4),
            "kelly_f_used":   round(f_used, 4),
            "win_rate":       win_rate,
            "avg_win":        avg_win_pct,
            "avg_loss":       avg_loss_pct,
            "sizing_method":  "Kelly_Half",
            "valid":          kelly_shares > 0,
        }

    def vol_parity_weight(self,
                           annual_vol_pct: float,
                           target_vol_pct: float = None,
                           regime: str = "sideways") -> float:
        """
        단일 종목 Volatility-Scaled 비중 (참고용).

        정의: 종목의 연변동성이 target_vol_pct일 때 비중 = max_position_pct
              변동성이 높을수록 비중을 낮춰 각 종목의 변동성 기여를 동일화.

        공식: weight = (target_vol / annual_vol) * max_position_pct
        근거: 변동성 15% 기준 종목이 max_position(10%)에 배정될 때,
              변동성 30% 종목은 5%, 변동성 7.5% 종목은 10%(캡)으로 스케일.

        target_vol 국면 조건부 (AQR 관행):
          bull    → 20%  (리스크 선호 → 더 큰 포지션 허용)
          sideways→ 15%  (기본값)
          bear    → 10%  (리스크 회피 → 포지션 축소, VIX>30 환경)

        ※ 정확한 포트폴리오 Vol-Parity는 PortfolioConstructor.portfolio_vol()
           사용 권장. 이 메서드는 단일 종목 참고용.

        반환: 계좌 대비 권장 비중 (%)
        """
        if annual_vol_pct <= 0:
            return 0.0
        # 국면별 target_vol (AQR Risk-Parity 관행)
        if target_vol_pct is None:
            target_vol_pct = {"bull": 20.0, "sideways": 15.0, "bear": 10.0}.get(regime, 15.0)
        max_pct = self.max_position_value / self.account_size * 100  # 10%
        weight = (target_vol_pct / annual_vol_pct) * max_pct
        return round(min(weight, max_pct), 2)

    def evaluate_stop(self,
                       current_price: float,
                       entry_price: float,
                       atr_stop: float,
                       days_held: int = 0,
                       time_stop_days: int = 20) -> dict:
        """
        현재 가격에서 손절 조건 평가.
        반환: {triggered, reason, pnl_pct, action}
        """
        pnl_pct = (current_price / entry_price - 1) * 100
        triggered = False
        reason    = ""

        # 하드 스탑
        if pnl_pct <= -self.hard_stop_pct:
            triggered = True
            reason    = f"하드스탑 -{self.hard_stop_pct}% 도달"

        # ATR 손절
        elif current_price <= atr_stop:
            triggered = True
            reason    = f"ATR손절 ${atr_stop:.2f} 이탈"

        # 시간 손절
        elif days_held >= time_stop_days and pnl_pct < 0:
            triggered = True
            reason    = f"시간손절 {time_stop_days}일 경과 + 손실 중"

        return {
            "triggered":    triggered,
            "reason":       reason,
            "pnl_pct":      round(pnl_pct, 2),
            "action":       "SELL" if triggered else "HOLD",
            "days_held":    days_held,
        }

    def _zero_position(self, reason: str) -> dict:
        return {"shares": 0, "position_value": 0, "position_pct": 0,
                "stop_price": 0, "stop_pct": 0, "hard_stop_pct": self.hard_stop_pct,
                "risk_amount": 0, "score_multiplier": 0,
                "sizing_method": "ZERO", "reason": reason, "valid": False}

    def get_position_html(self, ticker: str, price: float, atr: float,
                           smart_score: float, avg_vol: float,
                           annual_vol_pct: float) -> str:
        """종목 포지션 사이징 HTML 카드 생성."""
        from_atr = self.atr_position_size(price, atr, smart_score)
        cost_m   = TransactionCostModel()
        cost     = cost_m.estimate_cost(price, avg_vol,
                                         order_value=from_atr["position_value"],
                                         volatility_pct=annual_vol_pct / (252 ** 0.5))
        vol_w    = self.vol_parity_weight(annual_vol_pct)  # regime 기본값 sideways

        if not from_atr["valid"]:
            return f'<div class="risk-card-empty">{ticker}: 포지션 사이징 불가</div>'

        stop_c   = "#ff4466"
        size_c   = "var(--green)" if from_atr["position_pct"] >= 3 else "var(--yellow)"
        return f"""
        <div class="risk-card">
          <div class="risk-header">
            <span class="risk-ticker">{ticker}</span>
            <span class="risk-score" style="color:{'var(--green)' if smart_score>=70 else 'var(--yellow)'}">{smart_score:.0f}점</span>
            {cost_m.get_html_badge(cost)}
          </div>
          <div class="risk-body">
            <div class="risk-row">
              <span class="risk-lbl">ATR 사이징</span>
              <span class="risk-val" style="color:{size_c}">{from_atr['shares']}주 / ${from_atr['position_value']:,.0f} ({from_atr['position_pct']:.1f}%)</span>
            </div>
            <div class="risk-row">
              <span class="risk-lbl">손절가 (ATR×{self.atr_mult})</span>
              <span class="risk-val" style="color:{stop_c}">${from_atr['stop_price']:,.2f} (-{from_atr['stop_pct']:.1f}%)</span>
            </div>
            <div class="risk-row">
              <span class="risk-lbl">하드 스탑</span>
              <span class="risk-val" style="color:{stop_c}">-{self.hard_stop_pct}%</span>
            </div>
            <div class="risk-row">
              <span class="risk-lbl">왕복 거래비용</span>
              <span class="risk-val" style="color:#ff8844">{cost['roundtrip_pct']:.2f}% (${cost['total_dollar']*2:,.1f})</span>
            </div>
            <div class="risk-row">
              <span class="risk-lbl">Vol-Parity 비중</span>
              <span class="risk-val" style="color:#bc8cff">{vol_w:.1f}%</span>
            </div>
            <div class="risk-row">
              <span class="risk-lbl">손익분기점</span>
              <span class="risk-val" style="color:#8b949e">{cost['breakeven_move']*100:.2f}% 이상 상승 필요</span>
            </div>
          </div>
        </div>"""


# ──────────────────────────────────────────────────────────────
#  4. PortfolioConstructor — 상관관계·분산·포트폴리오 구성
# ──────────────────────────────────────────────────────────────
class PortfolioConstructor:
    """
    포트폴리오 수준 리스크 분석.

    기능:
      - 종목 간 상관 행렬 계산 (60일 수익률 기반)
      - 섹터 집중도 Herfindahl-Hirschman Index (HHI)
      - 분산 효과 측정 (포트폴리오 변동성 vs 가중평균 개별 변동성)
      - 유효 종목 수 (Effective N = 1 / sum(w_i^2))
      - 상관 과집중 경고 (평균 상관 > 0.6 시 포트폴리오 분산 무효)
    """

    def __init__(self, max_sector_weight: float = 30.0,
                 max_corr_threshold: float = 0.70):
        """
        max_sector_weight  : 단일 섹터 최대 비중 (%)
        max_corr_threshold : 이 이상 상관된 종목 쌍 경고
        """
        self.max_sector_weight  = max_sector_weight
        self.max_corr_threshold = max_corr_threshold

    def build_correlation_matrix(self, price_data: dict,
                                  lookback: int = 60) -> dict:
        """
        종목 간 상관 행렬 계산.

        Parameters
        ----------
        price_data : {ticker: pd.DataFrame} — DataFetcher._price_cache 형식
        lookback   : 수익률 계산 기간 (일)

        Returns
        -------
        dict with: matrix (DataFrame), high_corr_pairs, avg_corr,
                   diversification_ok
        """
        # 수익률 시리즈 수집
        ret_dict = {}
        for tk, cache_entry in price_data.items():
            try:
                df = cache_entry.get("df") if isinstance(cache_entry, dict) else cache_entry
                if df is None or df.empty:
                    continue
                c = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if len(c) >= lookback + 5:
                    ret_dict[tk] = c.pct_change().dropna().tail(lookback)
            except Exception as _e:
                logger.debug("항목 스킵: %s", _e)
                continue

        if len(ret_dict) < 2:
            return {"matrix": None, "high_corr_pairs": [], "avg_corr": 0.0,
                    "diversification_ok": True, "n_stocks": len(ret_dict)}

        # 공통 날짜 기준 정렬
        # ... (ret_dict에 데이터를 모으는 for문 끝) ...

    # 🎯 [핵심 추가] 데이터프레임으로 합치기 전, 모든 시리즈의 타임존을 일괄 제거 (세탁)
        clean_dict = {}
        for ticker, series in ret_dict.items():
            if series is not None and hasattr(series, 'index'):
                if series.index.tz is not None:
                    clean_dict[ticker] = series.tz_localize(None) # 시간표 제거
                else:
                    clean_dict[ticker] = series
            else:
                clean_dict[ticker] = series

    # 기존 ret_dict 대신 세탁된 clean_dict를 사용하여 병합
        ret_df = pd.DataFrame(clean_dict).dropna(how="all", axis=1).dropna(how="any", axis=0)
    # ...

        if ret_df.shape[1] < 2:
            return {"matrix": None, "high_corr_pairs": [], "avg_corr": 0.0,
                    "diversification_ok": True, "n_stocks": ret_df.shape[1]}

        corr_matrix = ret_df.corr()

        # 고상관 종목 쌍 추출
        high_pairs = []
        tickers = corr_matrix.columns.tolist()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                c_val = float(corr_matrix.iloc[i, j])
                if c_val >= self.max_corr_threshold:
                    high_pairs.append({
                        "ticker_a": tickers[i],
                        "ticker_b": tickers[j],
                        "corr":     round(c_val, 3),
                    })

        # 평균 상관 (대각 제외)
        n = len(tickers)
        if n > 1:
            mask   = ~np.eye(n, dtype=bool)
            avg_c  = float(corr_matrix.values[mask].mean())
        else:
            avg_c  = 0.0

        high_pairs.sort(key=lambda x: x["corr"], reverse=True)

        return {
            "matrix":            corr_matrix,
            "high_corr_pairs":   high_pairs[:10],   # 상위 10쌍
            "avg_corr":          round(avg_c, 3),
            "diversification_ok": avg_c < 0.5,
            "n_stocks":          n,
        }

    def sector_concentration(self, holdings: list) -> dict:
        """
        섹터 집중도 분석.

        holdings: [{"ticker": str, "etf": str, "weight": float}, ...]
        반환: HHI, 섹터별 비중, 집중도 경고
        """
        if not holdings:
            return {"hhi": 0, "sector_weights": {}, "concentrated": False, "warning": ""}

        # 섹터별 가중치 합산
        sector_w: dict = {}
        total_w = sum(h.get("weight", 1.0) for h in holdings)
        for h in holdings:
            sec = h.get("etf", "UNKNOWN")
            w   = h.get("weight", 1.0) / max(total_w, 1.0) * 100
            sector_w[sec] = sector_w.get(sec, 0) + w

        # HHI = sum(s_i^2), 완전분산=0, 완전집중=10000
        hhi = sum((w / 100) ** 2 for w in sector_w.values()) * 10000

        # 집중 초과 섹터
        over_concentrated = [s for s, w in sector_w.items()
                              if w > self.max_sector_weight]

        warning = ""
        if hhi > 2500:
            warning = f"⚠️ 고집중 포트폴리오 (HHI={hhi:.0f}) — 분산 필요"
        elif over_concentrated:
            warning = f"⚠️ 섹터 초과: {', '.join(over_concentrated)}"

        return {
            "hhi":              round(hhi, 1),
            "sector_weights":   {k: round(v, 1) for k, v in
                                  sorted(sector_w.items(), key=lambda x: -x[1])},
            "concentrated":     hhi > 1800,
            "over_sectors":     over_concentrated,
            "warning":          warning,
            "effective_n":      round(10000 / max(hhi, 1), 1),
        }

    def portfolio_vol(self, holdings: list, price_data: dict,
                       lookback: int = 60) -> dict:
        """
        포트폴리오 변동성 vs 개별 가중평균 변동성 비교.
        분산 효과 = 1 - (포트폴리오 변동성 / 가중평균 변동성)
        """
        if not holdings:
            return {"port_vol": 0, "weighted_avg_vol": 0, "diversification_benefit": 0}

        tickers = [h["ticker"] for h in holdings]
        weights = np.array([h.get("weight", 1.0) for h in holdings], dtype=float)
        weights /= weights.sum()

        ret_dict = {}
        for tk in tickers:
            try:
                cache_entry = price_data.get(tk)
                df = cache_entry.get("df") if isinstance(cache_entry, dict) else cache_entry
                if df is not None and not df.empty:
                    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
                    if len(c) >= lookback + 5:
                        ret_dict[tk] = c.pct_change().dropna().tail(lookback)
            except Exception as _e:
                logger.debug("항목 스킵: %s", _e)
                continue

        if len(ret_dict) < 2:
            return {"port_vol": 0, "weighted_avg_vol": 0, "diversification_benefit": 0}

        ret_df  = pd.DataFrame(ret_dict).dropna()
        tickers_ok = ret_df.columns.tolist()
        w_ok    = np.array([weights[tickers.index(t)] for t in tickers_ok
                             if t in tickers], dtype=float)
        if w_ok.sum() > 0:
            w_ok /= w_ok.sum()

        # 공분산 행렬
        cov     = ret_df.cov().values * 252   # 연환산
        port_v  = float(np.sqrt(w_ok @ cov @ w_ok)) * 100
        indiv_v = float(np.sqrt(np.diag(cov))) * 100   # 개별 연변동성
        wavg_v  = float(np.dot(w_ok, indiv_v))
        div_ben = max(0, (wavg_v - port_v) / max(wavg_v, 0.01) * 100)

        return {
            "port_vol":              round(port_v, 2),
            "weighted_avg_vol":      round(wavg_v, 2),
            "diversification_benefit": round(div_ben, 1),
            "effective_n":           round(1.0 / max((w_ok ** 2).sum(), 1e-9), 1),
        }

    def get_corr_html(self, corr_result: dict) -> str:
        """상관 행렬 요약 HTML."""
        high_pairs = corr_result.get("high_corr_pairs", [])
        avg_c      = corr_result.get("avg_corr", 0)
        n          = corr_result.get("n_stocks", 0)
        ok         = corr_result.get("diversification_ok", True)

        color  = "var(--green)" if ok else "#ff8844"
        status = "✅ 분산 양호" if ok else "⚠️ 상관 과집중"

        pairs_html = ""
        for p in high_pairs[:6]:
            c_val = p["corr"]
            c_col = "#ff4466" if c_val >= 0.85 else "#ff8844"
            pairs_html += (f'<div class="corr-pair">'
                           f'<span>{p["ticker_a"]} ↔ {p["ticker_b"]}</span>'
                           f'<span style="color:{c_col};font-weight:700">{c_val:.2f}</span>'
                           f'</div>')

        if not pairs_html:
            pairs_html = '<div style="color:var(--text-dim);font-size:.78rem">고상관 쌍 없음 (분산 충분)</div>'

        return f"""
        <div class="corr-summary">
          <div style="display:flex;gap:16px;margin-bottom:10px">
            <div class="corr-stat">
              <span style="font-size:1.1rem;font-weight:700;color:{color}">{status}</span>
            </div>
            <div class="corr-stat">
              <span style="color:#8b949e">평균 상관: </span>
              <span style="color:{color};font-weight:700">{avg_c:.3f}</span>
            </div>
            <div class="corr-stat">
              <span style="color:#8b949e">분석 종목: </span>
              <span style="color:#c9d1d9">{n}개</span>
            </div>
          </div>
          <div style="font-size:.8rem;color:#8b949e;margin-bottom:6px">
            ⚠️ 고상관 종목 쌍 (r≥{self.max_corr_threshold})
          </div>
          {pairs_html}
        </div>"""


# ──────────────────────────────────────────────────────────────
#  5. StressTestEngine — 2008/2020 테일리스크 시나리오
# ──────────────────────────────────────────────────────────────
class StressTestEngine:
    """
    역사적 테일리스크 시나리오 분석.

    시나리오:
      1. 2008 금융위기   : 2008-09-01 ~ 2009-03-06 (-56.8%)
      2. 2020 코로나 폭락: 2020-02-19 ~ 2020-03-23 (-33.9%)
      3. 2022 금리 충격  : 2022-01-03 ~ 2022-10-12 (-25.4%)
      4. 2011 유럽재정위기: 2011-04-29 ~ 2011-10-03 (-19.4%)

    분석 내용:
      - 각 시나리오 기간 동안 현재 보유 종목의 역사적 성과
      - SmartScore 고점수 종목 vs 저점수 종목 비교
      - 최대 낙폭 (MDD) / 회복 기간
      - 섹터별 방어력
    """

    SCENARIOS = {
        "2008_금융위기":    {"start": "2008-09-01", "end": "2009-03-06", "spy_dd": -56.8,
                              "description": "리먼브라더스 파산 → 글로벌 신용위기"},
        "2020_코로나폭락":  {"start": "2020-02-19", "end": "2020-03-23", "spy_dd": -33.9,
                              "description": "코로나19 팬데믹 공포 → 급속 폭락"},
        "2022_금리충격":    {"start": "2022-01-03", "end": "2022-10-12", "spy_dd": -25.4,
                              "description": "Fed 급격한 금리 인상 → 성장주 압박"},
        "2011_유럽재정위기": {"start": "2011-04-29", "end": "2011-10-03", "spy_dd": -19.4,
                              "description": "그리스 디폴트 위기 → 위험자산 매도"},
    }

    # 세션 내 장기 데이터 캐시 (period='max', Close 시리즈)
    _long_cache: dict = {}
    _long_lock  = threading.Lock()

    def _get_long_data(self, ticker: str):
        """
        장기 일봉 Close 시리즈 반환.
        캐시 미스 시 yfinance period='max' (auto_adjust=True) 다운로드.
        세션당 티커 1회만 실행.
        반환: pd.Series(index=DatetimeIndex) 또는 None
        """
        with self._long_lock:
            if ticker in self._long_cache:
                return self._long_cache[ticker]
        try:
            import io as _io, contextlib as _ctx
            with _ctx.redirect_stderr(_io.StringIO()):
                raw = yf.download(ticker, period="max", interval="1d",
                                   auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                with self._long_lock:
                    self._long_cache[ticker] = None
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).strip().title() for c in raw.columns]
            if "Close" not in raw.columns:
                with self._long_lock:
                    self._long_cache[ticker] = None
                return None
            c = pd.to_numeric(raw["Close"], errors="coerce").dropna()
            c.index = pd.to_datetime(c.index)
            result = c if not c.empty else None
            with self._long_lock:
                self._long_cache[ticker] = result
            return result
        except Exception:
            with self._long_lock:
                self._long_cache[ticker] = None
            return None

    def run_scenario(self, ticker: str, scenario_key: str,
                      price_data: dict = None) -> dict:
        """
        단일 종목 × 단일 시나리오 성과 분석.
        데이터 소스: _long_cache (period='max') 우선, 없으면 price_data 폴백.

        Returns
        -------
        dict with: ticker, scenario, start_price, end_price, return_pct,
                   mdd, recovery_days, vs_spy, data_available
        """
        scenario = self.SCENARIOS.get(scenario_key, {})
        if not scenario:
            return {"error": f"시나리오 없음: {scenario_key}"}

        start_dt = pd.Timestamp(scenario["start"])
        end_dt   = pd.Timestamp(scenario["end"])

        # 장기 데이터 우선 (price_data는 1y만 있어 시나리오 기간 포함 안 됨)
        c = self._get_long_data(ticker)
        if c is None:
            # 폴백: price_data (주로 2022 이후 종목 한정으로 부분 커버)
            cache_entry = (price_data or {}).get(ticker)
            if cache_entry is None:
                return {"ticker": ticker, "scenario": scenario_key,
                        "data_available": False, "note": "장기 데이터 없음"}
            try:
                df = cache_entry.get("df") if isinstance(cache_entry, dict) else cache_entry
                if df is None or df.empty:
                    return {"ticker": ticker, "scenario": scenario_key,
                            "data_available": False, "note": "빈 데이터"}
                c = pd.to_numeric(df["Close"], errors="coerce").dropna()
                c.index = pd.to_datetime(c.index)
            except Exception as _e:
                return {"ticker": ticker, "scenario": scenario_key,
                        "data_available": False, "note": str(_e)}

        try:

            # 시나리오 기간 데이터 슬라이싱
            c_period = c[(c.index >= start_dt) & (c.index <= end_dt)]

            if len(c_period) < 5:
                return {"ticker": ticker, "scenario": scenario_key,
                        "data_available": False, "note": f"기간 내 데이터 부족({len(c_period)}일)"}

            start_p = float(c_period.iloc[0])
            end_p   = float(c_period.iloc[-1])
            ret_pct = (end_p / start_p - 1) * 100

            # MDD 계산
            roll_max = c_period.cummax()
            drawdown = (c_period - roll_max) / roll_max * 100
            mdd      = float(drawdown.min())

            # SPY 대비 상대 성과
            vs_spy = ret_pct - scenario["spy_dd"]

            # 회복 기간 추정 (시나리오 종료 후 원래 가격 회복까지)
            c_after = c[c.index > end_dt].head(504)   # 최대 2년
            recovery_days = None
            if len(c_after) > 0:
                recovered = c_after[c_after >= start_p]
                if len(recovered) > 0:
                    recovery_days = int((recovered.index[0] - end_dt).days)

            return {
                "ticker":          ticker,
                "scenario":        scenario_key,
                "description":     scenario["description"],
                "start_price":     round(start_p, 2),
                "end_price":       round(end_p, 2),
                "return_pct":      round(ret_pct, 2),
                "mdd":             round(mdd, 2),
                "vs_spy":          round(vs_spy, 2),
                "spy_dd":          scenario["spy_dd"],
                "recovery_days":   recovery_days,
                "data_available":  True,
                "n_days":          len(c_period),
            }
        except Exception as _e:
            return {"ticker": ticker, "scenario": scenario_key,
                    "data_available": False, "note": str(_e)}

    def run_portfolio_stress(self, tickers: list, price_data: dict = None) -> dict:
        """
        복수 종목 × 전체 시나리오 스트레스 테스트.
        장기 데이터를 병렬로 미리 다운로드 후 분석.

        Returns
        -------
        dict: {scenario_key: {결과 리스트, 요약통계}}
        """
        # ── 장기 데이터 병렬 다운로드 (캐시 미스 종목만) ────────
        uncached = [tk for tk in tickers if tk not in self._long_cache]
        if uncached:
            print(f"  📉 스트레스용 장기 데이터 다운로드 {len(uncached)}개...")
            from concurrent.futures import ThreadPoolExecutor as _SE, as_completed as _SEC
            with _SE(max_workers=8) as _pool:
                _futs = {_pool.submit(self._get_long_data, tk): tk for tk in uncached}
                _ok = sum(1 for f in _SEC(_futs) if f.result() is not None)
            print(f"  ✅ 장기 데이터 {_ok}/{len(uncached)}개 취득")

        results = {}
        for sc_key in self.SCENARIOS:
            sc_results = []
            for tk in tickers:
                r = self.run_scenario(tk, sc_key, price_data)
                if r.get("data_available", False):
                    sc_results.append(r)

            if sc_results:
                rets    = [r["return_pct"] for r in sc_results]
                results[sc_key] = {
                    "ticker_results": sc_results,
                    "summary": {
                        "avg_return":  round(float(np.mean(rets)), 2),
                        "worst":       round(float(np.min(rets)), 2),
                        "best":        round(float(np.max(rets)), 2),
                        "pct_positive": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                        "n_available": len(sc_results),
                        "spy_dd":      self.SCENARIOS[sc_key]["spy_dd"],
                        "description": self.SCENARIOS[sc_key]["description"],
                    }
                }
        return results

    def get_stress_html(self, stress_results: dict) -> str:
        """스트레스 테스트 결과 HTML 섹션 생성."""
        if not stress_results:
            return '<div class="stress-empty">스트레스 테스트 데이터 없음 (10년치 데이터 필요)</div>'

        html = ""
        for sc_key, sc_data in stress_results.items():
            summary = sc_data.get("summary", {})
            sc_info = self.SCENARIOS.get(sc_key, {})
            avg_r   = summary.get("avg_return", 0)
            worst   = summary.get("worst", 0)
            spy_dd  = summary.get("spy_dd", 0)
            pct_pos = summary.get("pct_positive", 0)
            n_avail = summary.get("n_available", 0)
            desc    = summary.get("description", "")

            avg_c  = "var(--green)" if avg_r > spy_dd else "#ff8844" if avg_r > spy_dd * 0.7 else "#ff4466"
            ws_c   = "#ff4466" if worst < -30 else "#ff8844" if worst < -15 else "var(--yellow)"
            pp_c   = "var(--green)" if pct_pos >= 50 else "#ff8844"

            # 상위 3개 / 하위 3개 종목
            ranked = sorted(sc_data.get("ticker_results", []),
                            key=lambda x: x.get("return_pct", -999), reverse=True)
            top3   = ranked[:3]
            bot3   = ranked[-3:]

            def _ticker_span(r):
                rc = r["return_pct"]
                c  = "var(--green)" if rc > 0 else "#ff4466"
                return (f'<span style="color:#c9d1d9">{r["ticker"]}</span>'
                        f'<span style="color:{c};margin-left:4px">{rc:+.1f}%</span>')

            top_str = " · ".join(_ticker_span(r) for r in top3)
            bot_str = " · ".join(_ticker_span(r) for r in bot3)

            html += f"""
            <div class="stress-card">
              <div class="stress-header">
                <span class="stress-name">{sc_key.replace("_", " ")}</span>
                <span class="stress-spy" style="color:#ff4466">SPY {spy_dd:+.1f}%</span>
                <span class="stress-desc">{desc}</span>
              </div>
              <div class="stress-stats">
                <div class="stress-stat">
                  <span class="st-lbl">평균 수익률</span>
                  <span class="st-val" style="color:{avg_c}">{avg_r:+.1f}%</span>
                </div>
                <div class="stress-stat">
                  <span class="st-lbl">최악 종목</span>
                  <span class="st-val" style="color:{ws_c}">{worst:+.1f}%</span>
                </div>
                <div class="stress-stat">
                  <span class="st-lbl">양수 비율</span>
                  <span class="st-val" style="color:{pp_c}">{pct_pos:.0f}%</span>
                </div>
                <div class="stress-stat">
                  <span class="st-lbl">데이터 종목</span>
                  <span class="st-val">{n_avail}개</span>
                </div>
              </div>
              <div class="stress-tickers">
                <div style="font-size:.73rem;color:var(--text-dim);margin-bottom:3px">방어 상위: {top_str}</div>
                <div style="font-size:.73rem;color:var(--text-dim)">낙폭 최대: {bot_str}</div>
              </div>
            </div>"""

        return html


# ── 전역 싱글톤 인스턴스 ─────────────────────────────────────
_audit_logger   = AuditLogger()
_cost_model     = TransactionCostModel(account_type="retail")
_risk_manager   = RiskManager(account_size=100_000,
                               risk_per_trade_pct=1.0,
                               max_position_pct=10.0,
                               atr_multiplier=2.0,
                               hard_stop_pct=8.0)
_portfolio_ctor = PortfolioConstructor(max_sector_weight=25.0,  # 기관 표준 25%
                                        max_corr_threshold=0.70)
_stress_engine  = StressTestEngine()


# ──────────────────────────────────────────────────────────────
#  장 상태 감지
# ──────────────────────────────────────────────────────────────
def market_status() -> Dict:
    et  = datetime.now(ZoneInfo("America/New_York"))
    wd  = et.weekday()
    hm  = et.hour * 60 + et.minute
    if wd >= 5:
        return {"status":"closed","label":"🔴 주말 휴장","et":et,"intraday":False,"regime":"sideways"}
    if 240<=hm<570:
        return {"status":"pre","label":"🟡 프리마켓","et":et,"intraday":True,"regime":"sideways"}
    if 570<=hm<960:
        return {"status":"open","label":"🟢 본장 진행중","et":et,"intraday":True,"regime":"sideways"}
    if 960<=hm<1200:
        return {"status":"after","label":"🟠 애프터마켓","et":et,"intraday":True,"regime":"sideways"}
    return {"status":"closed","label":"🔴 장 마감","et":et,"intraday":False,"regime":"sideways"}

# ──────────────────────────────────────────────────────────────
#  설정
# ──────────────────────────────────────────────────────────────
OUTPUT_FILE    = "market_dashboard.html"
REFRESH_SEC    = 300   # 5분

# 기존 SPDR 11개 + 테마 ETF 9개
SECTORS = {
    # ── SPDR 기본 11개 ──────────────────────────────────────
    "XLK":  ("기술",         "💻", "SPDR"),
    "XLY":  ("임의소비재",   "🛍️", "SPDR"),
    "XLF":  ("금융",         "🏦", "SPDR"),
    "XLV":  ("헬스케어",     "🏥", "SPDR"),
    "XLE":  ("에너지",       "⛽", "SPDR"),
    "XLI":  ("산업",         "🏭", "SPDR"),
    "XLB":  ("소재",         "🪨", "SPDR"),
    "XLRE": ("부동산",       "🏠", "SPDR"),
    "XLP":  ("필수소비재",   "🛒", "SPDR"),
    "XLU":  ("유틸리티",     "⚡", "SPDR"),
    "XLC":  ("통신",         "📡", "SPDR"),
    # ── 테마 ETF 9개 ────────────────────────────────────────
    "ITA":  ("방산",         "🛡️", "Theme"),
    "XAR":  ("항공우주",     "🚀", "Theme"),
    "SOXX": ("반도체",       "🔬", "Theme"),
    "CIBR": ("사이버보안",   "🔐", "Theme"),
    "SKYY": ("클라우드",     "☁️", "Theme"),
    "XBI":  ("바이오",       "🧬", "Theme"),
    "LIT":  ("리튬/배터리",  "🔋", "Theme"),
    "FINX": ("핀테크",       "💳", "Theme"),
    "ICLN": ("청정에너지",   "🌱", "Theme"),
}

MACRO_TICKERS = {
    "^VIX":     "VIX 공포지수",
    "HYG":      "하이일드 채권",
    "TLT":      "장기국채",
    "GLD":      "금",
    "DX-Y.NYB": "달러인덱스",
    "^TNX":     "10년물 금리",
}

WATCHLIST = [
    # ══ 484종목 유니버스 (계층1:200 계층2:200 계층3:84) ══
    # backtest v8와 동일 유니버스 — IC 검정력 3배 향상
    # 계층1 대형 (시총>10B)
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
    # 계층2 중형 (시총 2~10B) — Wyckoff 신호 최강
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
    # 계층3 소형/고베타
    "MARA","RIOT","HUT","BTBT","CLSK","CIFR","WULF","IREN","CORZ","BITF",
    "IONQ","RGTI","QBTS","QUBT","ARQQ","RKLB","SPIR","MNTS","ASTR","LLAP",
    "SPCE","VORB","RKT","ACHR","JOBY","SOUN","BBAI","CRDO","MAPS","MDGL",
    "HRMY","KROS","TARS","VKTX","GPCR","NRIX","RLMD","MNKD","AGEN","AGIO",
    "TWST","CDNA","TNGX","NVTA","WKHS","NKLA","GOEV","FSR","ZEV","FFIE",
    "PTRA","HYZN","HYLN","DRVN","PLUG","BLDP","FCEL","STEM","FREYR","NRGV",
    "MP","LAC","LI","SLI","LTHM","PLL","TELL","LNG","NFE","FLNG",
    "GLNG","LILM","ARCHER","RIDE","LIDR","OUST","LAZR","VLDR","INVZ","ARROW",
    "OPAL","ADEX","EVTOL","BTDR",
]

# ── 섹터별 상세 종목 (대형 + 중견 + 중소) ─────────────────────
SECTOR_STOCKS: Dict[str, List[str]] = {
    # 기술 (XLK)
    "XLK": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","CSCO","IBM","TXN",
        "QCOM","AMAT","LRCX","KLAC","MU","ADI","MRVL","INTC","ON","STX",
        "PSTG","NTAP","HPE","JNPR","VIAV","FORM","COHU","ACLS","ONTO","CAMT",
        "SITM","ALGM","SLAB","DIOD","AMBA","DUOL","ZM","U","SERV","ARAI","SOUN",
    ],
    # 임의소비재 (XLY)
    "XLY": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
        "EBAY","ETSY","RH","WSM","RVLV","BOOT","XPOF","VSCO",
        "FIVE","OLLI","BIG","CONN","CAKE","DRI","TXRH","BJRI","FWRG",
    ],
    # 금융 (XLF)
    "XLF": [
        "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","CB","PGR",
        "MET","PRU","AFL","CINF","GL","SFG","WRB","HIG","FNF",
        "NYCB","PACW","WAL","BKU","FFIN","TOWN","HOMB","FULT","WSFS",  # CADE 상장폐지(2024) 제거
        "LPLA","RJF","PIPR","SF",
    ],
    # 헬스케어 (XLV)
    "XLV": [
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "GILD","REGN","VRTX","BIIB","ILMN","IDXX","IQV","CRL","MEDP","HALO",
        "ACAD","ARWR","BEAM","NTLA","EDIT","CRSP","FATE","KYMR","TMDX","RXRX",
        "SANA","VERV","IMVT","RVMD",
    ],
    # 에너지 (XLE)
    "XLE": [
        "XOM","CVX","COP","EOG","SLB","PXD","MPC","PSX","VLO","OXY",
        "DVN","FANG","HAL","BKR","NOV","RIG","VAL","BORR","PTEN","NE",
        "SM","CPE","REI","SWN","RRC","CNX","AR","MTDR","ROCC",  # CIVI 상장폐지(2024) 제거
        "TALO","NEXT","VTLE","NNE","USEG",
    ],
    # 산업 (XLI)
    "XLI": [
        "CAT","DE","HON","UPS","RTX","BA","GE","MMM","EMR","ETN",
        "PH","ROK","XYL","GNRC","AIRC","SHYF","ARCB","SAIA","ODFL",  # REVG 상장폐지(2024) 제거
        "JBHT","WERN","CHRW","GXO","RXO","SNDR","TFII","HTLD","MRTN","USX",
        "MATX","KEX","SALT","BWXT","DRS",
    ],
    # 소재 (XLB)
    "XLB": [
        "LIN","APD","SHW","FCX","NEM","NUE","STLD","RS","VMC","MLM",
        "CF","MOS","IPI","RYAM","SLVM","TROX","KRO","HBM","TECK","AA",
        "CENX","CRS","HAYN","KALU","MTAL","MP","UUUU","NXE","DNN","URG",
    ],
    # 부동산 (XLRE)
    "XLRE": [
        "AMT","PLD","CCI","EQIX","PSA","SPG","O","WELL","AVB","EQR",
        "MAA","UDR","NNN","STOR","EPRT","NTST","PINE","LAND","AFCG","STWD",
        "BRSP","SACH","GPMT","TRTX","FBRT","RC","BXMT","KREF","ARI","MFA",
    ],
    # 필수소비재 (XLP)
    "XLP": [
        "PG","KO","PEP","COST","WMT","PM","MO","KMB","CL","GIS",
        "SJM","CAG","MKC","HRL","LW","SMPL","FRPT","UTZ","NWFL","PFGC",
        "USFD","SYY","CHEF","COKE","FIZZ","CELH","MNST","KDP",
    ],
    # 유틸리티 (XLU)
    "XLU": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","XEL","WEC",
        "ES","CMS","NI","EVRG","ATO","OGE","NWE","RUN","NOVA",
        "ARRY","ENPH","SEDG","SHLS","FTCI","CWEN","BEP","FLNC","STEM",
    ],
    # 통신 (XLC)
    "XLC": [
        "META","GOOGL","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","FOX",
        "PARA","WBD","LYV","SIRI","LUMN","IRDM","GSAT","DISH","AMCX",
        "MSGM","FUBO","DOYU","HUYA",
    ],
    # 방산 (ITA)
    "ITA": [
        "LMT","RTX","NOC","GD","BA","HII","TXT","HEI","LDOS","SAIC",
        "BAH","CACI","MANT","PSN","DRS","BWXT","KTOS","AVAV","RCAT","JOBY",
        "ACHR","BLDE","AXON","ATRO","CPI",
    ],
    # 항공우주 (XAR)
    "XAR": [
        "RTX","BA","LMT","GD","NOC","HEI","TXT","AXON","KTOS","AVAV",
        "RCAT","JOBY","ACHR","ASTR","AIR","ESLT",
        "ATRO","MOOG","CPI","DRS","VSE","LDOS","SAIC","BAH","SATL","SIDU","FLY",
    ],
    # 반도체 (SOXX)
    "SOXX": [
        "NVDA","AMD","AVGO","QCOM","MU","AMAT","LRCX","KLAC","ADI","MRVL",
        "INTC","ON","TXN","MPWR","WOLF","SITM","ALGM","SLAB","DIOD","AMBA",
        "ACLS","ONTO","CAMT","FORM","COHU","POWI","IOSP","SWKS","QRVO","MTSI",
        "IXYS","AEHR","OLED","AXTI","PDFS","ALAB",
    ],
    # 사이버보안 (CIBR)
    "CIBR": [
        "PANW","CRWD","ZS","FTNT","CYBR","S","OKTA","TENB","QLYS","VRNT",
        "RPD","SAIL","RDWR","ACLS","CWAN","ASAN","BB","SCWX",
        "MIME","EVTC","IMMR","XPEL",
    ],
    # 클라우드 (SKYY)
    "SKYY": [
        "AMZN","MSFT","GOOGL","CRM","NOW","WDAY","TEAM","ZM","DOCU","MDB",
        "SNOW","DDOG","NET","HUBS","BILL","CFLT","GTLB","ESTC","APPN","PEGA",
        "NCNO","ALTR","ASAN","SMAR","FROG","BRZE","SPRK","CODA","PAYO","TASK",
    ],
    # 바이오 (XBI)
    "XBI": [
        "MRNA","BNTX","REGN","VRTX","BIIB","GILD","AMGN","ILMN","BEAM","NTLA",
        "EDIT","CRSP","FATE","KYMR","TMDX","RXRX","SANA","VERV","PRME","IMVT",
        "RVMD","ACAD","ARWR","HALO","MEDP","INSM","NTRA","EXAS","PCVX","RCKT",
        "LEGN","ALDX","YMAB","PRAX","APLS",
    ],
    # 리튬/배터리 (LIT)
    "LIT": [
        "ALB","SQM","LTHM","LAC","PLL","SGML","ATLX","LI","NIO",
        "XPEV","RIVN","LCID","CHPT","BLNK","EVGO","FREYR","MVST",
        "NRGV","FLUX","ACMR","MP","CQQQ",
    ],
    # 핀테크 (FINX)
    "FINX": [
        "V","MA","PYPL","SQ","AFRM","UPST","LC","OPEN","SOFI","NRDS",
        "HOOD","COIN","MSTR","RIOT","MARA","HUT","BTBT","CIFR","BITF","WULF",
        "IREN","CLSK","CORZ","SDIG",
    ],
    # 청정에너지 (ICLN)
    "ICLN": [
        "NEE","ENPH","SEDG","FSLR","PLUG","BE","CWEN","BEP","ARRY","SHLS",
        "FTCI","NOVA","RUN","SPWR","MAXN","CSIQ","JKS","DQ","SOL","HASI",
        "GPRE","REX","ALTO","AMRC","CLNE","GEVO",
    ],
}

# ──────────────────────────────────────────────────────────────
#  스마트머니 점수 엔진 — 7요소 100점 만점
# ──────────────────────────────────────────────────────────────

def _cs_grade(score: float):
    """크로스섹셔널 점수(0~100) → (grade, badge)"""
    if   score >= 80: return "상위 20%", "🔥"
    elif score >= 60: return "상위 40%", "✅"
    elif score >= 40: return "중립",     "⚖️"
    elif score >= 20: return "하위 40%", "⚠️"
    else:             return "하위 20%", "🔴"


class SmartScoreEngine:
    """
    8개 요소로 스마트머니 유입 강도를 0~115점 측정 (옵션 없으면 100점 만점).

    [3] 매집 국면 탐지      25점  ← 최우선: 가장 이른 신호
    [7] OBV 다이버전스      20점  ← 상승 다이버전스(가격↓+OBV↑)에 최고 가중
                                     1.0σ 이상만 유의미로 인정, 연속 함수로 절벽 제거
    [2] 거래량 구조         18점  ← 상승일/하락일 거래량 비율
    [8] 옵션 신호           15점  ← 기관 직접 베팅 증거 (데이터 있을 때만)
    [4] 모멘텀 가속         12점  ← 진입 타이밍 확인용
    [1] 상대 강도 RS        10점  ← 후행 정보, 보조 역할로 축소
    [5] BB 압축 돌파         8점  ← [3]과 보완 관계
    [6] 추세 일관성          7점  ← 순수 보조지표

    OBV 시나리오 우선순위:
      🔥 상승다이버전스  = 가격≤0% + OBV≥1.0σ  → 최대 13점 (폭락 중 매집 = 최고 신호)
      ⚡ 횡보매집OBV    = 가격±1% + OBV≥1.0σ  → 9점    (가격 안 올랐는데 OBV 쌓임)
      ✅ OBV동반상승    = 가격≥1% + OBV≥1.5σ  → 5~7점  (이미 알려진 정보)
      ⚠️ 분산매도경고   = 가격↑  + OBV<-0.5σ → 0점    (고점 매도 신호)

    75~100 : 🔥 강한 매집    55~75 : ✅ 관심
    40~55  : ⚖️ 중립        25~40 : ⚠️ 약세
    0~25   : 🔴 강한 매도압력
    """

    def calc(self, c: pd.Series, v: pd.Series,
             h: pd.Series, lo: pd.Series,
             sector_rets: Dict[str, float],
             opt_data: Dict = None,
             regime: str = "sideways") -> Dict:
        """
        c        : Close 시리즈
        v        : Volume 시리즈
        h        : High 시리즈
        lo       : Low 시리즈
        sector_rets : {"spy": 5일수익률, "sector": 5일수익률}
        opt_data    : {"pc_ratio": float, "iv_pct": float} — 없으면 [8] 생략
        """
        score   = 0.0
        details = {}
        _pct    = c.pct_change()  # 캐시 — [1]~[7] 공통 사용 (중복 계산 방지)
        ret_1d  = float(_pct.iloc[-1]) * 100

        # ══════════════════════════════════════════════════════
        #  배점 원칙: 이른 신호 + 신뢰도 높은 신호에 집중
        #  [3] 매집탐지  25  ← 가장 이른 신호, 핵심
        #  [7] OBV       20  ← 자금흐름 선행, 핵심
        #  [2] 거래량구조 18  ← 중요하나 [3][7]과 일부 겹침
        #  [8] 옵션신호  15  ← 기관 직접 베팅, 있을 때 최고 신뢰
        #  [4] 모멘텀가속 12  ← 후행이지만 진입 타이밍 확인용
        #  [1] RS         10  ← 이미 오른 정보, 과대평가 방지
        #  [5] BB압축      8  ← [3]과 겹침, 보조 역할
        #  [6] 추세일관성   7  ← 순수 보조지표
        #  합계 기본 100 (옵션 데이터 있으면 115)
        # ══════════════════════════════════════════════════════

        # ── [1] 상대 강도 RS (10점) ──────────────────────────
        # 배점 축소 이유: RS는 이미 상승이 반영된 후행 정보.
        # 매집 탐지가 목적이라면 "이미 오른 종목"을 우대하면 안 됨.
        ret_5d  = float(c.pct_change(5).iloc[-1])  * 100
        ret_20d = float(c.pct_change(20).iloc[-1]) * 100
        spy_r5  = sector_rets.get("spy", 0.0)
        sec_r5  = sector_rets.get("sector", spy_r5)

        rs_vs_spy    = ret_5d - spy_r5
        rs_vs_sector = ret_5d - sec_r5

        # ── [1-A] SPY 대비 RS (6점) — Jegadeesh & Titman(1993) 기준 ──
        # 원칙: RS는 연속 함수 — 음수도 정보 (0 이하라고 0점 처리 시 정보 손실)
        # 근거: J&T(1993) 모멘텀 팩터는 long-short 구조.
        #       음수 RS = 약세 정보 → 점수로 반영해야 discriminating power 유지.
        # 스케일: rs = +6% → 6점 만점, rs = 0% → 3점(중립), rs = -6% → 0점
        # 단일 윈도우(5일): 이 시스템은 단기 매집 탐지 목적이므로 5일 유지
        #   (Levy 1967의 26주는 중장기 모멘텀 — 목적이 다름)
        rs_spy_score = max(0.0, min(6.0, (rs_vs_spy / 6.0 + 0.5) * 6.0))

        # ── [1-B] 섹터 대비 RS (4점) ─────────────────────────────────
        # 섹터 대비 초과분: 섹터 내 상대 강도 측정 (SPY 대비와 다른 차원)
        # 동일한 연속 함수 적용: rs = +4% → 4점, 0% → 2점, -4% → 0점
        rs_sec_score = max(0.0, min(4.0, (rs_vs_sector / 4.0 + 0.5) * 4.0))

        s1 = rs_spy_score + rs_sec_score
        score += s1
        details["RS"] = {"score": round(s1, 1), "max": 10,
                         "vs_spy": round(rs_vs_spy, 2), "vs_sector": round(rs_vs_sector, 2)}

        # ── [2] 거래량 구조 (18점) ───────────────────────────
        # 개선: 전체 기간 UD ratio는 최근 매집 신호를 희석시킴
        # → 20일(단기)과 60일(중기) UD ratio 분리 + 트렌드 가속도 측정
        rets = c.pct_change().dropna()
        # vol_surge: 오늘 거래량 제외 (장중 미완성 문제)
        # avg_vol_5: 전일 포함 최근 5일, avg_vol_20: 전일 기준 20일 베이스
        v_ex_today  = v.iloc[:-1]   # 오늘 제외
        avg_vol_20  = float(v_ex_today.rolling(20).mean().iloc[-1]) if len(v_ex_today) > 20 else float(v_ex_today.mean())
        avg_vol_5   = float(v_ex_today.rolling(5).mean().iloc[-1])  if len(v_ex_today) > 5  else avg_vol_20
        vol_surge   = avg_vol_5 / max(avg_vol_20, 1.0)

        # UD ratio — NaN 방어 + 보합일 처리
        # reindex 후 NaN이 생길 수 있으므로 fillna(0)로 방어
        v_aligned = v.reindex(rets.index).fillna(0)

        # 최근 20일 UD ratio — Granville(1963) 원저 방식
        # 보합일(rets == 0) 제외: 방향성 정보 없는 거래량은 계산에서 배제
        # 근거: Granville OBV 원저 — 보합일은 미분류(neither up nor down)
        rets_20 = rets.iloc[-20:]
        v_20    = v_aligned.iloc[-20:]
        up_vol_20   = float(v_20[rets_20 > 0].mean()) if (rets_20 > 0).sum() > 0 else 1.0
        down_vol_20 = float(v_20[rets_20 < 0].mean()) if (rets_20 < 0).sum() > 0 else 1.0
        # NaN 방어: mean()이 NaN이면 fallback
        if not (up_vol_20   > 0): up_vol_20   = 1.0
        if not (down_vol_20 > 0): down_vol_20 = 1.0
        ud_ratio_20 = up_vol_20 / max(down_vol_20, 1.0)
        if not (0.01 < ud_ratio_20 < 100):
            ud_ratio_20 = 1.0

        # 최근 60일 UD ratio — 동일 방식
        rets_60 = rets.iloc[-60:] if len(rets) >= 60 else rets
        v_60    = v_aligned.iloc[-60:] if len(v_aligned) >= 60 else v_aligned
        up_vol_60   = float(v_60[rets_60 > 0].mean()) if (rets_60 > 0).sum() > 0 else 1.0
        down_vol_60 = float(v_60[rets_60 < 0].mean()) if (rets_60 < 0).sum() > 0 else 1.0
        if not (up_vol_60   > 0): up_vol_60   = 1.0
        if not (down_vol_60 > 0): down_vol_60 = 1.0
        ud_ratio_60 = up_vol_60 / max(down_vol_60, 1.0)
        if not (0.01 < ud_ratio_60 < 100):
            ud_ratio_60 = 1.0

        ud_ratio = ud_ratio_20
        ud_accel = ud_ratio_20 / max(ud_ratio_60, 0.5)

        # UD 비율 점수 (9점)
        ud_score = max(0.0, min(9.0, (ud_ratio - 0.8) / 0.7 * 9.0))
        # 매집 가속 보너스 (2점)
        ud_accel_bonus = min(2.0, max(0.0, (ud_accel - 1.1) / 0.4 * 2.0))
        # 거래량 급증 (7점)
        # [3] 매집탐지와 이중 패널티 방지: surge > 2.5 시 [2]에서는 패널티 없이 0점 유지
        # ([3] pa_vol_bonus에서 이미 처리)
        if vol_surge <= 2.5:
            surge_score = max(0.0, min(7.0, (vol_surge - 0.8) / 0.7 * 7.0))
        else:
            surge_score = 7.0 * max(0.0, 1.0 - (vol_surge - 2.5) * 0.3)  # 완만한 감쇠
        s2 = min(18.0, ud_score + ud_accel_bonus + surge_score)
        score += s2
        details["거래량구조"] = {"score": round(s2, 1), "max": 18,
                                 "ud_ratio":    round(ud_ratio, 2),
                                 "ud_ratio_60": round(ud_ratio_60, 2),
                                 "ud_accel":    round(ud_accel, 2),
                                 "vol_surge":   round(vol_surge, 2)}

        # ── [3] 매집 국면 탐지 (25점) ────────────────────────
        # 배점 최대 이유: 가장 이른 신호. 여기서 걸러야 나머지가 의미 있음.
        #
        # 개선 내용:
        #  A) ATR 기반 변동성 압축 — max-min 방식의 gap 이상치 민감도 해소
        #  B) 바닥신호: 60일 기준 + Selling Climax(SC) / Secondary Test(ST) 분리
        #  C) OBV 선행: 5일 변화량 → 20일 slope 다이버전스 (진짜 선행)

        # ── 패턴 A: ATR 기반 횡보 압축 (최대 12점) ──────────
        # ATR은 gap에 강건 — Wyckoff TR(Trading Range) 식별에 적합
        _tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14  = float(_tr.rolling(14).mean().iloc[-1])  if len(_tr) >= 14 else float(_tr.mean())
        atr_63  = float(_tr.rolling(63).mean().iloc[-1])  if len(_tr) >= 63 else atr_14 * 1.3
        atr_compression = atr_14 / max(atr_63, 1e-8)  # < 0.8이면 압축

        # 압축 강도에 비례한 연속 점수 (절벽 없음)
        if atr_compression < 0.65:
            pa_base = min(9.0, (0.65 - atr_compression) / 0.25 * 9.0 + 3.0)
        elif atr_compression < 0.80:
            pa_base = max(0.0, (0.80 - atr_compression) / 0.15 * 4.0)
        else:
            pa_base = 0.0

        # 거래량 보너스: 횡보 중 거래량 증가 = 세력 유입 (최대 3점)
        # 단, 극단적 급증(>2.5)은 덤핑 의심으로 보너스 없음
        if 1.3 <= vol_surge <= 2.5:
            pa_vol_bonus = min(3.0, (vol_surge - 1.3) / 0.8 * 3.0)
        elif vol_surge > 2.5:
            pa_vol_bonus = max(0.0, 1.5 - (vol_surge - 2.5) * 1.5)
        else:
            pa_vol_bonus = 0.0
        pa_score = min(12.0, pa_base + pa_vol_bonus)

        # ── 패턴 B: 바닥신호 (최대 8점) — Wyckoff SC/ST ────
        # 개선: 20일 → 60일 기준으로 확장 (진짜 바닥은 60일 저점 기준)
        vol_drying = avg_vol_5 / max(avg_vol_20, 1.0)
        avg_vol_60_acc = float(v.rolling(60).mean().iloc[-1]) if len(v) >= 60 else avg_vol_20

        # 60일 저점 근처 위치 (< 0.25 = 하위 25%)
        low_60  = float(lo.iloc[-60:].min())  if len(lo) >= 60 else float(lo.min())
        high_60 = float(h.iloc[-60:].max())   if len(h)  >= 60 else float(h.max())
        near_low_60 = (float(c.iloc[-1]) - low_60) / max(high_60 - low_60, 1e-8)

        # Selling Climax 감지: 최근 15일 내 거래량 스파이크(≥2.5×60일평균) + 큰 하락 후 반등
        vol_spike_15 = float(v.iloc[-15:].max()) / max(avg_vol_60_acc, 1.0)
        ret_15d = float(c.pct_change(15).iloc[-1]) * 100 if len(c) > 15 else 0.0
        sc_signal = vol_spike_15 >= 2.5 and ret_15d < -7.0 and ret_5d > 2.0  # +2% 이상 반등이어야 노이즈 배제

        # ST (Secondary Test): 거래량 감소 + 60일 저점 근처
        st_signal = vol_drying < 0.65 and near_low_60 < 0.25

        if sc_signal and st_signal:        pb_score = 8.0  # SC + ST 동시 = 최강 바닥
        elif sc_signal:                    pb_score = 6.0  # SC만
        elif st_signal:                    pb_score = 5.0  # ST만
        elif vol_drying < 0.70 and near_low_60 < 0.35: pb_score = 3.0
        elif vol_drying < 0.75:            pb_score = 1.0
        else:                              pb_score = 0.0

        # ── 패턴 C: OBV 선행 다이버전스 (최대 5점) ──────────────
        # [v2 재설계] 세력 설거지 + 개미 매수 오인 방지
        #
        # 기존 문제:
        #   slope만 보면 세력이 대량 매도 후 빠져나간 뒤
        #   거래량이 줄어든 상태(delta_std 감소)에서 개미 소량 매수도
        #   1σ 이상으로 계산되어 OBV선행으로 오인됨
        #
        # 3가지 필터 추가:
        #   [F1] OBV 회복률: 직전 하락폭 대비 최근 5일 회복폭
        #        진짜 세력 재진입 >= 25% 회복 / 개미 매수 < 10%
        #   [F2] OBV 저점 경과일: 최근 20일 저점이 몇 일 전인지
        #        진짜 반전은 저점 다진 후 상승 / 설거지 직후는 저점 경과 짧음
        #   [F3] 거래량 품질: 상승 거래량 / 하락 거래량 비율
        #        세력 재진입은 매수 거래량 >= 매도 거래량 수준
        obv_series = (v * np.sign(_pct.fillna(0))).cumsum()
        pc_score   = 0.0
        _obv_quality_flags = {"recovery": 0.0, "bottom_days": 0, "vol_quality": 0.0}

        if len(obv_series) >= 22:
            _x    = np.arange(20, dtype=float)
            _c20  = c.iloc[-20:].values.astype(float)
            _o20  = obv_series.iloc[-20:].values.astype(float)
            _c_slope = np.polyfit(_x, _c20 / max(_c20[0], 1e-8), 1)[0]
            _o_slope = np.polyfit(_x, _o20 / max(abs(_o20[0]), 1.0), 1)[0]

            # ── [F1] OBV 회복률 계산 ─────────────────────────────
            # 최근 20일 OBV 최저점 찾기
            _o20_min_idx = int(np.argmin(_o20))
            _o20_min     = _o20[_o20_min_idx]
            _o20_max_bef = float(np.max(_o20[:_o20_min_idx+1])) if _o20_min_idx > 0 else _o20[0]
            _obv_drop    = _o20_max_bef - _o20_min          # 하락폭 (양수면 진짜 하락)
            _obv_recover = _o20[-1] - _o20_min               # 저점 이후 회복폭

            if _obv_drop > 0:
                obv_recovery = _obv_recover / max(abs(_obv_drop), 1.0)
            else:
                obv_recovery = 1.0  # 하락 없었으면 회복률 문제없음

            # ── [F2] OBV 저점 경과일 ─────────────────────────────
            # 최저점이 현재로부터 몇 일 전인지 (최근일수록 짧음)
            obv_bottom_days = 19 - _o20_min_idx  # 0 = 오늘이 저점, 19 = 20일 전이 저점

            # ── [F3] 거래량 품질 ──────────────────────────────────
            # 최저점 이후 상승 구간 거래량 vs 직전 하락 구간 거래량 비교
            # 하락 구간: 저점 이전, 상승 구간: 저점 이후
            _v20 = v.iloc[-20:].values.astype(float)
            if _o20_min_idx > 0 and _o20_min_idx < 19:
                _vol_down = float(np.nanmean(_v20[:_o20_min_idx]))    # 하락 구간 평균 거래량
                _vol_up   = float(np.nanmean(_v20[_o20_min_idx+1:]))  # 상승 구간 평균 거래량
                vol_quality = _vol_up / max(_vol_down, 1.0)
            elif _o20_min_idx == 0:
                vol_quality = 1.0   # 20일 전부터 상승 중 — 문제없음
            else:
                vol_quality = 0.0   # 아직 바닥도 못 찍음

            _obv_quality_flags = {
                "recovery":    round(obv_recovery, 3),
                "bottom_days": obv_bottom_days,
                "vol_quality": round(vol_quality, 3),
            }

            # ── slope 기반 기본 점수 (기존 로직 유지) ────────────
            if _c_slope < -0.0005 and _o_slope > 0.001:
                _pc_raw = min(5.0, _o_slope * 300 + 3.0)     # 강한 Bullish Divergence
            elif abs(_c_slope) < 0.0005 and _o_slope > 0.001:
                _pc_raw = min(4.0, _o_slope * 200 + 2.0)     # 횡보 다이버전스
            elif _c_slope > 0.001 and _o_slope < -0.001:
                _pc_raw = 0.0                                  # 역다이버전스 (분산 매도)
            else:
                _pc_raw = max(0.0, min(2.0, _o_slope * 100 + 1.0))

            # ── 3개 필터 게이팅 ───────────────────────────────────
            # 조건 1: OBV 회복률 < 0.15 → 설거지 후 극소량 반등
            #         slope가 양수여도 OBV가 아직 바닥 수준
            if obv_recovery < 0.15:
                _pc_raw *= 0.2   # 80% 감점 — 신호 신뢰도 없음

            # 조건 2: 저점 경과일 < 3일 → 설거지 직후 아직 방향 불명확
            #         OBV slope가 반전됐다고 볼 수 없는 초기 단계
            elif obv_bottom_days < 3:
                _pc_raw *= 0.4   # 60% 감점

            # 조건 3: 거래량 품질 < 0.50 → 매도 거래량이 매수의 2배 이상
            #         세력 이탈 후 개미 소량 매수 패턴
            if vol_quality < 0.50 and obv_recovery < 0.30:
                _pc_raw *= 0.5   # 추가 50% 감점 (누적)

            pc_score = _pc_raw

        s3 = min(25.0, pa_score + pb_score + pc_score)
        score += s3

        accum_tags = []
        if pa_score >= 8:  accum_tags.append("횡보매집")
        if pb_score >= 5:  accum_tags.append("바닥신호")
        if pb_score >= 6 and sc_signal: accum_tags.append("🔥SC바닥확인")
        # OBV선행 태그: slope 점수 + 3필터 모두 통과해야 인정
        # 필터 통과 기준:
        #   recovery >= 0.25 : 직전 하락의 25% 이상 회복 (개미 매수 배제)
        #   bottom_days >= 3 : 저점 찍은 지 3일 이상 (설거지 직후 배제)
        #   vol_quality >= 0.50 OR recovery >= 0.40 : 거래량 품질 확인
        _f = _obv_quality_flags
        _obv_tag_ok = (
            pc_score >= 3
            and _f["recovery"]    >= 0.25
            and _f["bottom_days"] >= 3
            and (_f["vol_quality"] >= 0.50 or _f["recovery"] >= 0.40)
        )
        if _obv_tag_ok:
            accum_tags.append("OBV선행")

        details["매집탐지"] = {
            "score": round(s3, 1), "max": 25,
            "pa": round(pa_score, 1), "pb": round(pb_score, 1), "pc": round(pc_score, 1),
            "compression":   round(atr_compression, 2),
            "near_low_60":   round(near_low_60, 2),
            "vol_dry":       round(vol_drying, 2),
            "sc_signal":     sc_signal,
            "tags":          accum_tags,
            # OBV선행 필터 상태 (디버깅·UI용)
            "obv_recovery":    round(_obv_quality_flags.get("recovery",    0.0), 3),
            "obv_bottom_days": _obv_quality_flags.get("bottom_days", 0),
            "obv_vol_quality": round(_obv_quality_flags.get("vol_quality", 0.0), 3),
        }

        # ── [4] 모멘텀 가속 (12점) ───────────────────────────
        # 진입 타이밍 확인용. 중요하지만 후행이라 배점 축소.
        ret_3d  = float(c.pct_change(3).iloc[-1])  * 100

        # accel_ratio: 분모 클리핑을 0.5→1.5로 상향
        # 기존 0.5에서는 ret_20d=0.3%처럼 보합 시 3일 소폭 상승만으로 만점 가능
        # 1.5로 올리면 20일이 보합이어도 과장 없이 합리적 범위로 수렴
        accel_ratio = ret_3d / max(abs(ret_20d), 1.5)
        accel_score = max(0.0, min(8.0, (accel_ratio + 1) * 4))

        # direction_bonus: [1] RS와 중복 방지
        # ret_5d > 0 조건 제거 — ret_5d는 이미 [1]에서 측정
        # 단기 모멘텀 연속성(1d+3d)만 확인하되 1점으로 축소
        direction_bonus = 1.0 if (ret_1d > 0 and ret_3d > 0 and ret_20d > 0) else \
                          0.5 if (ret_1d > 0 and ret_3d > 0) else 0.0

        # reversal_bonus: 거래량 동반 반전 신호
        # 근거: Karpoff(1987) "Relation Between Price Changes and Trading Volume"
        #       진짜 반전에는 거래량이 20일 평균의 1.5x 이상 필요 (보수적 중간값)
        #       1.5x: 통계적으로 유의미한 거래량 이탈 최소 기준
        #       회복률 조건: ret_5d / abs(ret_20d) > 0.50 — 하락분의 50% 이상 회복
        #       이중 확인(price + volume) = Blau et al.(2009) 반전 신호 표준
        _rev_vol_ok   = vol_surge > 1.5                              # Karpoff 기준
        _rev_ret_ok   = ret_20d < -5.0 and ret_5d > 3.0
        _rev_ratio_ok = ret_5d / max(abs(ret_20d), 0.1) > 0.50      # 하락분 50% 회복
        reversal_bonus = 2.0 if (_rev_vol_ok and _rev_ret_ok and _rev_ratio_ok) else \
                         1.0 if (_rev_ret_ok and _rev_ratio_ok) else 0.0

        s4 = min(12.0, accel_score + direction_bonus + reversal_bonus)
        score += s4
        details["모멘텀가속"] = {"score": round(s4, 1), "max": 12,
                                 "accel": round(accel_ratio, 2),
                                 "reversal": reversal_bonus > 0}

        # ── [5] BB 압축 돌파 (8점) ───────────────────────────
        # [3] 매집탐지와 일부 겹치므로 보조 역할로 축소.
        std20  = float(c.rolling(20).std().iloc[-1])
        std60  = float(c.rolling(60).std().iloc[-1]) if len(c) >= 60 else std20
        bb_compression = std20 / max(std60, 1e-8)

        # ── BB압축 점수 재설계 ─────────────────────────────────
        # 기존 문제:
        #   A) 팽창(>1.0)인데도 1~2점 제공 → 역방향 보상
        #   B) 이산 점프 → 경계값 ±0.002에서 2점 차이 절벽
        #   C) 방향 확인(상승돌파)과 그냥 압축의 점수 차이 너무 작음
        #
        # 개선:
        #   - 압축 강도를 연속 함수로 (0~5점)
        #   - 팽창(>1.0)이면 무조건 0점
        #   - 상승돌파 확인 시 추가 3점 (총 8점)
        if bb_compression <= 1.0:
            # 압축 점수: 1.0→0점, 0.5→5점 (선형)
            compress_score = min(5.0, max(0.0, (1.0 - bb_compression) / 0.5 * 5.0))
        else:
            compress_score = 0.0  # 팽창 중 = 0점 (기존 1~2점 오류 수정)

        # 방향 확인: 상승돌파(ret_5d > 2%) = 압축 후 진짜 폭발
        # 상승폭이 클수록 추가 점수 (최대 3점)
        if bb_compression < 0.85 and ret_5d > 1.0:
            direction_score = min(3.0, (ret_5d - 1.0) / 3.0 * 3.0)
        elif bb_compression < 0.85 and ret_5d < -2.0:
            # 압축 후 하락돌파 = 역방향 신호 → 압축 점수 반감
            compress_score *= 0.5
            direction_score = 0.0
        else:
            direction_score = 0.0

        s5 = min(8.0, compress_score + direction_score)
        score += s5
        details["BB압축"] = {"score": round(s5, 1), "max": 8,
                              "compression": round(bb_compression, 2)}

        # ── [6] 추세 일관성 (7점) ────────────────────────────
        # 순수 보조지표. 다른 신호 보강 역할.
        recent_20 = c.pct_change().iloc[-20:].dropna()
        up_ratio  = float((recent_20 > 0).mean())

        # streak: 최근 20일 내에서만 계산 (전체 시리즈 불필요)
        streak = 0
        _last20 = c.pct_change().iloc[-20:].dropna().values
        for r in reversed(_last20):
            if r > 0:   streak += 1
            else: break

        # 하락 streak도 계산 (페널티용)
        down_streak = 0
        for r in reversed(_last20):
            if r < 0:   down_streak += 1
            else: break

        streak_bonus = min(2.0, streak * 0.4)
        consistency_score = max(0.0, min(5.0, (up_ratio - 0.4) / 0.2 * 5.0))

        # 근거: Da et al.(2014) "Frog-in-Pan" — 추세 일관성은 up_ratio 단일 측정으로 충분
        # down_penalty 제거: IC 검증 없는 주관적 threshold는 오버피팅 위험
        # up_ratio < 0.4이면 consistency_score = 0, streak = 0 → s6 = 0 으로 자연 수렴
        s6 = max(0.0, min(7.0, consistency_score + streak_bonus))
        score += s6
        details["추세일관성"] = {"score": round(s6, 1), "max": 7,
                                  "up_ratio": round(up_ratio * 100, 1),
                                  "streak": streak}

        # ── [7] OBV 다이버전스 (20점) ────────────────────────
        #
        # 설계 원칙:
        #  1) 1.0 sigma 이상만 "유의미한 이동"으로 인정 (통계적 기준)
        #  2) 핵심은 '다이버전스' — 가격 보합/하락 + OBV 상승이 최고점
        #  3) 단순 동반상승(가격↑+OBV↑)은 이미 알려진 정보 → 낮은 점수
        #  4) 이산 점프 없는 연속 함수로 절벽 효과 제거
        #
        # obv: [3] 패턴C에서 이미 동일하게 계산된 obv_series 재사용 (중복 계산 제거)
        obv = obv_series
        # ── Z-score 기반 OBV 측정 (delta 방식) ─────────────────
        # 핵심: cumsum OBV 자체의 rolling std는 autocorrelation으로
        # 과대 추정됨 → 노이즈 45%가 1σ 초과 (오탐 과다).
        # 올바른 방법: OBV 일별 변화량(delta)의 std를 기준으로 사용.
        # delta std × √기간 = 해당 기간 OBV 변화의 기대 변동폭
        obv_delta       = obv.diff()  # 일별 OBV 변화량
        _delta_std      = float(obv_delta.rolling(20).std().iloc[-1]) if len(obv) >= 20 else float(obv_delta.std())
        _delta_std      = max(_delta_std, 1.0)
        obv_5d_chg_abs  = float(obv.iloc[-1] - obv.iloc[-6])  if len(obv) > 6  else 0.0
        obv_10d_chg_abs = float(obv.iloc[-1] - obv.iloc[-11]) if len(obv) > 11 else obv_5d_chg_abs
        # 분모: delta_std × √기간 (랜덤워크 기준 기대 변동폭)
        obv_std         = _delta_std  # details 호환용 별칭
        obv_5d_rel  = obv_5d_chg_abs  / max(_delta_std * (5  ** 0.5), 1.0)
        obv_10d_rel = obv_10d_chg_abs / max(_delta_std * (10 ** 0.5), 1.0)

        price_5d_ret = ret_5d

        # ── 기본 점수 (13점): 5일 Z-score 기반 시나리오 ─────
        # Z-score(obv_5d_rel) 자체는 유지하되, 맥락(가격 위치) 강화
        # 가격이 60일 저점 근처면 같은 OBV 신호도 더 의미있음
        _obv_ctx_bonus = 0.0
        if near_low_60 < 0.30:   _obv_ctx_bonus = 1.5   # 저점권 OBV 신호 가중
        elif near_low_60 < 0.50: _obv_ctx_bonus = 0.7

        # [A] 강한 Bullish Divergence — 가격 하락 + OBV 유의미 상승
        if obv_5d_rel >= 1.0 and price_5d_ret <= 0:
            price_penalty = min(2.0, abs(price_5d_ret) / 4.0 * 2.0)
            obv_base = min(13.0, 9.0 + price_penalty + _obv_ctx_bonus)

        # [B] 소폭 상승 + OBV 강세 — 0 < price_5d_ret < 2% 구간
        # ([A]에서 price_5d_ret <= 0을 이미 처리하므로 실제 적용 범위는 양수 소폭 상승)
        elif obv_5d_rel >= 1.0 and 0 < price_5d_ret < 2.0:
            obv_base = min(10.0, 8.0 + _obv_ctx_bonus)

        # [C] 약한 다이버전스 — 소폭 상승 + 강한 OBV
        elif obv_5d_rel >= 1.5 and 0 <= price_5d_ret < 3.0:
            obv_base = 6.0

        # [D] 동반상승 — 이미 반영된 정보
        elif obv_5d_rel >= 1.0 and price_5d_ret >= 1.0:
            obv_base = 4.0

        # [E] 잡음 구간
        elif 0.5 <= obv_5d_rel < 1.0:
            obv_base = 2.5

        # [F] 역다이버전스 — 분산 매도 위험
        elif obv_5d_rel < -0.5 and price_5d_ret > 0:
            obv_base = 0.0

        # [G] 약세 지속
        elif obv_5d_rel < -1.0 and price_5d_ret < 0:
            obv_base = 0.5

        else:
            obv_base = 1.5

        # ── Multi-timeframe 확인 보너스 (7점) ────────────────
        # 개선: 10일 Z-score + 20일 slope 삼중 확인
        # 5일(단기) + 10일(중기) + 20일slope(추세) 모두 일치할 때 신뢰도 최고
        if len(obv) >= 22:
            _xo = np.arange(20, dtype=float)
            _o20v = obv.iloc[-20:].values.astype(float)
            _c20v = c.iloc[-20:].values.astype(float)
            _obv_slope_20 = np.polyfit(_xo, _o20v / max(abs(_o20v[0]), 1.0), 1)[0]
            _prc_slope_20 = np.polyfit(_xo, _c20v / max(_c20v[0], 1e-8), 1)[0]
            slope_diverge = _obv_slope_20 > 0.001 and _prc_slope_20 < 0.0005
        else:
            _obv_slope_20 = 0.0
            slope_diverge = False

        # 3중 확인: 5일 Z-score + 10일 Z-score + 20일 slope 방향
        if obv_10d_rel >= 0.8 and obv_5d_rel >= 1.0 and slope_diverge:
            obv_trend = min(7.0, (obv_10d_rel - 0.8) / 0.6 * 4.0 + 4.0)  # 최대 보너스
        elif obv_10d_rel >= 0.8 and obv_5d_rel >= 1.0:
            obv_trend = min(5.0, (obv_10d_rel - 0.8) / 0.6 * 3.0 + 2.0)
        elif obv_10d_rel >= 0.5 and obv_5d_rel >= 0.5:
            obv_trend = 1.5
        elif slope_diverge and obv_5d_rel >= 0.5:
            obv_trend = 2.0  # slope만 확인되면 소폭 보너스
        elif obv_10d_rel < -0.5 and obv_5d_rel < 0:
            obv_trend = 0.0
        else:
            obv_trend = 0.8

        # ── [7] OBV 회복률·거래량 품질 필터 게이팅 ─────────────
        # 패턴C와 동일 원리: 세력 설거지 후 개미 소량 매수 과대평가 방지
        # _obv_quality_flags는 패턴C에서 이미 계산됨 (같은 obv_series 사용)
        _f7 = _obv_quality_flags
        if _f7["recovery"] < 0.15:
            # OBV 회복률 15% 미만 = 설거지 직후 극소량 반등
            # Z-score가 1σ 이상이어도 신뢰도 없음 → 70% 감점
            obv_base  *= 0.30
            obv_trend *= 0.30
        elif _f7["recovery"] < 0.25 and _f7["vol_quality"] < 0.50:
            # 회복률 25% 미만 + 매수 거래량 열세 = 개미 매수 패턴
            obv_base  *= 0.55
            obv_trend *= 0.55
        elif _f7["bottom_days"] < 3:
            # 저점 찍은 지 3일 미만 = 반전 방향 불확실
            obv_base  *= 0.70
            obv_trend *= 0.70

        s7 = min(20.0, obv_base + obv_trend)
        score += s7

        # 다이버전스 유형 태깅 (디버깅·UI 표시용)
        if obv_5d_rel >= 1.0 and price_5d_ret <= 0:
            obv_signal = "🔥 상승다이버전스"
        elif obv_5d_rel >= 1.0 and abs(price_5d_ret) < 1.0:
            obv_signal = "⚡ 횡보매집OBV"
        elif obv_5d_rel >= 1.0:
            obv_signal = "✅ OBV동반상승"
        elif obv_5d_rel < -0.5 and price_5d_ret > 0:
            obv_signal = "⚠️ 분산매도경고"
        else:
            obv_signal = "—"

        details["OBV"] = {"score": round(s7, 1), "max": 20,
                          "obv_5d_rel":    round(obv_5d_rel, 2),
                          "obv_10d_rel":   round(obv_10d_rel, 2),
                          "slope_diverge": slope_diverge,
                          "signal":        obv_signal}

        # ── [8] 옵션 신호 (15점, 데이터 있을 때만) ───────────
        # 기관이 실제로 베팅한 증거. 있을 때 가장 직접적인 신호.
        s8 = 0.0
        if opt_data:
            pc       = opt_data.get("pc_ratio", None)
            iv_pct   = opt_data.get("iv_pct",   None)
            iv_slope = opt_data.get("iv_slope",  None)

            opt_score = 0.0
            # P/C 비율 (9점)
            if pc is not None:
                if pc < 0.5:    opt_score += 9.0   # 강한 콜 우세
                elif pc < 0.7:  opt_score += 6.0   # 콜 우세
                elif pc < 0.9:  opt_score += 3.0   # 약한 콜 우세
                elif pc < 1.2:  opt_score += 0.0   # 중립
                else:           opt_score -= 3.0   # 풋 우세

            # IV 퍼센타일 (5점) — 낮을수록 저비용 콜 = 기관 확신
            if iv_pct is not None:
                if iv_pct < 20:   opt_score += 5.0
                elif iv_pct < 35: opt_score += 3.0
                elif iv_pct < 50: opt_score += 1.0
                elif iv_pct > 80: opt_score -= 2.0

            # IV 수축 보너스 (1점)
            if iv_slope is not None and iv_slope < -0.5:
                opt_score += 1.0

            s8 = max(0.0, min(15.0, opt_score))
            score += s8
            details["옵션신호"] = {
                "score":    round(s8, 1), "max": 15,
                "pc_ratio": round(pc,     2) if pc     is not None else None,
                "iv_pct":   round(iv_pct, 1) if iv_pct is not None else None,
            }

        # ── 최종 점수 & 등급 ─────────────────────────────────
        # 기본 100점 만점 (옵션 데이터 있으면 115점 만점)
        max_score = 115.0 if opt_data else 100.0
        total = round(min(max_score, max(0.0, score)), 1)
        if total >= 75:   grade, badge = "강한 매집",     "🔥"
        elif total >= 55: grade, badge = "관심 종목",     "✅"
        elif total >= 40: grade, badge = "중립",          "⚖️"
        elif total >= 25: grade, badge = "약세",          "⚠️"
        else:             grade, badge = "강한 매도압력", "🔴"

        factor = self._calc_factor(c, v, sector_rets, regime)
        # ── 크로스섹셔널 변환용 원점수 ────────────────────────
        # 실제 details 키에 맞게 매핑
        # 단기: OBV.obv_5d_rel / 거래량구조.vol_surge / 거래량구조.ud_ratio
        #       BB압축.compression(역수) / 매집탐지.pa
        # 장기: _calc_factor details
        fd = factor.get("details", {})
        _det_obv  = details.get("OBV",        {})
        _det_vol  = details.get("거래량구조",   {})
        _det_acc  = details.get("매집탐지",     {})
        _det_bb   = details.get("BB압축",       {})
        _det_mom  = details.get("모멘텀가속",   {})
        _det_rs   = details.get("RS",           {})
        # ── 신규 raw 신호 계산 ──────────────────────────────────
        # [단기 신규1] RS vs SPY — 5일 상대강도
        # 근거: Levy(1967), IBD RS Rating — 강한 종목이 더 강해지는 관성
        # 직교성: vol_surge(거래량 차원)와 독립 — 가격 상대성 차원
        _raw_rs = rs_vs_spy  # 이미 calc()에서 계산됨 (ret_5d - spy_r5)

        # [단기 신규2] 5일 가격 가속도
        # 근거: Da et al.(2014) Frog-in-Pan — 지속적 소폭 상승이 급등보다 강함
        # 수식: 5일 수익률 - 20일 수익률/4 (최근 1주가 전월 일평균 대비 얼마나 빠른가)
        # 직교성: vol_surge(거래량), bb_z(변동성 수렴)와 다른 차원
        _raw_accel = ret_5d - (ret_20d / 4.0)

        # [장기 신규1] 52주 고점 근접도
        # 근거: George & Hwang(2004) — AFA 최우수논문
        #       52주 고점 근처 종목이 이후 6~12M 초과수익 (앵커링 편향 역이용)
        # 수식: (현재가 - 52주저점) / (52주고점 - 52주저점) → 0~1
        n_c = len(c)
        if n_c >= 252:
            _h52 = float(c.iloc[-252:].max())
            _l52 = float(c.iloc[-252:].min())
        else:
            _h52 = float(c.max())
            _l52 = float(c.min())
        _range52  = max(_h52 - _l52, float(c.iloc[-1]) * 0.01)
        _raw_52w  = (float(c.iloc[-1]) - _l52) / _range52  # 0~1

        # [장기 신규2] 추세 일관성 (상승일 비율)
        # 근거: Da et al.(2014) Frog-in-Pan — 지속적 작은 상승 = 강한 추세
        # up_ratio: 이미 calc()에서 계산됨 (20일 상승일 비율)
        _raw_cons = up_ratio  # 0~1

        raw = {
            # ── 단기 신호 4개 (OBV/UD비율 제거, RS/가속도 신규) ──
            # IC 4회 연속 음수인 obv_z, div 제거
            # IC 양수 + 직교성 확인된 신호로 대체
            "vol_z":  float(_det_vol.get("vol_surge", 1)) - 1.0,  # 거래량 서지 — IC +0.022
            "bb_z":   1.0 - float(_det_bb.get("compression", 1)), # BB압축 역수 — IC +0.010, vol과 직교
            "rs":     _raw_rs,                                      # RS vs SPY 5일 — Levy(1967)
            "accel":  _raw_accel,                                   # 가격 가속도 — Da et al.(2014)
            # ── 장기 신호 4개 (OBV장기/퀄리티 제거, 52w/일관성 신규) ──
            "mom":    float(fd.get("모멘텀12M1M", {}).get("score", 50)) - 50,  # IC +0.045
            "bab":    float(fd.get("저변동성BAB", {}).get("score", 50)) - 50,  # IC +0.032
            "w52":    _raw_52w,    # 52주 고점 근접도 — George & Hwang(2004)
            "cons":   _raw_cons,   # 추세 일관성 — Da et al.(2014) Frog-in-Pan
        }
        return {
            "total":   total,
            "grade":   grade,
            "badge":   badge,
            "details": details,
            "factor":  factor,
            "raw":     raw,
            "ret_1d":  round(ret_1d,  2),
            "ret_5d":  round(ret_5d,  2),
            "ret_20d": round(ret_20d, 2),
        }


    _regime_cache: dict = {}
    _REGIME_TTL: int = 3600

    @classmethod
    def detect_regime(cls, spy_c: "pd.Series", vix_val: float = 20.0) -> str:
        """
        현재 시장 국면 감지 — 'bull'|'bear'|'sideways' (1시간 캐시).

        ※ Look-ahead bias 주의사항:
           이 메서드는 현재 시점(실시간 대시보드)에서만 호출됨.
           과거 백테스트 루프 내에서 호출하면 미래 국면 정보 누출 발생 → 금지.
           현재 용도(실시간 신호 생성)에서는 정상 — 오늘 국면으로 오늘 신호 평가.

        기준:
          bull     : SPY > 200MA AND VIX < 20
          bear     : SPY < 200MA OR  VIX >= 30
          sideways : 그 외
        """
        import time as _t
        now = _t.time()
        if cls._regime_cache.get("ts", 0) + cls._REGIME_TTL > now:
            return cls._regime_cache.get("regime", "sideways")
        regime = "sideways"
        try:
            if len(spy_c) >= 200:
                ma200 = float(spy_c.rolling(200).mean().iloc[-1])
                cur   = float(spy_c.iloc[-1])
                if   cur > ma200 and vix_val < 20:  regime = "bull"
                elif cur < ma200 and vix_val > 25:  regime = "bear"
        except Exception as _e:
            logger.debug("예외 무시: %s", _e)
        cls._regime_cache = {"regime": regime, "ts": now}
        return regime

    def _calc_factor(self, c: pd.Series, v: pd.Series,
                     sector_rets: Dict[str, float],
                     regime: str = "sideways") -> Dict:
        """
        장기 FactorScore 0~100 — 국면별 동적 가중치
        
        [v2 개선 — IC 역전 문제 해결]
        기존 문제:
          f1(12M 절대 모멘텀): 급등 후 역회귀 → IC=-0.095 역방향
          f2(저변동성 BAB):    소형 풀에서 고변동 성장주가 더 좋음 → IC=-0.079 역방향
          f4(변동성 비율):     BAB와 동일한 역전
        
        개선:
          f1 → 모멘텀 '지속성' (6M 가속도 + Calmar형 수익/MDD)
          f2 → Vol-adjusted momentum (샤프형: 수익/변동성)
          f4 → 추세 퀄리티 (200일 이평선 + 52주 고점 근접도)
        """
        n = len(c)

        # ── f1: 모멘텀 지속성 ────────────────────────────────
        # 단순 12M 수익률(이미 반영된 정보)이 아닌
        # 최근 6M이 12M 대비 가속되고 있는가 + 낙폭 대비 수익(Calmar형)
        if n >= 253:   # c.iloc[-253] 접근 → n이 253 이상이어야 안전
            r12  = float(c.iloc[-1] / c.iloc[-253] - 1)
            r6   = float(c.iloc[-1] / c.iloc[-127] - 1) if n >= 127 else r12 / 2
            r1m  = float(c.iloc[-1] / c.iloc[-22]  - 1) if n >= 22  else 0.0
            # 12M 기간 내 MDD (최대낙폭)
            _roll_max_252 = c.iloc[-252:].cummax()
            _mdd_252      = float(((c.iloc[-252:] - _roll_max_252) / _roll_max_252).min())
            # 모멘텀 가속도: 최근 6M 수익이 전체 12M의 절반보다 크면 가속
            accel = r6 - (r12 / 2.0)
            # Calmar형: 12M 수익 / |MDD| (하락 대비 얼마나 올랐나)
            calmar = r12 / max(abs(_mdd_252), 0.05)
            # 종합 (가속도 60% + Calmar 40%)
            f1_raw = accel * 3.0 * 0.6 + min(calmar / 3.0, 1.0) * 0.4
            f1 = max(0.0, min(1.0, f1_raw * 0.5 + 0.5))
        elif n >= 127:
            r6  = float(c.iloc[-1] / c.iloc[-127] - 1)
            r1m = float(c.iloc[-1] / c.iloc[-22]  - 1) if n >= 22 else 0.0
            f1 = max(0.0, min(1.0, (r6 - r1m * 2 + 0.05) * 5.0 * 0.5 + 0.5))
        else:
            r6  = float(c.pct_change(min(n-1, 63)).iloc[-1])
            f1  = max(0.0, min(1.0, (r6 + 0.05) * 4.0 * 0.5 + 0.5))

        # ── f2: Vol-adjusted Momentum (샤프형) ──────────────
        # 기존 BAB(저변동성 우대)는 소형 성장주 풀에서 역전됨
        # → 변동성 자체를 벌하지 말고 '변동성 대비 수익'으로 평가
        # 고변동성이어도 그 이상 올랐으면 좋은 것
        if n >= 126:
            _ret_6m  = float(c.iloc[-1] / c.iloc[-127] - 1) if n >= 127 else 0.0
            _vol_63  = float(c.pct_change().rolling(63).std().iloc[-1]) * (252 ** 0.5)
            _vol_63  = max(_vol_63, 0.05)  # 최소 5% 연변동성
            sharpe6m = _ret_6m / _vol_63   # 샤프형 지표
            # 정규화: sharpe6m ≈ [-2, +3] → [0, 1]
            f2 = max(0.0, min(1.0, (sharpe6m + 0.5) / 2.5))
        else:
            f2 = 0.5

        # ── f3: OBV 장기 추세 ────────────────────────────────
        # 기존 유지 (IC +0.010, 정방향으로 작동 중)
        obv = (v * np.sign(c.pct_change().fillna(0))).cumsum()
        if n >= 64:    # obv.iloc[-64] 접근 → n이 64 이상이어야 안전
            o63      = float(obv.iloc[-1] - obv.iloc[-64])
            _odelta  = obv.diff()
            _odstd63 = max(float(_odelta.rolling(63).std().iloc[-1]), 1.0)
            f3 = max(0.0, min(1.0, (o63 / max(_odstd63 * (63 ** 0.5), 1.0) + 1.5) / 3.0))
        else:
            f3 = 0.5

        # ── f4: 추세 퀄리티 ─────────────────────────────────
        # 기존 변동성 비율(BAB와 동일 역전)을 대체
        # 200일 이평선 위 여부 + 52주 고점 근접도
        # → '지금 추세가 살아있는가'를 직접 측정
        if n >= 200:
            _ma200    = float(c.rolling(200).mean().iloc[-1])
            _price    = float(c.iloc[-1])
            _high52   = float(c.iloc[-252:].max()) if n >= 252 else float(c.max())
            _low52    = float(c.iloc[-252:].min()) if n >= 252 else float(c.min())
            # 200일선 위: 0.5점, 아래: 0점
            _above    = 0.5 if _price > _ma200 else 0.0
            # 52주 범위에서의 상대 위치 (0=저점, 1=고점)
            _range52  = max(_high52 - _low52, _price * 0.01)
            _pos52    = (_price - _low52) / _range52
            # 상위 60% 이상이면 강한 추세
            _trend    = max(0.0, min(0.5, (_pos52 - 0.4) / 0.6 * 0.5))
            f4 = _above + _trend
        elif n >= 63:
            _ma63  = float(c.rolling(min(63, n)).mean().iloc[-1])
            f4 = 0.6 if float(c.iloc[-1]) > _ma63 else 0.35
        else:
            f4 = 0.5

        # ── 국면별 가중치 ────────────────────────────────────
        # mom=지속성모멘텀, vam=변동성조정모멘텀, obv=OBV장기, qual=추세퀄리티
        W_MAP = {
            "bull":     {"mom": 0.35, "vam": 0.30, "obv": 0.25, "qual": 0.10},
            "bear":     {"mom": 0.15, "vam": 0.25, "obv": 0.20, "qual": 0.40},
            "sideways": {"mom": 0.25, "vam": 0.30, "obv": 0.30, "qual": 0.15},
        }
        w  = W_MAP.get(regime, W_MAP["sideways"])
        ft = round((f1 * w["mom"] + f2 * w["vam"] + f3 * w["obv"] + f4 * w["qual"]) * 100, 1)

        if   ft >= 70: fg, fb = "장기매력",   "💎"
        elif ft >= 50: fg, fb = "장기관심",   "📈"
        elif ft >= 35: fg, fb = "장기중립",   "📊"
        else:          fg, fb = "장기비선호", "📉"

        # details — 백테스트 raw 추출과 키 이름 호환 유지
        return {
            "total": ft, "grade": fg, "badge": fb, "regime": regime,
            "details": {
                "모멘텀12M1M": {
                    "score":    round(f1 * 100, 1),
                    "ret_12m":  round(float(c.pct_change(min(252, n-1)).iloc[-1]) * 100, 1) if n >= 63 else 0.0,
                    "ret_1m":   round(float(c.pct_change(min(21, n-1)).iloc[-1]) * 100, 1)  if n >= 21 else 0.0,
                    "note":     "가속도+Calmar형",
                },
                "저변동성BAB": {
                    "score":      round(f2 * 100, 1),
                    "annual_vol": round(float(c.pct_change().rolling(min(63, n)).std().iloc[-1]) * (252**0.5) * 100, 1) if n >= 20 else 0.0,
                    "note":       "Vol-adj모멘텀(샤프형)",
                },
                "OBV장기":   {"score": round(f3 * 100, 1)},
                "퀄리티":    {
                    "score": round(f4 * 100, 1),
                    "note":  "200일이평+52주고점근접",
                },
            },
        }


def _last_13f_date() -> str:
    """
    13F 신고 마감일 추정.
    분기 종료(3/31, 6/30, 9/30, 12/31) 후 45일이 신고 마감.
    반환 형식: "2024 Q3 (2024-11-14 기준)" 처럼 명확하게 표시.
    """
    today = datetime.now()
    # 분기 종료일 목록 (올해 + 작년)
    quarters = []
    for y in [today.year - 1, today.year]:
        quarters += [
            datetime(y, 3, 31), datetime(y, 6, 30),
            datetime(y, 9, 30), datetime(y, 12, 31),
        ]
    # 45일 더한 마감일 중 오늘 이전인 가장 최근 것
    deadlines = [(q, q + timedelta(days=45)) for q in quarters]
    past = [(q, d) for q, d in deadlines if d <= today]
    if not past:
        return "데이터 날짜 미상"
    q_end, deadline = past[-1]
    # 분기 이름
    qnum = {3:"Q1", 6:"Q2", 9:"Q3", 12:"Q4"}[q_end.month]
    return f"{q_end.year} {qnum} (마감 {deadline.strftime('%Y-%m-%d')}, ~{(today-deadline).days}일 전 데이터)"


# ──────────────────────────────────────────────────────────────
#  데이터 수집
# ──────────────────────────────────────────────────────────────
class DataFetcher:

    # ── 클래스 레벨 캐시 (프로세스 생존 동안 유지) ────────────
    _price_cache: Dict[str, Dict]  = {}
    _slow_cache:  Dict[str, Dict]  = {}
    _ref_cache:   Dict[str, float] = {}
    _ref_cache_ts: float           = 0.0

    # ── 스레드 안전 락 ─────────────────────────────────────────
    import threading as _threading
    _price_lock = _threading.Lock()   # _price_cache 읽기/쓰기 보호
    _slow_lock  = _threading.Lock()   # _slow_cache  읽기/쓰기 보호

    _PRICE_TTL  = 30  * 60
    _SLOW_TTL   = 180 * 60
    _DISK_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".price_cache.pkl")

    def __init__(self):
        self._sse = SmartScoreEngine()
        self._load_disk_cache()

    @classmethod
    def _load_disk_cache(cls):
        """프로세스 시작 시 디스크 캐시 로드 (30분 이내 데이터만)."""
        if cls._price_cache:
            return  # 이미 로드됨
        try:
            if not os.path.exists(cls._DISK_CACHE):
                return
            with open(cls._DISK_CACHE, "rb") as f:
                saved = pickle.load(f)
            now = time.time()
            loaded = 0
            for tk, entry in saved.items():
                if (now - entry["ts"]) < cls._PRICE_TTL:
                    cls._price_cache[tk] = entry
                    loaded += 1
            if loaded:
                print(f"  💾 디스크 캐시 로드: {loaded}개 종목")
        except Exception as _e:
            logger.debug("예외 무시: %s", _e)

    @classmethod
    def _save_disk_cache(cls):
        """현재 캐시를 디스크에 저장."""
        try:
            with open(cls._DISK_CACHE, "wb") as f:
                pickle.dump(dict(cls._price_cache), f)
        except Exception as _e:
            logger.debug("예외 무시: %s", _e)

    def _load_ref_rets(self):
        """SPY + 모든 섹터 ETF의 5일 수익률 — 30분 캐시. bulk 1회 호출."""
        now = time.time()
        if DataFetcher._ref_cache and (now - DataFetcher._ref_cache_ts) < self._PRICE_TTL:
            return
        tickers = ["SPY"] + list(SECTORS.keys())
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                raw = yf.download(tickers, period="1mo", interval="1d",
                                  group_by="column", auto_adjust=False, progress=False)
            for tk in tickers:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        # group_by="column": level0=Price, level1=Ticker
                        tickers_in = raw.columns.get_level_values(1).unique().tolist()
                        if tk not in tickers_in:
                            DataFetcher._ref_cache[tk] = 0.0; continue
                        c_raw = raw[("Close", tk)] if ("Close", tk) in raw.columns else raw[("Adj Close", tk)]
                        c = pd.to_numeric(c_raw, errors="coerce").dropna()
                    else:
                        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                        c = pd.to_numeric(raw[col], errors="coerce").dropna()
                    DataFetcher._ref_cache[tk] = float(c.pct_change(5).iloc[-1]) * 100 if len(c) >= 5 else 0.0
                except Exception as _e:
                    DataFetcher._ref_cache[tk] = 0.0
        except Exception as _e:
            for tk in tickers:
                DataFetcher._ref_cache[tk] = 0.0
        DataFetcher._ref_cache_ts = now

    # 실시간 현재가 캐시 (5분 TTL — price_cache와 동일)
    _rt_cache: Dict[str, Dict] = {}

    def _get_realtime_price(self, ticker: str) -> float:
        """
        fast_info로 현재가(regularMarketPrice) 반환.
        장 마감 후에도 마지막 거래가를 줌.
        실패 시 None 반환 → 호출부에서 일봉 Close로 fallback.
        """
        now = time.time()
        cached = DataFetcher._rt_cache.get(ticker)
        if cached and (now - cached["ts"]) < self._PRICE_TTL:
            return cached["price"]
        try:
            fi = yf.Ticker(ticker).fast_info
            # yfinance 버전별 속성명 차이 대응
            price = (
                getattr(fi, "last_price", None)
                or getattr(fi, "regularMarketPrice", None)
                or (fi["lastPrice"] if hasattr(fi, "__getitem__") else None)
            )
            price = float(price) if price and float(price) > 0 else None
        except Exception:
            price = None
        DataFetcher._rt_cache[ticker] = {"price": price, "ts": now}
        return price

    def _get_price_data(self, ticker: str) -> pd.DataFrame:
        """일봉 OHLCV — 30분 캐시. 스레드 락으로 레이스 컨디션 방지."""
        now = time.time()
        with DataFetcher._price_lock:
            cached = DataFetcher._price_cache.get(ticker)
            if cached and (now - cached["ts"]) < self._PRICE_TTL:
                return cached["df"].copy()

        # ticker를 인자로 명시 전달 — 클로저 캡처 금지
        raw = DataFetcher._download_and_clean(ticker, "1y")
        if raw.empty:
            raw = DataFetcher._download_and_clean(ticker, "2y")

        # 빈 DataFrame은 캐시에 저장하지 않음
        # → 상장폐지/일시 오류 종목이 캐시를 오염시켜 계속 빈 데이터 반환하는 버그 방지
        # → 다음 호출 시 재시도 가능
        if not raw.empty:
            with DataFetcher._price_lock:
                DataFetcher._price_cache[ticker] = {"df": raw, "ts": now}

        return raw.copy() if not raw.empty else pd.DataFrame()

    @staticmethod
    def _download_and_clean(ticker: str, period: str) -> pd.DataFrame:
        """
        yf.Ticker.history() 사용 — yf.download()는 멀티스레드에서 세션 오염 버그 있음.
        Ticker 객체는 인스턴스별로 독립적이라 스레드 세이프.
        """
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                tk_obj = yf.Ticker(ticker)
                df = tk_obj.history(period=period, interval="1d")

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.copy()

            # history()는 단층 컬럼 반환이 기본 — 혹시 MultiIndex면 해제
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [str(col).strip().title() for col in df.columns]

            for col in ("Close", "Volume", "High", "Low"):
                if col not in df.columns:
                    return pd.DataFrame()

            df = df.dropna(subset=["Close"]).sort_index()

            for col in ("Close", "Volume", "High", "Low", "Open"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df.dropna(subset=["Close"])
        except Exception:
            return pd.DataFrame()

    def _get_slow_data(self, ticker: str) -> Dict:
        """옵션·공매도·어닝 — 3시간 캐시. 스레드 락 보호."""
        now = time.time()
        with DataFetcher._slow_lock:
            cached = DataFetcher._slow_cache.get(ticker)
            if cached and (now - cached["ts"]) < self._SLOW_TTL:
                return cached["data"]
        # 락 밖에서 API 호출
        data = self.fetch_option_short_earning(ticker)
        with DataFetcher._slow_lock:
            DataFetcher._slow_cache[ticker] = {"data": data, "ts": now}
        return data

    def fetch_option_short_earning(self, ticker: str) -> Dict:
        """
        종목별 옵션 P/C·IV, 공매도 비율, 어닝 발표일을 한 번에 수집.
        실패 항목은 None으로 반환 (부분 실패 허용).
        """
        result = {
            "pc_ratio":    None,
            "iv_pct":      None,
            "iv_slope":    None,
            "short_pct":   None,   # 공매도 비율 (float, %)
            "short_ratio": None,   # 숏커버 일수 (float)
            "earn_date":   None,   # 다음 실적 발표일 (str)
            "earn_days":   None,   # 오늘부터 몇 일 후 (int)
        }
        try:
            tk = yf.Ticker(ticker)

            # ── 옵션 데이터 ────────────────────────────────────
            try:
                exps = tk.options
                if exps:
                    # 가장 가까운 만기 2개 평균으로 P/C 계산
                    total_call_oi = 0; total_put_oi = 0
                    iv_list = []
                    for exp in exps[:2]:
                        chain = tk.option_chain(exp)
                        calls = chain.calls
                        puts  = chain.puts
                        total_call_oi += int(calls["openInterest"].sum())
                        total_put_oi  += int(puts["openInterest"].sum())
                        # ATM 근처 IV (현재가 ±10% 내) — fast_info 버전 호환
                        try:
                            fi = tk.fast_info
                            price_now = float(fi["lastPrice"] if hasattr(fi, '__getitem__')
                                              else getattr(fi, 'last_price', 0) or 0)
                        except Exception as _e:
                            price_now = 0.0
                        if price_now > 0:
                            atm_calls = calls[
                                (calls["strike"] >= price_now * 0.9) &
                                (calls["strike"] <= price_now * 1.1)
                            ]
                            if not atm_calls.empty:
                                iv_list.extend(atm_calls["impliedVolatility"].dropna().tolist())

                    if total_call_oi > 0:
                        result["pc_ratio"] = round(total_put_oi / max(total_call_oi, 1), 3)
                    if iv_list:
                        iv_now = float(np.mean(iv_list)) * 100
                        # IV 퍼센타일 근사: 히스토리 없이 절대값으로 구간 추정
                        # 30% 미만 = 저점, 30~60 = 보통, 60% 이상 = 고점
                        if iv_now < 20:   result["iv_pct"] = 10
                        elif iv_now < 30: result["iv_pct"] = 25
                        elif iv_now < 45: result["iv_pct"] = 45
                        elif iv_now < 60: result["iv_pct"] = 65
                        else:             result["iv_pct"] = 85
            except Exception as _e:
                logger.debug("예외 무시: %s", _e)

            # ── 공매도 데이터 ──────────────────────────────────
            # ETF는 Yahoo Finance quoteSummary에 fundamentals 없음 → 404
            # fast_info.quote_type으로 ETF 판별 후 스킵
            try:
                _qtype = ""
                try:    _qtype = str(getattr(tk.fast_info, "quote_type", "") or "").upper()
                except Exception as _e:
                    logger.debug("예외 무시: %s", _e)
                _is_etf = (_qtype == "ETF") or ticker.upper() in {
                    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","LQD",
                    "XLK","XLF","XLV","XLE","XLI","XLB","XLRE","XLP","XLU","XLC",
                    "XLY","ITA","XAR","SOXX","CIBR","SKYY","XBI","LIT","FINX","ICLN",
                    "VXX","UVXY","SQQQ","TQQQ","SPXL","SPXS","ARKK","ARKG","ARKW",
                }
                if not _is_etf:
                    import io as _io, contextlib as _ctx
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    with _TPE(max_workers=1) as _p:
                        # tk.info 호출 시 stderr 억제 (404 경고 방지)
                        def _get_info():
                            with _ctx.redirect_stderr(_io.StringIO()):
                                return tk.info
                        _fut = _p.submit(_get_info)
                        try:    info = _fut.result(timeout=6)
                        except Exception as _e:
                            logger.debug("info 조회 타임아웃: %s", _e)
                            info = {}
                    shares_out   = info.get("sharesOutstanding", 0) or 0
                    shares_short = info.get("sharesShort",       0) or 0
                    short_ratio  = info.get("shortRatio",        0) or 0
                    if not shares_out:
                        shares_out = getattr(tk.fast_info, "shares", 0) or 0
                    if shares_out and shares_short:
                        result["short_pct"]   = round(shares_short / shares_out * 100, 2)
                        result["short_ratio"] = round(float(short_ratio or 0), 1)
            except Exception as _e:
                logger.debug("예외 무시: %s", _e)

            # ── 어닝 캘린더 ───────────────────────────────────
            try:
                cal = tk.calendar
                # yfinance 버전별로 dict 또는 DataFrame 반환 — .empty는 DataFrame만 지원
                cal_ok = False
                if isinstance(cal, dict) and cal:
                    cal_ok = True
                elif hasattr(cal, 'empty') and not cal.empty:
                    cal_ok = True

                if cal_ok:
                    if isinstance(cal, dict):
                        ed_raw = cal.get("Earnings Date")
                        earn_dt = (ed_raw[0] if isinstance(ed_raw, list) and ed_raw
                                   else ed_raw)
                    else:
                        earn_dt = cal.iloc[0, 0] if len(cal.columns) > 0 else None

                    if earn_dt is not None:
                        if hasattr(earn_dt, 'date'):
                            earn_dt = earn_dt.date()
                        today    = datetime.now().date()
                        days_out = (earn_dt - today).days
                        if 0 <= days_out <= 60:
                            result["earn_date"] = str(earn_dt)
                            result["earn_days"] = days_out
            except Exception as _e:
                logger.debug("예외 무시: %s", _e)

        except Exception as _e:
            logger.debug("예외 무시: %s", _e)
        return result

    def fetch_institutional_data(self) -> Dict:
        """
        내부자 매수 (SEC Form 4 기반) + 기관 보유 비율 (major_holders).
        Senate/House S3 API 403 → yfinance insider_purchases / major_holders 로 대체.
        """
        result = {
            "congress": [], "congress_updated": None,
            "top13f": [],   "top13f_updated":   None,
            "error": None,
        }
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fmt_amt(val):
            try:
                n = float(val or 0)
                if n >= 1e9:  return f"${n/1e9:.1f}B"
                if n >= 1e6:  return f"${n/1e6:.1f}M".replace(".0M","M")
                if n >= 1e3:  return f"${int(n/1e3)}K"
                return f"${int(n)}"
            except Exception as _e:
                logger.debug("금액 변환 오류: %s", _e)
                return "금액 미상"

        # ── [A] 내부자 매수: yfinance insider_purchases (SEC Form 4) ──
        WATCH = ["NVDA","MSFT","AAPL","AMZN","META","TSLA",
                 "GOOGL","AMD","PLTR","CRWD","PANW","SPY",
                 "AVGO","LLY","JPM","COIN","MSTR","HOOD"]

        def _get_insiders(sym):
            rows = []
            try:
                with ThreadPoolExecutor(max_workers=1) as p:
                    df = p.submit(lambda: yf.Ticker(sym).insider_purchases).result(timeout=8)
                if df is None or (hasattr(df, 'empty') and df.empty):
                    return rows
                for _, row in df.iterrows():
                    try:
                        date_val = str(row.get("startDate") or row.get("Start Date",""))[:10]
                        shares   = int(row.get("shares") or row.get("Shares", 0) or 0)
                        val      = float(row.get("value") or row.get("Value", 0) or 0)
                        insider  = str(row.get("filerName") or row.get("Filer Name",""))
                        if not date_val or shares <= 0: continue
                        dt = datetime.strptime(date_val, "%Y-%m-%d")
                        days_ago = (datetime.now() - dt).days
                        if days_ago > 90: continue
                        rows.append({
                            "ticker": sym, "type": "매수",
                            "amount": _fmt_amt(val),
                            "amount_raw": str(val),
                            "rep": insider, "party": "내부자",
                            "date": date_val, "days_ago": days_ago,
                        })
                    except Exception as _e:
                        logger.debug("내부자 행 파싱 오류: %s", _e)
                        continue
            except Exception as _e:
                logger.debug("예외 무시: %s", _e)
            return rows

        trades = []
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                futs = {pool.submit(_get_insiders, s): s for s in WATCH}
                for fut in as_completed(futs, timeout=35):
                    try: trades.extend(fut.result(timeout=5))
                    except Exception as _e:
                        logger.debug("예외 무시: %s", _e)
        except Exception as e:
            result["error"] = (result["error"] or "") + f" insider:{type(e).__name__}"

        if trades:
            trades.sort(key=lambda x: x["days_ago"])
            result["congress"] = trades[:20]
            result["congress_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"  ✅ 내부자 매수 {len(result['congress'])}건")
        else:
            print(f"  ⚠️ 내부자 매수 없음 (최근 90일)")

        # ── [B] 기관 보유: major_holders ────────────────────────
        SYMS = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL",
                "TSLA","AVGO","LLY","JPM","V","UNH",
                "AMD","PLTR","PANW","CRWD","LMT","RTX"]

        def _inst_pct(sym):
            try:
                with ThreadPoolExecutor(max_workers=1) as p:
                    mh = p.submit(lambda: yf.Ticker(sym).major_holders).result(timeout=6)
                if mh is None or (hasattr(mh, 'empty') and mh.empty):
                    return sym, None
                for _, row in mh.iterrows():
                    label = str(row.iloc[1]).lower() if len(row) > 1 else ""
                    if "institution" in label:
                        pct = float(row.iloc[0])
                        return sym, (pct * 100 if pct <= 1.0 else pct)
            except Exception as _e:
                logger.debug("예외 무시: %s", _e)
            return sym, None

        inst_summary = []
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                futs = {pool.submit(_inst_pct, s): s for s in SYMS}
                for fut in as_completed(futs, timeout=30):
                    try:
                        sym, pct = fut.result(timeout=5)
                        if pct is not None:
                            inst_summary.append({
                                "ticker": sym,
                                "inst_pct": round(float(pct), 1),
                                "as_of": _last_13f_date(),
                            })
                    except Exception as _e:
                        logger.debug("예외 무시: %s", _e)
        except Exception as e:
            print(f"  ⚠️ 기관 보유 실패: {type(e).__name__}: {e}")
            result["error"] = (result["error"] or "") + f" 13f:{type(e).__name__}"

        if inst_summary:
            inst_summary.sort(key=lambda x: x["inst_pct"], reverse=True)
            result["top13f"] = inst_summary[:15]
            result["top13f_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"  ✅ 기관 보유 {len(result['top13f'])}개 종목")
        else:
            print(f"  ⚠️ 기관 보유 데이터 없음")

        return result
    def _intraday_price_ret_vol(self, ticker: str, avg_vol_20: float,
                                 mstat: Dict,
                                 avg_vol_1d: float = 0.0
                                 ) -> Tuple[float, float, float, float]:
        """
        5분봉 기반 현재가·수익률·거래량배수 반환 (페이스 환산 없음).
        반환: (price, ret_1d, vol_ratio_20d, vol_ratio_1d)
          vol_20d = today_누적 / 20일평균
          vol_1d  = today_누적 / 전일_일봉
        """
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ri = yf.Ticker(ticker).history(period="2d", interval="5m")
            if isinstance(ri.columns, pd.MultiIndex):
                ri.columns = ri.columns.get_level_values(0)
            ri.columns = [str(col).strip().title() for col in ri.columns]
            if ri.empty or "Close" not in ri.columns:
                return None, None, None, None
            ci = pd.to_numeric(ri["Close"],  errors="coerce").dropna()
            vi = pd.to_numeric(ri["Volume"], errors="coerce").dropna()
            if len(ci) < 2: return None, None, None, None
            today      = ci.index[-1].date()
            today_mask = [x.date() == today for x in ci.index]
            prev_mask  = [not x for x in today_mask]
            today_vi   = vi.iloc[[i for i,x in enumerate(today_mask) if x]]
            prev_vi    = vi.iloc[[i for i,x in enumerate(prev_mask)  if x]]
            today_vol  = float(today_vi.sum()) if len(today_vi) > 0 else 0.0
            prev_vol   = float(prev_vi.sum())  if len(prev_vi)  > 0 else float(avg_vol_1d or avg_vol_20)
            today_ci   = ci.iloc[[i for i,x in enumerate(today_mask) if x]]
            price      = float(ci.iloc[-1])
            open_p     = float(today_ci.iloc[0]) if len(today_ci) > 0 else float(ci.iloc[0])
            ret_1d     = (price / open_p - 1) * 100 if open_p > 0 else 0.0
            vol_20d    = today_vol / max(avg_vol_20, 1.0)
            vol_1d     = today_vol / max(prev_vol,   1.0)
            return price, ret_1d, vol_20d, vol_1d
        except Exception as _e:
            return None, None, None, None

    def fetch_sector(self, mstat: Dict) -> pd.DataFrame:
        intraday = mstat["intraday"]
        print(f"  📡 섹터 수집 {'[실시간 5분봉]' if intraday else '[일봉]'} ...")
        rows = []
        # ── 캐시에서 데이터 추출 (bulk 다운로드는 run_once에서 완료) ──
        all_tickers = ["SPY"] + list(SECTORS.keys())
        bulk = {}
        for tk in all_tickers:
            with DataFetcher._price_lock:
                cached = DataFetcher._price_cache.get(tk)
            if cached and not cached["df"].empty:
                bulk[tk] = cached["df"].copy()

        spy_c = pd.to_numeric(bulk["SPY"]["Close"], errors="coerce").dropna() if "SPY" in bulk else pd.Series(dtype=float)

        for etf, info in SECTORS.items():
            name, emoji, group = info
            try:
                raw_d = bulk.get(etf)
                if raw_d is None or raw_d.empty or "Close" not in raw_d.columns: continue
                cd = pd.to_numeric(raw_d["Close"],  errors="coerce").dropna()
                vd = pd.to_numeric(raw_d["Volume"], errors="coerce").dropna()
                vd = vd.reindex(cd.index)
                if len(cd) < 5: continue

                avg_vol_20 = float(vd.rolling(20).mean().iloc[-1]) if len(vd) > 20 else float(vd.mean())
                avg_vol_1d = float(vd.iloc[-2]) if len(vd) > 1 else avg_vol_20
                ret_5d  = float(cd.pct_change(5).iloc[-1])  * 100
                ret_20d = float(cd.pct_change(20).iloc[-1]) * 100

                if intraday:
                    p,r1,vr20,vr1 = self._intraday_price_ret_vol(etf,avg_vol_20,mstat,avg_vol_1d)
                    price         = p   if p   is not None else float(cd.iloc[-1])
                    ret_1d        = r1  if r1  is not None else float(cd.pct_change(1).iloc[-1])*100
                    vol_ratio_20d = vr20 if vr20 is not None else float(vd.iloc[-1]/max(avg_vol_20,1))
                    vol_ratio_1d  = vr1  if vr1  is not None else float(vd.iloc[-1]/max(avg_vol_1d,1))
                else:
                    price     = float(cd.iloc[-1])
                    rt = self._get_realtime_price(etf)
                    if rt is not None: price = rt
                    ret_1d        = float(cd.pct_change(1).iloc[-1])*100
                    vol_ratio_20d = float(vd.iloc[-1]/max(avg_vol_20,1))
                    vol_ratio_1d  = float(vd.iloc[-1]/max(avg_vol_1d,1))

                rs = 1.0
                if len(spy_c) >= 20:
                    spy_r = float(spy_c.pct_change(20).iloc[-1]) * 100
                    rs    = ret_20d / (abs(spy_r) + 1e-8)

                momentum = ret_1d * 0.4 + ret_5d * 0.3 + ret_20d * 0.3
                rows.append({
                    "etf": etf, "name": name, "emoji": emoji, "group": group,
                    "price": round(price, 2),
                    "ret_1d": round(ret_1d, 2), "ret_5d": round(ret_5d, 2),
                    "ret_20d": round(ret_20d,2), "vol_ratio": round(vol_ratio_20d,2), "vol_ratio_1d": round(vol_ratio_1d,2),
                    "rs_spy": round(rs, 2), "momentum": round(momentum, 2),
                })
            except Exception as e:
                print(f"    ⚠️ {etf}: {e}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("momentum", ascending=False).reset_index(drop=True)
        print(f"  ✅ 섹터 {len(df)}개")
        return df

    def fetch_macro(self, mstat: Dict) -> Dict:
        intraday = mstat["intraday"]
        print(f"  📡 매크로 수집 {'[실시간]' if intraday else '[일봉]'} ...")
        result = {}

        for ticker, name in MACRO_TICKERS.items():
            try:
                # 캐시에서 조회 (run_once에서 bulk 다운로드 완료)
                with DataFetcher._price_lock:
                    cached = DataFetcher._price_cache.get(ticker)
                if not cached or cached["df"].empty:
                    continue
                df = cached["df"].copy()
                if "Close" not in df.columns:
                    continue
                c = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if c.empty:
                    continue
                price  = float(c.iloc[-1])
                ret_1d = float(c.pct_change(1).iloc[-1]) * 100

                if intraday:
                    try:
                        with contextlib.redirect_stderr(io.StringIO()):
                            ri = yf.Ticker(ticker).history(period="1d", interval="5m")
                        if ri is not None and not ri.empty:
                            ri.columns = [str(col).strip().title() for col in ri.columns]
                            if "Close" in ri.columns:
                                ci = pd.to_numeric(ri["Close"], errors="coerce").dropna()
                                if len(ci) >= 2:
                                    price  = float(ci.iloc[-1])
                                    ret_1d = (price / float(ci.iloc[0]) - 1) * 100
                    except Exception as _e:
                        logger.debug("예외 무시: %s", _e)

                result[ticker] = {
                    "name": name, "price": round(price, 2),
                    "ret_1d":  round(ret_1d, 2),
                    "ret_5d":  round(float(c.pct_change(5).iloc[-1])  * 100, 2),
                    "ret_20d": round(float(c.pct_change(20).iloc[-1]) * 100, 2),
                }
            except Exception as e:
                print(f"    ⚠️ {ticker}: {e}")
        print(f"  ✅ 매크로 {len(result)}개")
        return result

    def fetch_watchlist(self, mstat: Dict) -> pd.DataFrame:
        intraday = mstat["intraday"]
        print(f"  📡 종목 수집 {'[실시간]' if intraday else '[일봉]'} ...")
        self._load_ref_rets()
        spy_r5 = DataFetcher._ref_cache.get("SPY", 0.0)

        def _one(tk):
            try:
                raw_d = self._get_price_data(tk)
                if raw_d is None or raw_d.empty: return None
                if "Close" not in raw_d.columns: return None
                cd = pd.to_numeric(raw_d["Close"],  errors="coerce").dropna()
                vd = pd.to_numeric(raw_d["Volume"], errors="coerce").dropna()
                hd = pd.to_numeric(raw_d["High"],   errors="coerce").dropna()
                ld = pd.to_numeric(raw_d["Low"],    errors="coerce").dropna()
                if isinstance(cd, pd.DataFrame): cd = cd.iloc[:,0].dropna()
                if isinstance(vd, pd.DataFrame): vd = vd.iloc[:,0].dropna()
                if isinstance(hd, pd.DataFrame): hd = hd.iloc[:,0].dropna()
                if isinstance(ld, pd.DataFrame): ld = ld.iloc[:,0].dropna()
                idx = cd.index
                vd = vd.reindex(idx)
                hd = hd.reindex(idx)
                ld = ld.reindex(idx)
                if len(cd) < 20: return None
                avg_vol_20 = float(vd.rolling(20).mean().iloc[-1]) if len(vd) > 20 else float(vd.mean())
                avg_vol_1d = float(vd.iloc[-2]) if len(vd) > 1 else avg_vol_20
                if intraday:
                    p,r1,vr20,vr1 = self._intraday_price_ret_vol(tk,avg_vol_20,mstat,avg_vol_1d)
                    price         = p   if p   is not None else float(cd.iloc[-1])
                    vol_ratio_20d = vr20 if vr20 is not None else float(vd.iloc[-1]/max(avg_vol_20,1))
                    vol_ratio_1d  = vr1  if vr1  is not None else float(vd.iloc[-1]/max(avg_vol_1d,1))
                else:
                    rt = self._get_realtime_price(tk)
                    price         = rt if rt is not None else float(cd.iloc[-1])
                    vol_ratio_20d = float(vd.iloc[-1]/max(avg_vol_20,1))
                    vol_ratio_1d  = float(vd.iloc[-1]/max(avg_vol_1d,1))
                _rg = mstat.get("regime","sideways")
                ss  = self._sse.calc(cd, vd, hd, ld, {"spy": spy_r5}, regime=_rg)
                fs  = ss.get("factor", {})
                return {"ticker": tk, "price": round(price, 4),
                        "ret_1d": ss["ret_1d"], "ret_5d": ss["ret_5d"],
                        "ret_20d": ss["ret_20d"],
                        "vol_ratio": round(vol_ratio_20d,2), "vol_ratio_1d": round(vol_ratio_1d,2),
                        "smart_score": ss["total"], "grade": ss["grade"], "badge": ss["badge"],
                        "factor_score": round(float(fs.get("total",0)),1),
                        "factor_grade": str(fs.get("grade","—")),
                        "factor_badge": str(fs.get("badge","—")),
                        "factor_regime": str(fs.get("regime",_rg)),
                        "raw": ss.get("raw",{})}
            except Exception as _e:
                logger.debug("종목 처리 오류: %s", _e)
                return None

        rows = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            # 484종목: 20워커 (I/O bound — yfinance 캐시 hit율 높아 실제 부하 낮음)
            for r in pool.map(_one, WATCHLIST):
                if r: rows.append(r)

        df = pd.DataFrame(rows)
        if not df.empty:
            rows_wl = df.to_dict("records")
            rows_wl = self._apply_cross_sectional(rows_wl)
            df = pd.DataFrame(rows_wl)
            df = df.sort_values("cs_score", ascending=False).reset_index(drop=True)
        print(f"  ✅ 종목 {len(df)}개")
        return df


    @staticmethod
    def _apply_cross_sectional(rows: list) -> list:
        """
        크로스섹셔널 점수 변환.
        전체 종목의 raw 신호를 percentile(0~100)로 변환 후 가중 합산.
        → 항상 균등 분포 보장. 절대 점수의 뭉침 현상 제거.

        단기 cs_score (v5 재구성 — IC-weighting 백테스트 검증):
          vol_z  : 거래량 서지           IC 60d +0.022  Easley & O'Hara(1987)
          bb_z   : BB압축 역수           IC 60d +0.010  직교신호
          rs     : RS vs SPY 5일        Levy(1967) 상대강도 관성
          accel  : 가격 가속도(5d-20d/4) Da et al.(2014) Frog-in-Pan
          제거: obv_z(IC 4회 연속 음수), div(IC≈0 vol_z중복)

        장기 cf_score (v5 재구성):
          mom    : 6M 가속도+Calmar      IC 60d +0.045  Jegadeesh & Titman(1993)
          bab    : Vol-adj 샤프형        IC 60d +0.032  Frazzini & Pedersen(2014)
          w52    : 52주 고점 근접도       George & Hwang(2004) AFA최우수논문
          cons   : 추세 일관성(상승일%)   Da et al.(2014) Frog-in-Pan
          제거: obv_lt(IC 4회 연속 음수), qual(IC≈0)

        cs_combo: cs(단기) 40% + cf(장기) 60%
                  — mom IC(+0.045)+bab(+0.032) >> vol_z(+0.022) → cf 비중 우위
        """
        n = len(rows)
        if n < 2:
            for r in rows:
                sc = float(r.get("smart_score", 50))
                r["cs_score"] = sc; r["cf_score"] = sc; r["cs_combo"] = sc
                r["cs_grade"], r["cs_badge"] = _cs_grade(sc)
                r["cf_grade"], r["cf_badge"] = _cs_grade(sc)
            return rows

        import numpy as np

        def pct_rank(vals):
            arr = np.array(vals, dtype=float)
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                return np.full(n, 50.0)
            out = np.full(n, 50.0)
            idx = np.where(valid)[0]
            order = np.argsort(arr[valid])
            pcts  = np.linspace(0, 100, valid.sum())
            for rank_i, orig_i in enumerate(idx[order]):
                out[orig_i] = pcts[rank_i]
            return out

        # v5 신호 재구성 — IC 음수/중복 제거, 학술 검증 신호로 대체
        raw_st = ["vol_z", "bb_z", "rs",  "accel"]
        raw_lt = ["mom",   "bab",  "w52", "cons"]
        # 단기: vol_z 최강(IC+0.022) 40%, bb_z 직교성 유지 20%, rs/accel 균등 20%
        w_st   = [0.40, 0.20, 0.20, 0.20]
        # 장기: mom 최강(IC+0.045) 40%, bab 2위(IC+0.032) 30%, w52/cons 신규 15%씩
        w_lt   = [0.40, 0.30, 0.15, 0.15]

        pct_st = {k: pct_rank([r.get("raw",{}).get(k,0) for r in rows]) for k in raw_st}
        pct_lt = {k: pct_rank([r.get("raw",{}).get(k,0) for r in rows]) for k in raw_lt}

        for i, r in enumerate(rows):
            cs    = sum(pct_st[k][i]*w for k,w in zip(raw_st,w_st))
            cf    = sum(pct_lt[k][i]*w for k,w in zip(raw_lt,w_lt))
            combo = cs*0.40 + cf*0.60   # cf(장기) 우위 — mom/bab IC 강함
            r["cs_score"] = round(cs, 1)
            r["cf_score"] = round(cf, 1)
            r["cs_combo"] = round(combo, 1)
            r["cs_grade"], r["cs_badge"] = _cs_grade(cs)
            r["cf_grade"], r["cf_badge"] = _cs_grade(cf)
        return rows

    def fetch_sector_detail(self, etf: str, mstat: Dict) -> List[Dict]:
        """
        섹터 종목 상세 — 병렬 수집 + 2단계 캐시.
        
        속도 전략:
          - 일봉 OHLCV : 30분 캐시 (_get_price_data)
          - 옵션/공매도/어닝 : 3시간 캐시 (_get_slow_data)
          - 종목 처리 : ThreadPoolExecutor(20) 병렬 실행
          
        결과: 순차 ~40분 → 첫 실행 ~90초, 이후 갱신 ~15초
        """
        tickers = SECTOR_STOCKS.get(etf, [])
        if not tickers:
            return []

        self._load_ref_rets()
        spy_r5 = DataFetcher._ref_cache.get("SPY",  0.0)
        sec_r5 = DataFetcher._ref_cache.get(etf, spy_r5)

        # ── 캐시 히트율 확인 ─────────────────────────────────
        now = time.time()
        price_hits = sum(1 for tk in tickers
                         if tk in DataFetcher._price_cache
                         and (now - DataFetcher._price_cache[tk]["ts"]) < self._PRICE_TTL)
        slow_hits  = sum(1 for tk in tickers
                         if tk in DataFetcher._slow_cache
                         and (now - DataFetcher._slow_cache[tk]["ts"]) < self._SLOW_TTL)
        print(f"  📡 {etf} {len(tickers)}개 "
              f"[캐시 가격:{price_hits}/{len(tickers)} "
              f"슬로우:{slow_hits}/{len(tickers)}]")

        def _process_one(tk: str):   # 반환: dict 또는 None
            """종목 1개 처리 — 스레드 내부에서 실행."""
            try:
                raw = self._get_price_data(tk)
                if raw is None or raw.empty:
                    return None
                if "Close" not in raw.columns:
                    return None

                # 명시적 Series 추출 — squeeze()는 MultiIndex 잔재 시 오작동 가능
                c  = pd.to_numeric(raw["Close"],  errors="coerce").dropna()
                v  = pd.to_numeric(raw["Volume"], errors="coerce").dropna()
                h  = pd.to_numeric(raw["High"],   errors="coerce").dropna()
                lo = pd.to_numeric(raw["Low"],    errors="coerce").dropna()
                # DataFrame으로 왔으면 첫 컬럼만 사용
                if isinstance(c,  pd.DataFrame): c  = c.iloc[:,0].dropna()
                if isinstance(v,  pd.DataFrame): v  = v.iloc[:,0].dropna()
                if isinstance(h,  pd.DataFrame): h  = h.iloc[:,0].dropna()
                if isinstance(lo, pd.DataFrame): lo = lo.iloc[:,0].dropna()
                # 인덱스를 c 기준으로 정렬 통일
                idx = c.index
                v  = v.reindex(idx)
                h  = h.reindex(idx)
                lo = lo.reindex(idx)

                if len(c) < 20: return None

                avg_vol_20 = float(v.rolling(20).mean().iloc[-1]) if len(v) > 20 else float(v.mean())
                avg_vol_1d = float(v.iloc[-2]) if len(v) > 1 else avg_vol_20
                price = float(c.iloc[-1])

                # ATR(14)
                tr      = pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
                atr14   = float(tr.rolling(14).mean().iloc[-1])
                atr_pct = round(atr14 / price * 100, 2) if price > 0 else 0.0

                vol_ratio_20d = float(v.iloc[-1] / max(avg_vol_20, 1))
                vol_ratio_1d  = float(v.iloc[-1] / max(avg_vol_1d,  1))
                vol_ratio = vol_ratio_20d
                vol_std_val = c.pct_change().rolling(20).std().iloc[-1]
                vol_std   = float(vol_std_val * 100) if pd.notna(vol_std_val) else 0.0
                vol_grade = ("🟢 저" if vol_std < 1.5 else "🟡 중" if vol_std < 3.0
                             else "🟠 고" if vol_std < 5.0 else "🔴 극고")

                if mstat["intraday"]:
                    p,r1,vr20,vr1 = self._intraday_price_ret_vol(tk,avg_vol_20,mstat,avg_vol_1d)
                    if p    is not None: price         = p
                    if vr20 is not None: vol_ratio_20d = vr20; vol_ratio = vr20
                    if vr1  is not None: vol_ratio_1d  = vr1
                else:
                    rt = self._get_realtime_price(tk)
                    if rt is not None: price = rt

                ose = self._get_slow_data(tk)
                _rg = mstat.get("regime","sideways")
                ss  = self._sse.calc(c,v,h,lo,{"spy":spy_r5,"sector":sec_r5},opt_data=ose,regime=_rg)
                d   = ss["details"]
                fs  = ss.get("factor",{})

                sp = ose.get("short_pct") or 0
                # 숏스퀴즈 판단 기준 (3단계):
                # 1) 공매도 잔고 기준: sp >= 15% + 거래량 급증(1.5x) → 숏커버링 시작 가능
                # 2) 고공매도 단순 기준: sp >= 20% → 언제든 스퀴즈 잠재력
                # 3) 강화 조건: accel(가격 가속도) 양수 추가 → "진행 중" vs "잠재" 구분
                #    raw에서 accel = ret_5d - (ret_20d/4) → 최근 1주가 가속 중이면 양수
                _accel_val = float(ss.get("raw", {}).get("accel", 0))
                _squeeze_potential = bool((sp >= 15 and vol_ratio >= 1.5) or (sp >= 20))
                _squeeze_active    = bool(_squeeze_potential and _accel_val > 0)  # 가속 중 = 진행 중
                short_squeeze = _squeeze_potential  # HTML 배지 기준은 잠재력 포함
                squeeze_active = _squeeze_active    # 진행 중 여부 별도 저장

                # ── A등급 모듈 연동 ─────────────────────────────
                annual_vol = vol_std * (252 ** 0.5)
                avg_vol_num = avg_vol_20 if avg_vol_20 > 0 else 1.0
                cost_info = _cost_model.estimate_cost(
                    price=price,
                    avg_daily_volume=avg_vol_num,
                    order_value=_risk_manager.max_position_value,
                    volatility_pct=max(vol_std, 0.5),
                )
                risk_info = _risk_manager.atr_position_size(
                    price=price,
                    atr=atr14,
                    smart_score=float(ss["total"]),
                    cost_total_pct=cost_info["total_pct"],  # 종목별 실비용 전달
                )
                _audit_logger.log_signal(
                    ticker=tk,
                    score_result=ss,
                    regime=_rg,
                    price=price,
                    atr_pct=atr_pct,
                )

                return {
                    "ticker":      tk,
                    "price":       round(float(price), 2),
                    "ret_1d":      float(ss["ret_1d"]),
                    "ret_5d":      float(ss["ret_5d"]),
                    "ret_20d":     float(ss["ret_20d"]),
                    "vol_ratio":   round(float(vol_ratio), 2),
                    "atr_pct":     float(atr_pct),
                    "vol_std":     round(float(vol_std), 2),
                    "vol_grade":   str(vol_grade),
                    "smart_score": float(ss["total"]),
                    "grade":       str(ss["grade"]),
                    "badge":       str(ss["badge"]),
                    # 실제 details 키 기반 sc_ 매핑
                    "sc_pos":   float(d.get("매집탐지",  {}).get("score", 0)),  # 25점
                    "sc_obv":   float(d.get("OBV",       {}).get("score", 0)),  # 20점
                    "sc_vol":   float(d.get("거래량구조", {}).get("score", 0)),  # 18점
                    "sc_opt":   float(d.get("옵션신호",  {}).get("score", 0)),  # 15점
                    "sc_accel": float(d.get("모멘텀가속",{}).get("score", 0)),  # 12점
                    "sc_rs":    float(d.get("RS",        {}).get("score", 0)),  # 10점
                    "sc_bb":    float(d.get("BB압축",    {}).get("score", 0)),  #  8점
                    "sc_trend": float(d.get("추세일관성",{}).get("score", 0)),  #  7점
                    "accum_tags": list(d.get("매집탐지", {}).get("tags",   [])),
                    "up_ratio":   float(d.get("추세일관성",{}).get("up_ratio", 0)),
                    "streak":     int(d.get("추세일관성",  {}).get("streak",   0)),
                    "vs_spy":     float(d.get("RS",       {}).get("vs_spy",   0)),
                    "reversal":   bool(d.get("모멘텀가속",{}).get("reversal", False)),
                    "factor_score":  round(float(fs.get("total",0)),1),
                    "factor_grade":  str(fs.get("grade","—")),
                    "factor_badge":  str(fs.get("badge","—")),
                    "factor_regime": str(fs.get("regime",_rg)),
                    "pc_ratio":    float(ose["pc_ratio"])    if ose.get("pc_ratio")    is not None else None,
                    "iv_pct":      float(ose["iv_pct"])      if ose.get("iv_pct")      is not None else None,
                    "short_pct":   float(ose["short_pct"])   if ose.get("short_pct")   is not None else None,
                    "short_ratio": float(ose["short_ratio"]) if ose.get("short_ratio") is not None else None,
                    "short_squeeze": short_squeeze,
                    "squeeze_active": squeeze_active,
                    "earn_date":   str(ose["earn_date"])     if ose.get("earn_date")   is not None else None,
                    "earn_days":   int(ose["earn_days"])     if ose.get("earn_days")   is not None else None,
                    "vol_ratio_1d": round(float(vol_ratio_1d), 2),
                    "raw": ss.get("raw",{}),
                    # ── A등급 추가 필드 ──────────────────────────
                    "annual_vol":         round(float(annual_vol), 2),
                    "cost_roundtrip_pct": cost_info["roundtrip_pct"],
                    "cost_class":         cost_info["cost_class"],
                    "cost_breakeven":     cost_info["breakeven_move"],
                    "risk_shares":        risk_info["shares"],
                    "risk_position_pct":  risk_info["position_pct"],
                    "risk_stop_price":    risk_info["stop_price"],
                    "risk_stop_pct":      risk_info["stop_pct"],
                    "risk_valid":         risk_info["valid"],
                }
            except Exception as _e:
                print(f"    ❌ {tk}: {type(_e).__name__}: {_e}")
                return None

        # ── 병렬 실행 (워커 20개) ─────────────────────────────
        rows = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_process_one, tk): tk for tk in tickers}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    rows.append(result)

        rows = self._apply_cross_sectional(rows)
        rows.sort(key=lambda x: x.get("cs_score", x.get("smart_score",0)), reverse=True)
        return rows


# ──────────────────────────────────────────────────────────────
#  시장 분석 엔진
# ──────────────────────────────────────────────────────────────
class MarketAnalyzer:
    def fear_greed(self, macro: Dict) -> Tuple[float, str, str]:
        """
        공포/탐욕 지수 계산 (0~100)
        VIX↓ + HYG↑ + TLT↓ + GLD안정 → 탐욕
        """
        score = 50.0
        signals = []

        vix = macro.get("^VIX", {})
        if vix:
            v = vix.get("price", 20)
            if v < 15:   score += 15; signals.append(f"VIX {v:.0f} (극도 안정)")
            elif v < 20: score += 8;  signals.append(f"VIX {v:.0f} (안정)")
            elif v < 25: score -= 5;  signals.append(f"VIX {v:.0f} (경계)")
            elif v < 30: score -= 15; signals.append(f"VIX {v:.0f} (불안)")
            else:        score -= 25; signals.append(f"VIX {v:.0f} (공포)")

        hyg = macro.get("HYG", {})
        if hyg:
            r = hyg.get("ret_5d", 0)
            if r > 1:    score += 10; signals.append(f"HYG +{r:.1f}% (신용위험↓)")
            elif r > 0:  score += 4
            elif r > -1: score -= 5
            else:        score -= 15; signals.append(f"HYG {r:.1f}% (신용위험↑)")

        tlt = macro.get("TLT", {})
        if tlt:
            r = tlt.get("ret_5d", 0)
            # TLT 하락 = 금리 상승 = 위험자산 선호
            if r < -1:   score += 8;  signals.append(f"TLT {r:.1f}% (금리↑·위험선호)")
            elif r > 2:  score -= 10; signals.append(f"TLT +{r:.1f}% (안전자산 도피)")

        dxy = macro.get("DX-Y.NYB", {})
        if dxy:
            r = dxy.get("ret_5d", 0)
            if r > 1:    score -= 8;  signals.append(f"DXY +{r:.1f}% (달러 강세·위험↓)")
            elif r < -1: score += 8;  signals.append(f"DXY {r:.1f}% (달러 약세·위험↑)")

        score = max(0, min(100, score))
        if score >= 75:   label, color = "극도 탐욕", "var(--green)"
        elif score >= 55: label, color = "탐욕",      "#88ff44"
        elif score >= 45: label, color = "중립",      "#ffdd00"
        elif score >= 25: label, color = "공포",      "#ff8844"
        else:             label, color = "극도 공포", "#ff3344"

        return score, label, color

    def sector_rotation_phase(self, sector_df: pd.DataFrame) -> str:
        """
        섹터 로테이션 국면 감지
        경기 사이클: 회복→확장→과열→침체
        """
        if sector_df.empty:
            return "데이터 부족"
        top3 = sector_df.head(3)["etf"].tolist()
        bot3 = sector_df.tail(3)["etf"].tolist()

        # 경기 확장: XLK·XLY·XLF 강세
        expansion = sum(1 for e in ["XLK","XLY","XLF"] if e in top3)
        # 과열: XLE·XLB·XLI 강세
        overheating = sum(1 for e in ["XLE","XLB","XLI"] if e in top3)
        # 침체: XLP·XLU·XLV 강세 (방어주)
        recession = sum(1 for e in ["XLP","XLU","XLV"] if e in top3)
        # 회복: XLF·XLI 강세, XLU 약세
        recovery = sum(1 for e in ["XLF","XLI"] if e in top3)

        if expansion >= 2:   return "📈 경기 확장 — 성장주·기술주 유리"
        if overheating >= 2: return "🔥 경기 과열 — 원자재·에너지 유리"
        if recession >= 2:   return "🛡️ 경기 침체 우려 — 방어주로 자금 이동"
        if recovery >= 2:    return "🌱 경기 회복 — 금융·산업주 선행"
        return "↔️ 혼조 — 방향성 불명확"

    def big_picture(self, sector_df: pd.DataFrame, macro: Dict,
                    fg_score: float, fg_label: str, rotation: str) -> Dict:
        top_sector  = sector_df.iloc[0] if not sector_df.empty else None
        bot_sector  = sector_df.iloc[-1] if not sector_df.empty else None
        vix_val     = macro.get("^VIX", {}).get("price", 20)
        hyg_r5      = macro.get("HYG", {}).get("ret_5d", 0)

        lines = []
        strategy = "관망"

        if fg_score >= 70:
            lines.append("시장 전반에 탐욕 심리 — 과열 주의, 차익 실현 고려")
            strategy = "⚠️ 신규 매수 신중 / 기존 포지션 익절 준비"
        elif fg_score >= 55:
            lines.append("완만한 탐욕 — 추세 추종 유효")
            strategy = "📈 모멘텀 상위 섹터 종목 집중"
        elif fg_score >= 45:
            lines.append("중립 국면 — 선택적 매매")
            strategy = "⚖️ 섹터 로테이션 따라가되 포지션 작게"
        elif fg_score >= 25:
            lines.append("공포 심리 우세 — 반등 기회 탐색")
            strategy = "⚡ 과매도 구간 단기 반등 포착 (소규모)"
        else:
            lines.append("극도 공포 — 바닥 근접 가능성, 분할 매수 검토")
            strategy = "🎯 분할 매수 / VIX 하락 확인 후 진입"

        if top_sector is not None:
            lines.append(f"자금 집중: {top_sector['emoji']} {top_sector['name']} "
                         f"({top_sector['ret_5d']:+.1f}% 5일)")
        if bot_sector is not None:
            lines.append(f"자금 이탈: {bot_sector['emoji']} {bot_sector['name']} "
                         f"({bot_sector['ret_5d']:+.1f}% 5일)")

        lines.append(rotation)

        return {"summary": " | ".join(lines[:2]), "strategy": strategy,
                "details": lines, "vix": vix_val}


# ──────────────────────────────────────────────────────────────
#  HTML 생성
# ──────────────────────────────────────────────────────────────
def build_html(sector_df, macro, watchlist_df, fg_score, fg_label, fg_color,
               rotation, big, mstat, sector_detail_data: Dict,
               radar_stocks: List, squeeze_stocks: List,
               earn_upcoming: List, inst_data: Dict,
               stress_results: Dict = None,
               corr_result: Dict = None,
               concentration: Dict = None) -> str:

    now_str  = datetime.now().strftime("%Y년 %m월 %d일  %H:%M")
    # now_str은 HTML 생성 시각 표시 (JS 실시간 시계로 보완)
    intraday = mstat["intraday"]
    if intraday:
        status_badge = f'<span style="font-family:var(--mono);font-size:.72rem;background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.4);border-radius:5px;padding:2px 9px;color:var(--green);margin-left:10px">{mstat["label"]} · 실시간 5분봉 · {REFRESH_SEC//60}분 자동갱신</span>'
    else:
        status_badge = f'<span style="font-family:var(--mono);font-size:.72rem;background:rgba(255,100,0,.1);border:1px solid rgba(255,100,0,.3);border-radius:5px;padding:2px 9px;color:#ff8844;margin-left:10px">{mstat["label"]} · 전일 종가 기준</span>'

    # 섹터 상세 데이터를 JSON으로 HTML에 embed
    # numpy bool_ / int64 / float64, pandas Timestamp 등 직렬화 처리
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.bool_):    return bool(obj)
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            if hasattr(obj, 'isoformat'):    return obj.isoformat()
            return super().default(obj)

    try:
        detail_json = json.dumps(sector_detail_data, ensure_ascii=False, cls=_SafeEncoder)
    except Exception as e:
        print(f"  ⚠️  detail_json 직렬화 실패: {e}")
        detail_json = "{}"

    # ── [A등급] 스트레스 테스트 HTML ─────────────────────────
    stress_html = _stress_engine.get_stress_html(stress_results or {})

    # ── [A등급] 상관분석 HTML ─────────────────────────────────
    corr_html = _portfolio_ctor.get_corr_html(corr_result or {})

    # ── [A등급] 포트폴리오 집중도 HTML ───────────────────────
    conc = concentration or {}
    hhi_val  = conc.get("hhi", 0)
    hhi_c    = "var(--green)" if hhi_val < 1000 else "var(--yellow)" if hhi_val < 2000 else "#ff4466"
    eff_n    = conc.get("effective_n", 0)
    conc_warn = conc.get("warning", "")
    sec_w    = conc.get("sector_weights", {})
    sec_rows = "".join(
        f'<div class="conc-row">'
        f'<span class="conc-sec">{s}</span>'
        f'<div class="conc-bar-wrap"><div class="conc-bar" style="width:{min(w,100):.0f}%;'
        f'background:{"#ff4466" if w>30 else "var(--yellow)" if w>20 else "var(--green)"}"></div></div>'
        f'<span class="conc-val" style="color:{"#ff4466" if w>30 else "var(--yellow)" if w>20 else "var(--green)"}">{w:.1f}%</span>'
        f'</div>'
        for s, w in list(sec_w.items())[:8]
    )
    concentration_html = f"""
    <div class="conc-summary">
      <div style="display:flex;gap:20px;margin-bottom:10px;flex-wrap:wrap">
        <div><span style="color:#8b949e">HHI 집중도: </span><span style="color:{hhi_c};font-weight:700">{hhi_val:.0f}</span><span style="color:var(--text-dim);font-size:.75rem"> (1800 이하 = 분산)</span></div>
        <div><span style="color:#8b949e">유효 섹터수: </span><span style="color:#c9d1d9;font-weight:700">{eff_n:.1f}개</span></div>
        {f'<div style="color:#ff8844">{conc_warn}</div>' if conc_warn else ''}
      </div>
      {sec_rows}
    </div>"""

    # ── [A등급] 리스크 사이징 HTML (레이더 상위 10종목) ────────
    risk_cards_html = ""
    for s in radar_stocks[:10]:
        if not s.get("risk_valid"):
            continue
        sc   = s["smart_score"]
        sc_c = "var(--green)" if sc >= 70 else "var(--yellow)" if sc >= 55 else "#ff8844"
        cost_c = {"LOW":"var(--green)","MEDIUM":"var(--yellow)","HIGH":"#ff8844","VERY_HIGH":"#ff4466"}.get(s.get("cost_class","MEDIUM"),"#888")
        stop_p = s.get("risk_stop_price", 0)
        stop_c = "#ff4466"
        risk_cards_html += f"""
        <div class="risk-card">
          <div class="risk-header">
            <span class="risk-ticker">{s['ticker']}</span>
            <span class="risk-score" style="color:{sc_c}">{sc:.0f}점</span>
            <span style="font-size:.7rem;color:{cost_c};border:1px solid {cost_c};border-radius:3px;padding:1px 5px">비용 {s.get('cost_roundtrip_pct',0):.2f}%왕복</span>
          </div>
          <div class="risk-body">
            <div class="risk-row"><span class="risk-lbl">ATR 사이징</span><span class="risk-val" style="color:var(--green)">{s.get('risk_shares',0)}주 ({s.get('risk_position_pct',0):.1f}%)</span></div>
            <div class="risk-row"><span class="risk-lbl">손절가 (ATR×2)</span><span class="risk-val" style="color:{stop_c}">${stop_p:,.2f} (-{s.get('risk_stop_pct',0):.1f}%)</span></div>
            <div class="risk-row"><span class="risk-lbl">손익분기 이동</span><span class="risk-val" style="color:#8b949e">{s.get('cost_breakeven',0)*100:.2f}% 이상</span></div>
            <div class="risk-row"><span class="risk-lbl">연변동성</span><span class="risk-val" style="color:#bc8cff">{s.get('annual_vol',0):.1f}%</span></div>
          </div>
        </div>"""

    if not risk_cards_html:
        risk_cards_html = '<div class="radar-empty">리스크 사이징 가능한 종목 없음</div>'

    # ── [A등급] 감사 로그 HTML ────────────────────────────────
    audit_html = _audit_logger.get_summary_html()

    # ── 숏스퀴즈 후보 섹션 ───────────────────────────────────
    if not squeeze_stocks:
        squeeze_html = '<div class="radar-empty">현재 숏스퀴즈 후보 종목 없음</div>'
    else:
        squeeze_html = ""
        for s in squeeze_stocks[:15]:  # 최대 15개
            sp   = s.get("short_pct")  or 0
            sr   = s.get("short_ratio") or 0
            r5c  = "var(--green)" if s["ret_5d"] >= 0 else "#ff4466"
            sc   = s["smart_score"]
            sc_c = "var(--green)" if sc >= 55 else "#ffdd00" if sc >= 40 else "#ff8844"
            # 스퀴즈 강도: 공매도 비율 기반
            intensity = "🔴 극고위험" if sp >= 30 else "🟠 고위험" if sp >= 20 else "🟡 주의"
            # 진행 중 여부: accel(가격 가속도) 양수 = 숏커버링 이미 시작
            # squeeze_active = short_squeeze AND accel > 0
            _active_badge = ""
            if s.get("squeeze_active"):
                _active_badge = '<span style="font-size:.62rem;color:#ff6b6b;border:1px solid #ff6b6b;border-radius:3px;padding:0 3px;margin-left:3px">▲진행중</span>'
            squeeze_html += f"""
            <div class="squeeze-row">
              <span class="sq-ticker">{s['ticker']}</span>
              <span class="sq-sector">{s['etf']}</span>
              <span class="sq-short" title="공매도 잔고 비율">{sp:.1f}%</span>
              <span class="sq-days" title="숏커버 소요 일수">{sr:.1f}일</span>
              <span style="color:{r5c};font-family:var(--mono);font-size:.78rem">{s['ret_5d']:+.1f}%</span>
              <span class="sq-intensity">{intensity}{_active_badge}</span>
              <span style="color:{sc_c};font-family:var(--mono);font-size:.82rem;font-weight:700">{sc:.0f}점</span>
            </div>"""

    # ── 어닝 캘린더 섹션 ─────────────────────────────────────
    if not earn_upcoming:
        earn_html = '<div class="radar-empty">60일 이내 실적 발표 예정 종목 없음</div>'
    else:
        earn_html = ""
        for e in earn_upcoming[:20]:  # 최대 20개
            days  = e["earn_days"]
            r5c   = "var(--green)" if e["ret_5d"] >= 0 else "#ff4466"
            sc    = e["smart_score"]
            sc_c  = "var(--green)" if sc >= 55 else "#ffdd00" if sc >= 40 else "#ff8844"
            if days == 0:   urgency = "🔴 오늘"
            elif days <= 3: urgency = f"🟠 {days}일 후"
            elif days <= 7: urgency = f"🟡 {days}일 후"
            else:           urgency = f"⚪ {days}일 후"
            earn_html += f"""
            <div class="earn-row">
              <span class="eq-date">{urgency}</span>
              <span class="eq-ticker">{e['ticker']}</span>
              <span class="eq-sector">{e['etf']}</span>
              <span class="eq-earndate" style="font-family:var(--mono);font-size:.72rem;color:var(--text-dim)">{e['earn_date']}</span>
              <span style="color:{r5c};font-family:var(--mono);font-size:.78rem">{e['ret_5d']:+.1f}%</span>
              <span style="color:{sc_c};font-family:var(--mono);font-size:.82rem;font-weight:700">{sc:.0f}점</span>
            </div>"""

    # ── 매집 레이더 카드 (숏스퀴즈·어닝 배지 추가) ───────────
    radar_rows_html = ""
    tag_colors = {"횡보매집": "var(--accent)", "바닥신호": "var(--yellow)", "OBV선행": "var(--green)"}

    if not radar_stocks:
        radar_rows_html = '<div class="radar-empty">현재 2개 이상 매집 태그를 가진 종목이 없습니다</div>'
    else:
        for s in radar_stocks:
            tags     = s.get("accum_tags", [])
            sc       = s["smart_score"]
            r1c      = "var(--green)" if s["ret_1d"]  >= 0 else "#ff4466"
            r5c      = "var(--green)" if s["ret_5d"]  >= 0 else "#ff4466"
            sc_color = ("var(--green)" if sc >= 75 else "#88ff44" if sc >= 55
                        else "#ffdd00" if sc >= 40 else "#ff8844")
            tag_html = "".join(
                f'<span class="radar-tag" style="border-color:{tag_colors.get(t,"#888")};'
                f'color:{tag_colors.get(t,"#888")}">{t}</span>'
                for t in tags
            )
            all3 = len(tags) == 3
            border_style = "border-color:rgba(0,212,255,.7);box-shadow:0 0 18px rgba(0,212,255,.15)" if all3 \
                           else "border-color:rgba(255,215,0,.4)"
            crown = "👑 " if all3 else ""

            # 추가 배지: 숏스퀴즈 / 어닝 임박
            extra_badges = ""
            if s.get("short_squeeze"):
                sp = s.get("short_pct") or 0
                _sq_label = f"🩳 숏 {sp:.0f}% ▲" if s.get("squeeze_active") else f"🩳 숏 {sp:.0f}%"
                extra_badges += f'<span class="radar-tag" style="border-color:#ff6b6b;color:#ff6b6b">{_sq_label}</span>'
            if s.get("earn_days") is not None:
                ed = s["earn_days"]
                ec = "#ff4466" if ed <= 3 else "#ff8844" if ed <= 7 else "#aaaaaa"
                extra_badges += f'<span class="radar-tag" style="border-color:{ec};color:{ec}">📅 실적 {ed}일 후</span>'

            radar_rows_html += f"""
            <div class="radar-card" style="{border_style}"
                 onclick="openDetail('{s['etf']}', '{s['sector']}', '🎯')"
                 title="{s['etf']} 섹터 — 클릭하면 상세 보기">
              <div class="radar-card-top">
                <span class="radar-ticker">{crown}{s['ticker']}</span>
                <span class="radar-sector">{s['etf']}</span>
                <span class="radar-score" style="color:{sc_color}">{sc:.0f}<span style="font-size:.65rem;opacity:.6">/100</span></span>
              </div>
              <div class="radar-price">${s['price']:,.2f}
                <span style="color:{r1c};font-size:.78rem;margin-left:6px">{s['ret_1d']:+.1f}%</span>
                <span style="color:{r5c};font-size:.72rem;margin-left:4px">5일 {s['ret_5d']:+.1f}%</span>
              </div>
              <div class="radar-tags">{tag_html}{extra_badges}</div>
              <div class="radar-bar-wrap">
                <div class="radar-bar" style="width:{int(sc)}%;background:{sc_color}"></div>
              </div>
            </div>"""

    # ── 기관 데이터 HTML ──────────────────────────────────────
    # 내부자 매수
    congress_list  = inst_data.get("congress", [])
    congress_upd   = inst_data.get("congress_updated", "미수집")
    t13f_list      = inst_data.get("top13f", [])
    t13f_upd       = inst_data.get("top13f_updated", "미수집")
    inst_err       = inst_data.get("error")

    # 데이터 신선도 경고 문구
    def staleness_badge(days: int) -> str:
        if days <= 10:   return f'<span style="color:var(--green)">({days}일 전)</span>'
        elif days <= 30: return f'<span style="color:var(--yellow)">({days}일 전)</span>'
        else:            return f'<span style="color:#ff8844">({days}일 전 ⚠️ 오래된 데이터)</span>'

    # 내부자 매수 섹션
    if not congress_list:
        congress_html = '<div class="radar-empty">내부자 매수 데이터 없음 (API 연결 실패)</div>'
    else:
        congress_html = ""
        for t in congress_list[:15]:
            days  = t.get("days_ago", 999)
            party = t.get("party","")
            party_color = "#4488ff" if "D" in party else "#ff6644" if "R" in party else "#aaaaaa"
            party_tag   = f'<span style="color:{party_color};font-size:.65rem">{party}</span>'
            stale = staleness_badge(days)
            congress_html += f"""
            <div class="inst-row">
              <span class="inst-date">{t['date']}<br>{stale}</span>
              <span class="inst-ticker">{t['ticker']}</span>
              <span class="inst-type" style="color:var(--green)">● 매수</span>
              <span class="inst-rep">{t['rep'][:18]} {party_tag}</span>
              <span class="inst-amount" style="font-size:.72rem;color:#8899aa"
                    title="{t.get('amount_raw', t['amount'])}">{t['amount']}</span>
            </div>"""

    # 13F 기관 보유 현황
    last_13f_label = _last_13f_date()
    if not t13f_list:
        t13f_html = '<div class="radar-empty">기관 보유 데이터 없음</div>'
    else:
        t13f_html = ""
        for item in t13f_list:
            pct   = item["inst_pct"]
            pct_c = "var(--green)" if pct >= 80 else "var(--yellow)" if pct >= 60 else "#8899aa"
            bar_w = int(pct)
            t13f_html += f"""
            <div class="inst-row">
              <span class="inst-ticker">{item['ticker']}</span>
              <span style="color:{pct_c};font-family:var(--mono);font-weight:700">{pct:.1f}%</span>
              <span style="flex:1;padding:0 8px">
                <div style="height:4px;background:var(--border);border-radius:2px">
                  <div style="width:{bar_w}%;height:100%;background:{pct_c};border-radius:2px"></div>
                </div>
              </span>
              <span style="font-size:.68rem;color:var(--text-dim)">{item['as_of']}</span>
            </div>"""

    # ── 섹터 카드 — SPDR / 테마 그룹 구분 ──────────────────
    sector_cards = ""
    cur_group    = None
    for idx, row in sector_df.iterrows():
        g = row.get("group", "SPDR")
        if g != cur_group:
            cur_group = g
            label = "📊 SPDR 기본 섹터" if g == "SPDR" else "🎯 테마 ETF"
            sector_cards += f'<div class="group-label">{label}</div>'
        r1  = row["ret_1d"];  r1c  = "var(--green)" if r1  >= 0 else "#ff4466"
        r5  = row["ret_5d"];  r5c  = "var(--green)" if r5  >= 0 else "#ff4466"
        r20 = row["ret_20d"]; r20c = "var(--green)" if r20 >= 0 else "#ff4466"
        vr20 = row.get("vol_ratio", 1.0)
        vr1d = row.get("vol_ratio_1d", vr20)
        if vr20 >= 2.0 or vr1d >= 2.0:
            vol_badge = f'<span class="vol-spike">🔥 20D:{vr20:.1f}x / 1D:{vr1d:.1f}x</span>'
        elif vr20 >= 1.5 or vr1d >= 1.5:
            vol_badge = f'<span class="vol-high">⬆ 20D:{vr20:.1f}x / 1D:{vr1d:.1f}x</span>'
        else:
            vol_badge = f'<span style="color:#4a8080;font-size:.75rem">20D:{vr20:.1f}x / 1D:{vr1d:.1f}x</span>'

        # 모멘텀 바
        pct = max(0, min(100, (row["momentum"] + 15) / 30 * 100))
        bar_color = "var(--green)" if row["momentum"] >= 0 else "#ff4466"

        sector_cards += f"""
        <div class="sector-card {'sector-top' if idx < 3 else 'sector-bot' if idx >= len(sector_df)-3 else ''}" data-etf="{row['etf']}" onclick="openDetail('{row['etf']}', '{row['name']}', '{row['emoji']}')">
          <div class="sector-header">
            <span class="sector-emoji">{row['emoji']}</span>
            <span class="sector-name">{row['name']}</span>
            <span class="sector-etf">{row['etf']}</span>
          </div>
          <div class="sector-price">${row['price']:,.2f}</div>
          <div class="sector-rets">
            <span style="color:{r1c}">1일 {r1:+.1f}%</span>
            <span style="color:{r5c}">5일 {r5:+.1f}%</span>
            <span style="color:{r20c}">20일 {r20:+.1f}%</span>
          </div>
          {vol_badge}
          <div class="momentum-bar-wrap">
            <div class="momentum-bar" style="width:{pct:.0f}%;background:{bar_color}"></div>
          </div>
          <div class="sector-rs">RS vs SPY: {row['rs_spy']:.2f}</div>
          <div class="sector-hint">▸ 클릭하여 종목 상세 보기</div>
        </div>"""

    # ── 매크로 카드 ───────────────────────────────────────────
    macro_cards = ""
    macro_order = ["^VIX","HYG","TLT","GLD","DX-Y.NYB","^TNX"]
    macro_icons = {"^VIX":"😰","HYG":"💳","TLT":"🏛️","GLD":"🥇","DX-Y.NYB":"💵","^TNX":"📊"}
    for tk in macro_order:
        if tk not in macro:
            continue
        m   = macro[tk]
        r1  = m["ret_1d"]; r1c = "var(--green)" if r1 >= 0 else "#ff4466"
        r5  = m["ret_5d"]; r5c = "var(--green)" if r5 >= 0 else "#ff4466"
        ico = macro_icons.get(tk, "📈")
        macro_cards += f"""
        <div class="macro-card">
          <div class="macro-icon">{ico}</div>
          <div class="macro-name">{m['name']}</div>
          <div class="macro-price">{m['price']:,.2f}</div>
          <div class="macro-rets">
            <span style="color:{r1c}">1일 {r1:+.2f}%</span>
            <span style="color:{r5c}">5일 {r5:+.2f}%</span>
          </div>
        </div>"""

    # ── 스마트머니 Top10 — cs_combo 기준 정렬 ───────────────────
    # cs_combo = 크로스섹셔널 단기(60%) + 장기(40%) 종합 순위
    _wl_sorted = (watchlist_df
                  .assign(_sort_key=lambda df: df.get("cs_combo", df.get("cs_score", df["smart_score"])))
                  .sort_values("_sort_key", ascending=False)
                  .head(10))
    smart_rows = ""
    for i, (_, row) in enumerate(_wl_sorted.iterrows()):
        r1c  = "var(--green)" if row["ret_1d"]  >= 0 else "#ff4466"
        r5c  = "var(--green)" if row["ret_5d"]  >= 0 else "#ff4466"
        r20c = "var(--green)" if row["ret_20d"] >= 0 else "#ff4466"
        vr20 = row.get("vol_ratio", 1.0)
        vr1d = row.get("vol_ratio_1d", vr20)
        vol_tag = "🔥" if (vr20 >= 2.0 or vr1d >= 2.0) else "⬆" if (vr20 >= 1.5 or vr1d >= 1.5) else ""
        rank_color = ["var(--yellow)","#c0c0c0","#cd7f32"] + ["#8899aa"]*7
        # cs_combo 우선, 없으면 cs_score, 없으면 smart_score
        sc      = float(row.get("cs_combo", row.get("cs_score", row["smart_score"])))
        sc_abs  = float(row.get("smart_score", 0))
        # RS vs SPY (raw에서 추출)
        _raw    = row.get("raw", {}) if isinstance(row.get("raw"), dict) else {}
        rs_val  = float(_raw.get("rs",    row.get("vs_spy", 0)))   # RS vs SPY 5일
        ac_val  = float(_raw.get("accel", 0))                       # 가격 가속도
        rs_c    = "var(--green)" if rs_val >= 2 else "#88ff44" if rs_val >= 0 else "#ff8844" if rs_val >= -2 else "#ff4466"
        ac_c    = "var(--green)" if ac_val >= 1 else "#88ff44" if ac_val >= 0 else "#ff8844" if ac_val >= -1 else "#ff4466"
        sc_color = ("var(--green)" if sc >= 80 else "#88ff44" if sc >= 60
                    else "#ffdd00" if sc >= 40 else "#ff8844" if sc >= 20 else "var(--red)")
        sc_w  = int(sc)
        badge = row.get("cs_badge", row.get("badge", ""))
        grade = row.get("cs_grade", row.get("grade", ""))
        smart_rows += f"""
        <tr>
          <td><span class="rank" style="color:{rank_color[i]}">#{i+1}</span></td>
          <td class="ticker-cell">{row['ticker']} {vol_tag}</td>
          <td style="font-family:var(--mono)">${row['price']:,.2f}</td>
          <td style="color:{r1c};font-family:var(--mono)">{row['ret_1d']:+.1f}%</td>
          <td style="color:{r5c};font-family:var(--mono)">{row['ret_5d']:+.1f}%</td>
          <td style="color:{r20c};font-family:var(--mono)">{row['ret_20d']:+.1f}%</td>
          <td class="vol-cell" style="font-family:var(--mono);line-height:1.6">
            <span style="color:{'#ff8844' if vr20>=2 else 'var(--yellow)' if vr20>=1.5 else '#aabbcc'}">{vr20:.1f}x</span>
            <span style="font-size:.65rem;color:#8899aa;display:block">전일 {vr1d:.1f}x</span>
          </td>
          <td style="font-family:var(--mono);text-align:center">
            <span style="color:{rs_c};font-weight:600">{rs_val:+.1f}%</span>
            <span style="font-size:.62rem;color:var(--text-dim);display:block">vs SPY</span>
          </td>
          <td style="font-family:var(--mono);text-align:center">
            <span style="color:{ac_c};font-weight:600">{ac_val:+.1f}</span>
            <span style="font-size:.62rem;color:var(--text-dim);display:block">가속</span>
          </td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="width:60px;height:5px;background:var(--border);border-radius:3px">
                <div style="width:{sc_w}%;height:100%;background:{sc_color};border-radius:3px"></div>
              </div>
              <span style="color:{sc_color};font-weight:700;font-family:var(--mono)">{sc:.0f}</span>
              <span style="font-size:.62rem;color:var(--text-dim);margin-left:2px">절대:{sc_abs:.0f}</span>
              <span style="font-size:.7rem;display:block;margin-top:1px">{badge} {grade}</span>
            </div>
          </td>
        </tr>"""

    # ── 공포탐욕 게이지 ───────────────────────────────────────
    gauge_angle = int(fg_score * 1.8 - 90)   # -90 ~ 90도

    # ── Big Picture 디테일 ────────────────────────────────────
    detail_items = "".join(f"<li>{d}</li>" for d in big["details"])

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZEUS Market Flow — {now_str}</title>
<meta http-equiv="refresh" content="{REFRESH_SEC}">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+KR:wght@300;400;700&display=swap" rel="stylesheet">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  :root {{
    --bg:       #0a0b0e;
    --bg2:      #12141a;
    --bg3:      #16181f;
    --border:   #1f2333;
    --border2:  #252a3a;
    --accent:   #5b6fff;
    --accent2:  #7c3aed;
    --green:    #22c55e;
    --red:      #ef4444;
    --yellow:   #f59e0b;
    --text:     #e8eaf0;
    --text-dim: #555d7a;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Pretendard', 'Segoe UI', system-ui, sans-serif;
    --display:  'Pretendard', sans-serif;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
    font-size: 14px;
    line-height: 1.6;
  }}
  body::before {{
    content:'';
    position:fixed; inset:0;
    background-image:
      linear-gradient(rgba(91,111,255,.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(91,111,255,.02) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events:none;
    z-index:0;
  }}
  .container {{ position:relative; z-index:1; max-width:1400px; margin:0 auto; padding:24px 20px; }}
</style>
</head>
<body>
<div class="container">

  <!-- 헤더 -->
  <div class="header">
    <div>
      <div class="header-title">ZEUS <span>MARKET</span> FLOW</div>
      <div class="header-sub">SMART MONEY DASHBOARD — DAILY BRIEFING {status_badge}</div>
    </div>
    <div class="header-time">
      <span class="live-dot"></span><span id="live-clock">{now_str}</span>
      <div style="font-size:.62rem;color:var(--text-dim);margin-top:2px">생성: {now_str}</div>
    </div>
  </div>

  <!-- BIG PICTURE -->
  <div class="big-picture">
    <div class="big-label">▸ TODAY'S BIG PICTURE</div>
    <div class="big-strategy">{big['strategy']}</div>
    <ul class="big-details">{detail_items}</ul>
  </div>

  <!-- 매집 레이더 -->
  <div class="section-header">🎯 매집 레이더 <span style="font-size:.75rem;font-family:var(--mono);color:var(--text-dim);font-weight:400;letter-spacing:0">— 횡보매집·바닥신호·OBV선행 중 2개 이상 동시 충족</span></div>
  <div class="radar-count">총 <span>{len(radar_stocks)}</span>개 포착 &nbsp;·&nbsp; 👑 = 3개 전부 충족 &nbsp;·&nbsp; 카드 클릭 시 섹터 상세 보기</div>
  <div class="radar-grid">{radar_rows_html}</div>

  <!-- 섹터 로테이션 -->
  <div class="rotation-badge">{rotation}</div>
  <div class="sector-grid">{sector_cards}</div>

  <!-- 공포탐욕 + 매크로 -->
  <div class="grid-2">
    <!-- 공포탐욕 -->
    <div>
      <div class="section-header">🧠 공포/탐욕 지수</div>
      <div class="fg-container">
        <div class="fg-gauge">
          <div class="fg-score-big">{fg_score:.0f}</div>
          <div class="fg-label-big">{fg_label}</div>
          <div class="fg-bar"><div class="fg-pointer"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:.65rem;color:var(--text-dim);font-family:var(--mono)">
            <span>극도 공포</span><span>공포</span><span>중립</span><span>탐욕</span><span>극도 탐욕</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 매크로 -->
    <div>
      <div class="section-header">🌐 매크로 지표</div>
      <div class="macro-grid">{macro_cards}</div>
    </div>
  </div>

  <!-- 숏스퀴즈 후보 -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px">
    <div>
      <div class="section-header">🩳 숏스퀴즈 후보
        <span style="font-size:.72rem;font-family:var(--mono);color:var(--text-dim);font-weight:400;letter-spacing:0"> — 공매도 잔고 ≥15% + 거래량 급증</span>
      </div>
      <div class="section-table-wrap">{squeeze_html}</div>
    </div>
    <div>
      <div class="section-header">📅 실적 발표 캘린더
        <span style="font-size:.72rem;font-family:var(--mono);color:var(--text-dim);font-weight:400;letter-spacing:0"> — 60일 이내 어닝 예정 (스마트점수 보유 종목)</span>
      </div>
      <div class="section-table-wrap">{earn_html}</div>
    </div>
  </div>

  <!-- 기관 데이터: 내부자 매수 + 기관 보유 -->
  <div class="section-header">🏛️ 기관 투자자 동향
    <span style="font-size:.72rem;font-family:var(--mono);color:var(--text-dim);font-weight:400;letter-spacing:0"> — SEC 공시 기반 · 지연 데이터 참고용</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px">

    <div>
      <div style="font-family:var(--mono);font-size:.72rem;color:var(--text-dim);margin-bottom:8px;display:flex;align-items:center;gap:8px">
        🏛️ 내부자 매수 (SEC Form 4 공시)
        <span style="color:#ff8844;font-size:.68rem">⚠️ 최대 45일 지연</span>
      </div>
      <div class="inst-section-wrap">
        <div class="congress-grid">{congress_html}</div>
        <div class="inst-staleness-note">
          📌 수집 시각: {congress_upd} &nbsp;·&nbsp;
          STOCK Act: 의원은 거래 후 45일 이내 신고 의무 &nbsp;·&nbsp;
          <span style="color:#ff8844">날짜 = 신고일 기준, 실제 거래일과 다를 수 있음</span>
        </div>
      </div>
    </div>

    <div>
      <div style="font-family:var(--mono);font-size:.72rem;color:var(--text-dim);margin-bottom:8px;display:flex;align-items:center;gap:8px">
        🏦 기관 보유 비율 기관 보유 Top
        <span style="color:#ff8844;font-size:.68rem">⚠️ 분기 지연</span>
      </div>
      <div class="inst-section-wrap">
        <div class="t13f-grid">{t13f_html}</div>
        <div class="inst-staleness-note">
          📌 기준 분기: {last_13f_label}<br>
          13F: $100M+ 운용 기관이 분기 종료 후 45일 이내 SEC에 신고 &nbsp;·&nbsp;
          <span style="color:#ff8844">현재 포지션과 다를 수 있음. 방향성 참고용으로만 활용</span>
        </div>
      </div>
    </div>

  </div>

  <!-- 스마트머니 Top10 -->
  <div class="section-header">🔥 스마트머니 유입 Top 10</div>
  <div class="smart-table-wrap">
    <table class="smart-table">
      <thead>
        <tr>
          <th>#</th><th>티커</th><th>현재가</th>
          <th>1일</th><th>5일</th><th>20일</th>
          <th>거래량비</th><th>RS vs SPY</th><th>가속도</th><th>종합점수(상대순위)</th>
        </tr>
      </thead>
      <tbody>{smart_rows}</tbody>
    </table>
  </div>

  <div class="footer">
    ZEUS MARKET FLOW DASHBOARD  ·  데이터: Yahoo Finance  ·
    캐시: 가격 {len(DataFetcher._price_cache)}종목 / 슬로우 {len(DataFetcher._slow_cache)}종목  ·
    본 자료는 투자 참고용이며 투자 손실에 대한 책임을 지지 않습니다
  </div>
</div>

<!-- 섹터 상세 모달 -->
<div class="modal-overlay" id="detailOverlay" onclick="closeOnOverlay(event)">
  <div class="modal" id="detailModal">
    <div class="modal-header">
      <div class="modal-title" id="modalTitle">섹터 <span>상세</span></div>
      <button class="modal-close" onclick="closeDetail()">✕</button>
    </div>
    <div class="modal-body">
      <div class="sort-tabs">
        <button class="sort-tab active" onclick="sortTable('cs_score')">🔥 단기(상대)</button>
        <button class="sort-tab" onclick="sortTable('cf_score')">💎 장기(상대)</button>
        <button class="sort-tab" onclick="sortTable('cs_combo')">⚡ 종합</button>
        <button class="sort-tab" onclick="sortTable('smart_score')">📊 단기(절대)</button>
        <button class="sort-tab" onclick="sortTable('ret_1d')">📅 1일 수익률</button>
        <button class="sort-tab" onclick="sortTable('ret_5d')">📅 5일 수익률</button>
        <button class="sort-tab" onclick="sortTable('vol_ratio')">📊 거래량 20D</button>
          <button class="sort-tab" onclick="sortTable('vol_ratio_1d')">📊 거래량 1D</button>
        <button class="sort-tab" onclick="sortTable('atr_pct')">⚡ 변동성(ATR)</button>
        <button class="sort-tab" onclick="sortTable('short_pct')">🩳 공매도 잔고</button>
      </div>
      <div class="detail-table-wrap">
        <table class="detail-table">
          <thead>
            <tr>
              <th>#</th>
              <th>티커 / 변동성</th>
              <th>현재가</th>
              <th>1일</th>
              <th>5일 / vs SPY</th>
              <th>20일</th>
              <th>거래량 / ATR</th>
              <th>7요소 점수바</th>
              <th>스마트점수 /100</th>
            </tr>
          </thead>
          <tbody id="detailTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// 섹터 상세 데이터 (Python이 embed)
const DETAIL_DATA = {detail_json};

let currentEtf  = null;
let currentSort = 'smart_score';

function openDetail(etf, name, emoji) {{
  currentEtf = etf;
  document.getElementById('modalTitle').innerHTML =
    emoji + ' ' + name + ' <span>(' + etf + ')</span>';
  document.getElementById('detailOverlay').classList.add('open');
  currentSort = 'smart_score';
  document.querySelectorAll('.sort-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.sort-tab')[0].classList.add('active');
  renderTable(etf, currentSort);
  document.body.style.overflow = 'hidden';
}}

function closeDetail() {{
  document.getElementById('detailOverlay').classList.remove('open');
  document.body.style.overflow = '';
}}

function closeOnOverlay(e) {{
  if (e.target === document.getElementById('detailOverlay')) closeDetail();
}}

function sortTable(field) {{
  currentSort = field;
  document.querySelectorAll('.sort-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  renderTable(currentEtf, field);
}}

function renderTable(etf, sortField) {{
  const rows = DETAIL_DATA[etf];
  if (!rows || rows.length === 0) {{
    document.getElementById('detailTableBody').innerHTML =
      '<tr><td colspan="10" style="text-align:center;padding:32px;color:var(--text-dim)">데이터 없음</td></tr>';
    return;
  }}

  const sorted = [...rows].sort((a, b) => (b[sortField] || 0) - (a[sortField] || 0));
  const rankColors = ['var(--yellow)','#c0c0c0','#cd7f32'];

  const html = sorted.map((r, i) => {{
    const r1c   = r.ret_1d  >= 0 ? 'var(--green)' : '#ff4466';
    const r5c   = r.ret_5d  >= 0 ? 'var(--green)' : '#ff4466';
    const r20c  = r.ret_20d >= 0 ? 'var(--green)' : '#ff4466';
    const rc    = rankColors[i] || '#8899aa';
    const volC  = r.vol_ratio >= 2.0 ? '#ff8844' : r.vol_ratio >= 1.5 ? 'var(--yellow)' : '#4a8080';
    const volFlag = r.vol_ratio >= 2.0 ? '🔥' : r.vol_ratio >= 1.5 ? '⬆' : '';
    const vr1d   = r.vol_ratio_1d ?? r.vol_ratio;
    const vol1dC = vr1d >= 2.0 ? '#ff8844' : vr1d >= 1.5 ? 'var(--yellow)' : '#4a8080';
    const atrC  = r.atr_pct >= 5 ? '#ff4466' : r.atr_pct >= 3 ? '#ff8844' : r.atr_pct >= 2 ? 'var(--yellow)' : 'var(--green)';
    const sc    = r.cs_score ?? r.smart_score ?? 0;   // 크로스섹셔널 단기
    const scC   = sc >= 80 ? 'var(--green)' : sc >= 60 ? '#88ff44' : sc >= 40 ? '#ffdd00' : sc >= 20 ? '#ff8844' : 'var(--red)';
    const csCombo = r.cs_combo ?? sc;
    const revTag = r.reversal ? '<span style="font-size:.65rem;color:#ff8844;border:1px solid #ff8844;border-radius:3px;padding:0 4px">반전</span>' : '';
    const accumTags = (r.accum_tags || []).map(t =>
      `<span style="font-size:.6rem;color:var(--accent);border:1px solid rgba(0,212,255,.4);border-radius:3px;padding:0 3px;margin-right:2px">${{t}}</span>`
    ).join('');
    const spyTag = r.vs_spy > 3 ? `<span style="font-size:.6rem;color:var(--green)">SPY+${{r.vs_spy.toFixed(1)}}%</span>` :
                   r.vs_spy < -3 ? `<span style="font-size:.6rem;color:#ff4466">SPY${{r.vs_spy.toFixed(1)}}%</span>` : '';

    // 8요소 미니 바 (새 배점 반영)
    const bars = [
      {{label:'매집',  val:r.sc_pos  ||0, max:25, tip:'매집탐지'}},
      {{label:'OBV',   val:r.sc_obv  ||0, max:20, tip:'OBV 추세'}},
      {{label:'거래량', val:r.sc_vol  ||0, max:18, tip:'거래량구조'}},
      {{label:'옵션',  val:r.sc_opt  ||0, max:15, tip:'옵션신호'}},
      {{label:'가속',  val:r.sc_accel||0, max:12, tip:'모멘텀가속'}},
      {{label:'RS',    val:r.sc_rs   ||0, max:10, tip:'상대강도'}},
      {{label:'BB',    val:r.sc_bb   ||0, max:8,  tip:'BB압축'}},
      {{label:'추세',  val:r.sc_trend||0, max:7,  tip:'추세일관성'}},
    ].map(b => {{
      const pct = Math.round((b.val / b.max) * 100);
      const bc  = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--yellow)' : '#ff4466';
      return `<div style="display:flex;align-items:center;gap:3px;margin-bottom:2px">
        <span style="font-size:.58rem;color:var(--text-dim);width:28px">${{b.label}}</span>
        <div style="width:42px;height:4px;background:var(--border);border-radius:2px">
          <div style="width:${{pct}}%;height:100%;background:${{bc}};border-radius:2px"></div>
        </div>
        <span style="font-size:.58rem;color:${{bc}};width:18px">${{b.val.toFixed(0)}}</span>
      </div>`;
    }}).join('');

    // 공매도 배지
    const sqTag = r.short_squeeze
      ? `<span style="font-size:.6rem;color:#ff6b6b;border:1px solid #ff6b6b;border-radius:3px;padding:0 3px">🩳 숏 ${{(r.short_pct||0).toFixed(0)}}%</span>` : '';
    // 어닝 배지
    const earnTag = r.earn_days != null
      ? `<span style="font-size:.6rem;color:${{r.earn_days<=3?'#ff4466':r.earn_days<=7?'#ff8844':'#888'}};border:1px solid;border-radius:3px;padding:0 3px">📅 ${{r.earn_days}}일 후</span>` : '';
    // P/C 배지
    const pcTag = r.pc_ratio != null
      ? `<span style="font-size:.6rem;color:${{r.pc_ratio<0.7?'var(--green)':'#aaa'}};border:1px solid;border-radius:3px;padding:0 3px">P/C ${{r.pc_ratio.toFixed(2)}}</span>` : '';

    return `<tr>
      <td><span class="rank" style="color:${{rc}}">#${{i+1}}</span></td>
      <td class="dt-ticker">
        ${{r.ticker}}<br>
        <span style="font-size:.65rem;color:var(--text-dim)">${{r.vol_grade}}</span>
        ${{revTag}} ${{accumTags}} ${{sqTag}} ${{earnTag}} ${{pcTag}}
      </td>
      <td style="font-family:var(--mono)">${{r.price.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}</td>
      <td style="color:${{r1c}};font-family:var(--mono)">${{r.ret_1d>=0?'+':''}}${{r.ret_1d.toFixed(1)}}%</td>
      <td style="color:${{r5c}};font-family:var(--mono)">${{r.ret_5d>=0?'+':''}}${{r.ret_5d.toFixed(1)}}%<br>${{spyTag}}</td>
      <td style="color:${{r20c}};font-family:var(--mono)">${{r.ret_20d>=0?'+':''}}${{r.ret_20d.toFixed(1)}}%</td>
      <td style="font-family:var(--mono)">
        ${{volFlag}} <span style='color:${{volC}}'>20D:${{r.vol_ratio.toFixed(1)}}x</span> <span style='color:#556'>|</span> <span style='color:${{vol1dC}}'>1D:${{vr1d.toFixed(1)}}x</span><br>
        <span style="color:${{atrC}};font-size:.72rem">ATR ${{r.atr_pct.toFixed(1)}}%</span>
      </td>
      <td>
        <div style="display:flex;flex-direction:column">
          ${{bars}}
        </div>
      </td>
      <td>
        <div style="text-align:center">
          <div style="font-family:var(--mono);font-size:1.1rem;font-weight:700;color:${{scC}}">${{sc.toFixed(0)}}</div>
          <div style="font-size:.65rem;color:${{scC}}">${{r.cs_badge??r.badge}} ${{r.cs_grade??r.grade}}</div>
          <div style="width:44px;height:4px;background:var(--border);border-radius:2px;margin:3px auto 0">
            <div style="width:${{sc}}%;height:100%;background:${{scC}};border-radius:2px"></div>
          </div>
          <div style="font-size:.6rem;color:var(--text-dim);margin-top:2px">단기</div>
        </div>
      </td>
      <td>
        <div style="text-align:center">
          ${{(()=>{{
            const fs=r.cf_score??r.factor_score??0;
            const fc=fs>=70?'var(--accent)':fs>=50?'#7b5fff':fs>=35?'#4a8080':'#2a3344';
            const rg=r.factor_regime||'sideways';
            const rgL=rg==='bull'?'🟢':rg==='bear'?'🔴':'🟡';
            return `<div style="font-family:var(--mono);font-size:1.1rem;font-weight:700;color:${{fc}}">${{fs.toFixed(0)}}</div>
                    <div style="font-size:.65rem;color:${{fc}}">${{r.cf_badge??r.factor_badge??'—'}} ${{r.cf_grade??r.factor_grade??'—'}}</div>
                    <div style="width:44px;height:4px;background:var(--border);border-radius:2px;margin:3px auto 0">
                      <div style="width:${{Math.min(fs,100)}}%;height:100%;background:${{fc}};border-radius:2px"></div>
                    </div>
                    <div style="font-size:.6rem;color:var(--text-dim);margin-top:2px">${{rgL}} 장기</div>`;
          }})()}}
        </div>
      </td>
    </tr>`;
  }}).join('');

  document.getElementById('detailTableBody').innerHTML = html;
}}

// ESC 키로 닫기
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDetail(); }});

// ── 실시간 시계 (한국 기준 현재시각 표시) ────────────────────
(function() {{
  function updateClock() {{
    const now = new Date();
    const koOpts = {{
      timeZone: 'Asia/Seoul',
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false
    }};
    const parts = new Intl.DateTimeFormat('ko-KR', koOpts).formatToParts(now);
    const p = Object.fromEntries(parts.map(x => [x.type, x.value]));
    const str = `${{p.year}}년 ${{p.month}}월 ${{p.day}}일  ${{p.hour}}:${{p.minute}}:${{p.second}} KST`;
    const el = document.getElementById('live-clock');
    if (el) el.textContent = str;
  }}
  updateClock();
  setInterval(updateClock, 1000);
}})();
</script>

<!-- ════════════════════════════════════════════════ A등급 모듈 ═══ -->

<!-- 1. 리스크 관리 & 포지션 사이징 -->
<div class="section-header" style="margin-top:28px">
  ⚖️ 리스크 관리 &amp; 포지션 사이징
  <span style="font-size:.75rem;font-family:var(--mono);color:var(--text-dim);font-weight:400">
    — ATR 기반 사이징 · 손절가 · 거래비용 반영 (계좌 $100,000 기준)
  </span>
</div>
<style>
.risk-card{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;
           margin-bottom:10px;display:inline-block;min-width:260px;max-width:320px;
           vertical-align:top;margin-right:10px}}
.risk-header{{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}}
.risk-ticker{{font-size:1rem;font-weight:700;color:var(--accent);font-family:var(--mono)}}
.risk-body{{font-size:.8rem}}
.risk-row{{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04)}}
.risk-lbl{{color:var(--text-dim)}}
.risk-val{{font-family:var(--mono);font-weight:600}}
</style>
<div style="overflow-x:auto;padding-bottom:8px">{risk_cards_html}</div>

<!-- 2. 포트폴리오 상관분석 & 섹터 집중도 -->
<div class="section-header" style="margin-top:28px">
  🔗 포트폴리오 분산 분석
  <span style="font-size:.75rem;font-family:var(--mono);color:var(--text-dim);font-weight:400">
    — 종목 간 상관 · 섹터 집중도 HHI · 분산 효과
  </span>
</div>
<style>
.corr-summary,.conc-summary{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px 20px;margin-bottom:16px}}
.corr-pair{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:.82rem}}
.conc-row{{display:flex;align-items:center;gap:10px;padding:3px 0;font-size:.82rem}}
.conc-sec{{min-width:70px;color:var(--text-dim)}}
.conc-bar-wrap{{flex:1;height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}}
.conc-bar{{height:100%;border-radius:3px}}
.conc-val{{min-width:44px;text-align:right;font-family:var(--mono);font-size:.78rem;font-weight:700}}
</style>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div>
    <div style="font-size:.82rem;color:var(--text-dim);margin-bottom:8px">📊 종목 간 상관 계수 (60일 수익률)</div>
    {corr_html}
  </div>
  <div>
    <div style="font-size:.82rem;color:var(--text-dim);margin-bottom:8px">🏢 섹터 집중도 (레이더 상위 20종목)</div>
    {concentration_html}
  </div>
</div>

<!-- 3. 스트레스 테스트 -->
<div class="section-header" style="margin-top:28px">
  📉 역사적 테일리스크 스트레스 테스트
  <span style="font-size:.75rem;font-family:var(--mono);color:var(--text-dim);font-weight:400">
    — 2008 금융위기 · 2020 코로나 · 2022 금리충격 · 2011 유럽재정위기
  </span>
</div>
<style>
.stress-card{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin-bottom:12px}}
.stress-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap}}
.stress-name{{font-size:.95rem;font-weight:700;color:var(--accent)}}
.stress-spy{{font-size:.85rem;font-weight:700}}
.stress-desc{{font-size:.75rem;color:var(--text-dim)}}
.stress-stats{{display:flex;gap:20px;margin-bottom:8px;flex-wrap:wrap}}
.stress-stat{{text-align:center}}
.st-lbl{{display:block;font-size:.7rem;color:var(--text-dim);margin-bottom:2px}}
.st-val{{font-size:.95rem;font-weight:700;font-family:var(--mono)}}
.stress-tickers{{border-top:1px solid rgba(255,255,255,.07);padding-top:7px}}
.stress-empty{{color:var(--text-dim);padding:16px;text-align:center;font-size:.85rem}}
</style>
{stress_html}

<!-- 4. 감사 추적 로그 -->
<div class="section-header" style="margin-top:28px">
  📋 신호 생성 근거 감사 추적 (Audit Log)
  <span style="font-size:.75rem;font-family:var(--mono);color:var(--text-dim);font-weight:400">
    — 이번 세션 신호 이력 · 컴포넌트 세부값 · zeus_audit_log.json 저장
  </span>
</div>
<style>
.audit-summary{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px 20px}}
.audit-stats{{display:flex;gap:20px;margin-bottom:14px;flex-wrap:wrap}}
.audit-stat{{text-align:center;background:rgba(255,255,255,.03);border-radius:6px;padding:8px 16px}}
.audit-val{{display:block;font-size:1.2rem;font-weight:700;font-family:var(--mono);color:var(--accent)}}
.audit-lbl{{display:block;font-size:.7rem;color:var(--text-dim);margin-top:2px}}
.audit-table{{width:100%;border-collapse:collapse;font-size:.8rem}}
.audit-table th{{background:rgba(255,255,255,.05);color:var(--text-dim);padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
.audit-table td{{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,.04)}}
.audit-empty{{color:var(--text-dim);padding:16px;font-size:.85rem}}
</style>
{audit_html}

</body>
</html>"""
    return html


# ──────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────
def _parse_bulk_global(raw_df, tk_list):
    """run_once용 전역 bulk 파서 (group_by=column)."""
    import pandas as pd
    result = {}
    if raw_df is None or raw_df.empty: return result
    if not isinstance(raw_df.columns, pd.MultiIndex):
        if len(tk_list)==1:
            df = raw_df.copy()
            df.columns = [str(c).strip().title() for c in df.columns]
            if "Adj Close" in df.columns:
                adj = pd.to_numeric(df["Adj Close"], errors="coerce")
                if not adj.isna().all(): df["Close"] = adj
            for col in list(df.columns): df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"]).sort_index() if "Close" in df.columns else pd.DataFrame()
            if not df.empty: result[tk_list[0]] = df
        return result
    tickers_in = raw_df.columns.get_level_values(1).unique().tolist()
    for tk in tk_list:
        try:
            if tk not in tickers_in: continue
            row = {}
            for price in raw_df.columns.get_level_values(0).unique():
                try:
                    v = raw_df[(price, tk)]
                    row[str(price).strip().title()] = pd.to_numeric(v, errors="coerce")
                except Exception as _e:
                    logger.debug("예외 무시: %s", _e)
            if not row: continue
            df = pd.DataFrame(row, index=raw_df.index)
            if "Adj Close" in df.columns:
                adj = df["Adj Close"]
                if not adj.isna().all(): df["Close"] = adj
            if "Close" not in df.columns: continue
            df = df.dropna(subset=["Close"]).sort_index()
            if not df.empty: result[tk] = df
        except Exception as _e:
            logger.debug("예외 무시: %s", _e)
    return result


def run_once(open_browser: bool = False):
    W = 64
    mstat    = market_status()
    fetcher  = DataFetcher()
    analyzer = MarketAnalyzer()

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _sr = yf.download("SPY", period="18mo", interval="1d", auto_adjust=False, progress=False)
            _vr = yf.download("^VIX", period="5d",  interval="1d", auto_adjust=False, progress=False)
        if isinstance(_sr.columns, pd.MultiIndex): _sr.columns = _sr.columns.get_level_values(0)
        _sc = pd.to_numeric(_sr.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        _vv = float(pd.to_numeric(_vr.get("Close", pd.Series(dtype=float)), errors="coerce").dropna().iloc[-1]) if not _vr.empty else 20.0
        _rg = SmartScoreEngine.detect_regime(_sc, _vv)
        mstat["regime"] = _rg
        _rl = {"bull":"🟢 상승장","bear":"🔴 하락장","sideways":"🟡 횡보장"}
        print(f"  📊 시장국면: {_rl[_rg]} (VIX {_vv:.1f})")
    except Exception as _e:
        mstat["regime"] = "sideways"
        print(f"  ⚠️  국면감지 실패 → 횡보장 기본값")

    # (regime 감지 완료 — 위에서 처리됨)

    print(f"\n{'━'*W}")
    print(f"  🌊 ZEUS MARKET FLOW DASHBOARD v4")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {mstat['label']}")
    print(f"{'━'*W}")

    # ── [1단계] 모든 필요 티커 수집 후 bulk 1회 다운로드 ──────────
    all_needed = list({
        tk
        for tks in SECTOR_STOCKS.values() for tk in tks
    } | set(WATCHLIST) | set(MACRO_TICKERS.keys()) | {"SPY"} | set(SECTORS.keys()))

    now_ts = time.time()
    uncached_all = [tk for tk in all_needed
                    if tk not in DataFetcher._price_cache
                    or (now_ts - DataFetcher._price_cache[tk]["ts"]) >= DataFetcher._PRICE_TTL]

    if uncached_all:
        print(f"\n  📦 전체 티커 bulk 다운로드 {len(uncached_all)}개 (청크 분할)...")
        ts_now = time.time()
        saved = 0

        # 500개 이상이면 Rate Limit 걸림 → 200개씩 청크로 나눠서 다운로드
        CHUNK = 200
        chunks = [uncached_all[i:i+CHUNK] for i in range(0, len(uncached_all), CHUNK)]

        def _parse_bulk(raw_df, tk_list):
            """group_by="column": level0=Price, level1=Ticker"""
            result = {}
            if raw_df is None or raw_df.empty:
                return result
            if not isinstance(raw_df.columns, pd.MultiIndex):
                if len(tk_list) == 1:
                    df = raw_df.copy()
                    df.columns = [str(c).strip().title() for c in df.columns]
                    if "Adj Close" in df.columns:
                        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
                        if not adj.isna().all(): df["Close"] = adj
                    for col in list(df.columns):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["Close"]).sort_index() if "Close" in df.columns else pd.DataFrame()
                    if not df.empty: result[tk_list[0]] = df
                return result

            # level0=Price(Close/High/...), level1=Ticker
            tickers_in = raw_df.columns.get_level_values(1).unique().tolist()
            for tk in tk_list:
                try:
                    if tk not in tickers_in: continue
                    row = {}
                    for price in raw_df.columns.get_level_values(0).unique():
                        try:
                            v = raw_df[(price, tk)]
                            row[str(price).strip().title()] = pd.to_numeric(v, errors="coerce")
                        except Exception as _e:
                            logger.debug("예외 무시: %s", _e)
                    if not row: continue
                    df = pd.DataFrame(row, index=raw_df.index)
                    if "Adj Close" in df.columns:
                        adj = df["Adj Close"]
                        if not adj.isna().all(): df["Close"] = adj
                    if "Close" not in df.columns: continue
                    df = df.dropna(subset=["Close"]).sort_index()
                    if not df.empty: result[tk] = df
                except Exception as _e:
                    logger.debug("예외 무시: %s", _e)
            return result
        for i, chunk in enumerate(chunks):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    raw_chunk = yf.download(
                        chunk, period="1y", interval="1d",
                        group_by="column", auto_adjust=False, progress=False
                    )
                parsed = _parse_bulk(raw_chunk, chunk)
                for tk, df in parsed.items():
                    with DataFetcher._price_lock:
                        DataFetcher._price_cache[tk] = {"df": df, "ts": ts_now}
                    saved += 1

                # bulk 누락 종목 개별 폴백
                # → bulk 다운로드에서 파싱 못한 종목(상장폐지·티커변경 제외)을
                #   개별 Ticker.history로 재시도
                missing = [tk for tk in chunk if tk not in parsed]
                if missing:
                    def _fallback_one(tk):
                        try:
                            import io as _io, contextlib as _ctx
                            with _ctx.redirect_stderr(_io.StringIO()):
                                df = yf.Ticker(tk).history(period="1y", interval="1d")
                            if df is None or df.empty:
                                return tk, None
                            df = df.copy()
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            df.columns = [str(c).strip().title() for c in df.columns]
                            if "Close" not in df.columns:
                                return tk, None
                            df = df.dropna(subset=["Close"]).sort_index()
                            return tk, df if not df.empty else None
                        except Exception as _e:
                            return tk, None
                    from concurrent.futures import ThreadPoolExecutor as _FBPool, as_completed as _fbc
                    with _FBPool(max_workers=10) as _fbp:
                        _futs = {_fbp.submit(_fallback_one, tk): tk for tk in missing}
                        fb_ok = 0
                        for _fut in _fbc(_futs):
                            tk, df = _fut.result()
                            if df is not None:
                                with DataFetcher._price_lock:
                                    DataFetcher._price_cache[tk] = {"df": df, "ts": ts_now}
                                saved += 1
                                fb_ok += 1
                    if fb_ok > 0:
                        print(f"    청크 {i+1} 폴백: {fb_ok}/{len(missing)}개 복구")

                print(f"    청크 {i+1}/{len(chunks)}: {len(parsed)}/{len(chunk)}개 저장")
                if i < len(chunks) - 1:
                    time.sleep(1)  # 청크 간 1초 대기
            except Exception as e:
                print(f"    청크 {i+1} 실패: {type(e).__name__}: {e}")

        print(f"  ✅ bulk 캐시 {saved}/{len(uncached_all)}개 저장")
        DataFetcher._save_disk_cache()
        print(f"  💾 디스크 캐시 저장 완료")
    else:
        print(f"\n  📦 전체 캐시 히트 ({len(all_needed)}개), 다운로드 생략")

    # ── [2단계] 캐시 기반으로 각 데이터 수집 ─────────────────────
    sector_df    = fetcher.fetch_sector(mstat)
    macro        = fetcher.fetch_macro(mstat)
    watchlist_df = fetcher.fetch_watchlist(mstat)

    if sector_df.empty:
        print("  ❌ 섹터 데이터 없음"); return

    # 섹터별 상세 종목 수집 — 캐시 채운 후 병렬 처리 (bulk 다운로드 없음)
    print(f"\n  📦 섹터 상세 수집 (병렬, {len(SECTOR_STOCKS)}개 섹터)...")
    t0 = time.time()
    sector_detail_data: Dict[str, List] = {}

    def _fetch_one_sector(etf):
        return etf, fetcher.fetch_sector_detail(etf, mstat)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one_sector, etf): etf for etf in SECTOR_STOCKS.keys()}
        for fut in as_completed(futures):
            etf, data = fut.result()
            sector_detail_data[etf] = data

    elapsed = time.time() - t0
    total_stocks = sum(len(v) for v in sector_detail_data.values())
    print(f"  ✅ 상세 종목 {total_stocks}개 수집 완료 ({elapsed:.0f}초)")

    # ── 매집 레이더: 태그 2개 이상 종목 추출 ──────────────
    radar_stocks  = []
    squeeze_stocks = []
    earn_upcoming: List[Dict] = []
    radar_seen   = set()   # 중복 방지
    squeeze_seen = set()   # 중복 방지
    earn_seen    = set()

    for etf, stocks in sector_detail_data.items():
        sec_name = SECTORS.get(etf, ("?","",""))[0]
        for s in stocks:
            tags = s.get("accum_tags", [])
            tk   = s.get("ticker","")
            s_ext = {**s, "etf": etf, "sector": sec_name}

            # 매집 레이더 (태그 2개 이상, 중복 제거)
            if len(tags) >= 2 and tk not in radar_seen:
                radar_seen.add(tk)
                radar_stocks.append(s_ext)

            # 숏스퀴즈 후보 (중복 제거)
            if s.get("short_squeeze") and tk not in squeeze_seen:
                squeeze_seen.add(tk)
                squeeze_stocks.append(s_ext)

            # 어닝 캘린더 (중복 제거)
            if s.get("earn_date") and tk not in earn_seen:
                earn_seen.add(tk)
                earn_upcoming.append({
                    "ticker":    tk,
                    "sector":    sec_name,
                    "etf":       etf,
                    "earn_date": s["earn_date"],
                    "earn_days": s["earn_days"],
                    "price":     s["price"],
                    "ret_5d":    s["ret_5d"],
                    "smart_score": s["smart_score"],
                    "cs_score":    s.get("cs_score", s["smart_score"]),
                })

    radar_stocks.sort(key=lambda x: x.get("cs_score", x.get("smart_score",0)), reverse=True)
    squeeze_stocks.sort(key=lambda x: x.get("short_pct", 0), reverse=True)
    earn_upcoming.sort(key=lambda x: x["earn_days"])

    print(f"  🎯 매집 레이더 포착: {len(radar_stocks)}개")
    print(f"  🩳 숏스퀴즈 후보:    {len(squeeze_stocks)}개")
    print(f"  📅 어닝 60일 이내:   {len(earn_upcoming)}개")

    # ── [A등급] 스트레스 테스트 ───────────────────────────
    print(f"\n  📉 스트레스 테스트 실행 (역사적 시나리오 4개)...")
    # BUG FIX: watchlist_df["ticker"]는 Series of str → s가 이미 ticker 문자열
    _wl_tickers = (
        watchlist_df["ticker"].dropna().tolist()[:10]
        if "ticker" in watchlist_df.columns else []
    )
    _stress_tickers = list(dict.fromkeys(
        [s["ticker"] for s in radar_stocks[:20]] + _wl_tickers
    ))[:30]   # dict.fromkeys: 순서 유지 + 중복 제거
    stress_results = _stress_engine.run_portfolio_stress(
        _stress_tickers, DataFetcher._price_cache
    )
    _covered = sum(1 for v in stress_results.values() if v.get("summary", {}).get("n_available", 0) > 0)
    print(f"  ✅ 스트레스 테스트 완료: {len(stress_results)}개 시나리오 / {_covered}개 유효")

    # ── [A등급] 포트폴리오 상관분석 ──────────────────────
    print(f"\n  🔗 포트폴리오 상관분석 실행...")
    corr_result = _portfolio_ctor.build_correlation_matrix(
        DataFetcher._price_cache, lookback=60
    )
    # BUG FIX: weight=1.0 (equal weight) — cs_score는 실제 포지션 금액이 아님
    # equal weight: "각 종목을 균등 금액으로 보유할 경우"의 섹터 분포를 보여줌
    radar_holdings = [
        {"ticker": s["ticker"], "etf": s.get("etf", "UNKNOWN"), "weight": 1.0}
        for s in radar_stocks[:20]
    ]
    concentration = _portfolio_ctor.sector_concentration(radar_holdings)
    print(f"  ✅ 상관분석 완료: {corr_result['n_stocks']}개 종목 / 평균상관 {corr_result['avg_corr']:.3f} / HHI {concentration['hhi']:.0f}")
    if concentration.get("warning"):
        print(f"  {concentration['warning']}")

    # ── 기관 데이터: 내부자 매수 + 기관 보유 ──────────────────
    print(f"\n  🏛️  기관 데이터 수집 (내부자 매수 + 기관 보유)...")
    inst_data = fetcher.fetch_institutional_data()
    print(f"  ✅ 내부자 매수 {len(inst_data['congress'])}건 / 13F 종목 {len(inst_data['top13f'])}개")

    fg_score, fg_label, fg_color = analyzer.fear_greed(macro)
    rotation = analyzer.sector_rotation_phase(sector_df)
    big      = analyzer.big_picture(sector_df, macro, fg_score, fg_label, rotation)

    print(f"\n  공포/탐욕: {fg_score:.0f}  {fg_label}")
    print(f"  로테이션: {rotation}")
    print(f"  전략: {big['strategy']}")

    html = build_html(sector_df, macro, watchlist_df,
                      fg_score, fg_label, fg_color, rotation, big,
                      mstat, sector_detail_data, radar_stocks,
                      squeeze_stocks, earn_upcoming, inst_data,
                      stress_results=stress_results,
                      corr_result=corr_result,
                      concentration=concentration)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  💾 저장: {out_path}")
    if open_browser:
        webbrowser.open(f"file://{out_path}")
    print(f"{'━'*W}")


def main():
    live_mode = "--live" in sys.argv

    if live_mode:
        print(f"\n  ▶ LIVE 모드 — {REFRESH_SEC//60}분마다 자동 갱신  (Ctrl+C 종료)")
        run_once(open_browser=True)
        try:
            while True:
                time.sleep(REFRESH_SEC)
                print(f"\n  🔄 [{datetime.now().strftime('%H:%M:%S')}] 데이터 갱신 중...")
                run_once(open_browser=False)
        except KeyboardInterrupt:
            print("\n  ⏹  종료")
    else:
        run_once(open_browser=True)


if __name__ == "__main__":
    main()