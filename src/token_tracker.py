# -*- coding: utf-8 -*-
"""统一 Token 统计器。

目标：
- 在 ingest / infer 两个阶段分别累计 token 消耗
- 支持按来源（llm_chat / llm_json / embedding / summarize）拆分
- 提供可视化调试输出与结构化快照
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Any, Dict

from .logger import get_logger

_logger = get_logger()
_PHASE: ContextVar[str] = ContextVar("token_phase", default="unknown")


@dataclass
class _UsageBucket:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_total_tokens: int = 0
    call_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class TokenUsageTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[str, Dict[str, _UsageBucket]] = {}

    def reset(self) -> None:
        with self._lock:
            self._data = {}

    def set_phase(self, phase: str) -> None:
        _PHASE.set((phase or "unknown").strip().lower())

    def get_phase(self) -> str:
        return _PHASE.get()

    def record(
        self,
        source: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        estimated: bool = False,
    ) -> None:
        phase = self.get_phase()
        source = (source or "unknown_source").strip().lower()

        with self._lock:
            phase_data = self._data.setdefault(phase, {})
            bucket = phase_data.setdefault(source, _UsageBucket())
            bucket.call_count += 1
            bucket.prompt_tokens += int(prompt_tokens or 0)
            bucket.completion_tokens += int(completion_tokens or 0)
            bucket.total_tokens += int(total_tokens or 0)
            if estimated:
                bucket.estimated_total_tokens += int(total_tokens or 0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            by_phase: Dict[str, Dict[str, Dict[str, int]]] = {}
            totals = _UsageBucket()

            for phase, sources in self._data.items():
                by_phase[phase] = {}
                for source, bucket in sources.items():
                    by_phase[phase][source] = bucket.to_dict()

                    totals.prompt_tokens += bucket.prompt_tokens
                    totals.completion_tokens += bucket.completion_tokens
                    totals.total_tokens += bucket.total_tokens
                    totals.estimated_total_tokens += bucket.estimated_total_tokens
                    totals.call_count += bucket.call_count

            return {
                "by_phase": by_phase,
                "totals": totals.to_dict(),
            }

    def format_debug(self) -> str:
        snap = self.snapshot()
        lines = ["\n========== Token Usage Debug =========="]
        lines.append(
            "TOTAL: calls={call_count}, prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, estimated={estimated_total_tokens}".format(
                **snap["totals"]
            )
        )

        for phase, sources in snap["by_phase"].items():
            lines.append(f"-- phase={phase}")
            for source, data in sources.items():
                lines.append(
                    "   * {source}: calls={call_count}, prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, estimated={estimated_total_tokens}".format(
                        source=source, **data
                    )
                )

        lines.append("======================================")
        return "\n".join(lines)


_tracker = TokenUsageTracker()


def reset_token_usage() -> None:
    _tracker.reset()


def set_token_phase(phase: str) -> None:
    _tracker.set_phase(phase)


def record_token_usage(source: str, **kwargs: Any) -> None:
    _tracker.record(source, **kwargs)


def get_token_usage_snapshot() -> Dict[str, Any]:
    return _tracker.snapshot()


def print_token_usage_debug(prefix: str = "") -> None:
    msg = _tracker.format_debug()
    if prefix:
        _logger.info("%s%s", prefix, msg)
    else:
        _logger.info("%s", msg)
