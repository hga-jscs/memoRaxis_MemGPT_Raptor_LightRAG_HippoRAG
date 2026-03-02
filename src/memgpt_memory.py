# src/memgpt_memory.py
# -*- coding: utf-8 -*-
import os
import time
from typing import Any, Dict, List, Optional
import re
import requests

from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence

logger = get_logger()


class MemGPTMemory(BaseMemorySystem):
    """Letta(MemGPT) archival-memory backend for memoRaxis BaseMemorySystem."""

    def __init__(
        self,
        agent_name: str,
        base_url: Optional[str] = None,
        model_handle: Optional[str] = None,
        embedding_handle: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.agent_name = agent_name
        self.base_url = (base_url or os.getenv("LETTA_BASE_URL") or "http://127.0.0.1:8283/v1").rstrip("/")
        self.model_handle = model_handle or os.getenv("LETTA_DEFAULT_LLM_HANDLE") or "letta/letta-free"
        self.embedding_handle = embedding_handle or os.getenv("LETTA_DEFAULT_EMBEDDING_HANDLE") or "letta/letta-free"
        self.timeout = timeout

        # 1) wait server ready
        self._wait_ready()

        # 2) prefer explicit agent id from env; otherwise find/create by name
        override = os.getenv("LETTA_AGENT_ID")
        if override:
            # validate agent exists to fail fast with clear error
            try:
                self._get_agent_by_id(override)
            except Exception as e:
                raise RuntimeError(
                    f"LETTA_AGENT_ID is set but agent not found or not accessible: {override}. "
                    f"Please create the agent first or unset LETTA_AGENT_ID. ({e})"
                ) from e
            self.agent_id = override
        else:
            self.agent_id = self._get_or_create_agent_id()

    def _wait_ready(self, max_wait_s: float = 30.0):
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Letta server not ready: {self.base_url}")

    def _req(self, method: str, path: str, *, params=None, json=None) -> Any:
        url = f"{self.base_url}{path}"
        r = requests.request(method, url, params=params, json=json, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Letta API {r.status_code} {method} {path}: {r.text[:500]}")
        return None if r.status_code == 204 else r.json()

    def _get_agent_by_id(self, agent_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/agents/{agent_id}")

    def _find_agent(self) -> Optional[Dict[str, Any]]:
        # Try server-side filtering by name (if supported)
        try:
            r = requests.get(f"{self.base_url}/agents", params={"name": self.agent_name}, timeout=self.timeout)
            if r.status_code == 200:
                lst = r.json()
                if isinstance(lst, list) and lst:
                    return lst[0]
        except Exception:
            pass

        # Fallback: list all and filter client-side (more robust)
        try:
            r = requests.get(f"{self.base_url}/agents", timeout=self.timeout)
            if r.status_code != 200:
                return None
            lst = r.json()
            if isinstance(lst, list):
                for a in lst:
                    if a.get("name") == self.agent_name:
                        return a
        except Exception:
            pass

        return None

    def _create_agent(self) -> Dict[str, Any]:
        payload = {"name": self.agent_name, "model": self.model_handle, "embedding": self.embedding_handle}
        return self._req("POST", "/agents", json=payload)

    def _get_or_create_agent_id(self) -> str:
        a = self._find_agent()
        if a:
            return a["id"]
        logger.info(f"[MemGPTMemory] creating agent: {self.agent_name}")
        return self._create_agent()["id"]

    def reset(self) -> None:
        a = self._find_agent()
        if a:
            self._req("DELETE", f"/agents/{a['id']}")
        # after reset, always create a fresh agent with current handles
        self.agent_id = self._create_agent()["id"]

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        tags = []
        if metadata:
            if "instance_idx" in metadata:
                tags.append(f"inst_{metadata['instance_idx']}")
            if "chunk_id" in metadata:
                tags.append(f"chunk_{metadata['chunk_id']}")
            if "doc_id" in metadata:
                tags.append(f"doc_{metadata['doc_id']}")

        payload = {"text": data, "tags": tags or None}
        self._req("POST", f"/agents/{self.agent_id}/archival-memory", json=payload)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        try:
            resp = self._req(
                "POST",
                "/passages/search",
                json={
                    "query": query,
                    "agent_id": self.agent_id,
                    "limit": top_k,
                },
            )

            if isinstance(resp, list) and resp:
                out: List[Evidence] = []
                for rank, item in enumerate(resp[:top_k]):
                    passage = item.get("passage") or {}
                    out.append(
                        Evidence(
                            content=passage.get("text", ""),
                            metadata={
                                "source": "letta_passages_search",
                                "score_source": "letta",
                                "score": float(item.get("score", 0.0)),
                                "rank": rank,
                                "passage_id": passage.get("id"),
                                "tags": passage.get("tags", []),
                            },
                        )
                    )
                return out

        except Exception as e:
            logger.warning(f"[MemGPTMemory] /passages/search failed, fallback: {e}")

        resp = self._req(
            "GET",
            f"/agents/{self.agent_id}/archival-memory/search",
            params={"query": query, "top_k": top_k},
        )
        results = resp.get("results", []) if isinstance(resp, dict) else []

        def _lexical_score(q: str, text: str) -> float:
            """
            简单词覆盖率：score = 命中的 query 关键词数 / query 关键词总数
            - 只保留长度>=3 的字母数字 token，减少 of/in/the 这类噪声
            - 这是 fallback 的调试分数，不是语义相似度
            """
            q = (q or "").lower()
            text = (text or "").lower()
            tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) >= 3]
            if not tokens:
                return 0.0
            uniq = set(tokens)
            hits = sum(1 for t in uniq if t in text)
            return float(hits) / float(len(uniq))

        out: List[Evidence] = []
        for rank, item in enumerate(results[:top_k]):
            content = item.get("content", "")
            out.append(
                Evidence(
                    content=content,
                    metadata={
                        "source": "memgpt_letta_archival",
                        "score_source": "lexical_fallback",
                        "score": float(_lexical_score(query, content)),
                        "rank": rank,
                        "memory_id": item.get("id"),
                        "timestamp": item.get("timestamp"),
                        "tags": item.get("tags", []),
                    },
                )
            )
        return out
