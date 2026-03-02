# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

import numpy as np
import requests

from .logger import get_logger
from .config import get_config
from .memory_interface import BaseMemorySystem, Evidence
from lightrag.utils import always_get_an_event_loop
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed, openai_complete_if_cache

logger = get_logger()


@dataclass
class _Buffers:
    chunks: List[str]
    full_parts: List[str]


class LightRAGMemory(BaseMemorySystem):
    """
    LightRAG backend for memoRaxis BaseMemorySystem.

    设计目标：
    - add_memory：仅做缓冲（与 memoRaxis chunk_context 的切块保持一致）
    - build_index：一次性写入 LightRAG 工作区（insert_custom_chunks）
    - retrieve：只做“检索返回证据”，不让 LightRAG 生成最终回答（回答由 R1/R2/R3 adaptor 生成）
    """

    def __init__(self, working_dir: str, mode: str = "naive"):
        self.working_dir = str(Path(working_dir))
        self.mode = mode

        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self._buf = _Buffers(chunks=[], full_parts=[])

        conf = get_config()
        llm_conf = conf.llm
        emb_conf = conf.embedding

        # -----------------------------
        # 1) Embedding function
        # -----------------------------
        emb_provider = str(emb_conf.get("provider", "openai")).lower()
        emb_model = str(emb_conf.get("model", "text-embedding-3-small"))
        emb_base_url = emb_conf.get("base_url")
        emb_api_key = emb_conf.get("api_key")
        emb_dim = int(emb_conf.get("dim", 1536))

        # LightRAG 的 openai_embed 是 async 且带装饰器；partial 时必须用 openai_embed.func 避免双重包装
        if emb_provider == "ark_multimodal":
            # 兼容你项目里常见的 ark_multimodal embedding 端点：POST {base_url}/embeddings/multimodal
            async def _ark_embed(texts: list[str], **kwargs) -> np.ndarray:
                url = (emb_base_url or "").rstrip("/") + "/embeddings/multimodal"
                if not url.startswith("http"):
                    raise ValueError(
                        f"[LightRAGMemory] embedding.base_url 非法（ark_multimodal 需要 http(s)）：{emb_base_url}"
                    )
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + str(emb_api_key),
                }
                payload = {
                    "model": emb_model,
                    "input": [{"type": "text", "text": (t or "").replace("\n", " ")} for t in texts],
                }
                r = requests.post(url, headers=headers, json=payload, timeout=90)
                r.raise_for_status()
                j = r.json()

                data = j.get("data")
                if isinstance(data, list):
                    embs = [item.get("embedding") for item in data]
                elif isinstance(data, dict) and "embedding" in data:
                    embs = [data["embedding"]]
                else:
                    raise RuntimeError(f"[LightRAGMemory] ark_multimodal 返回格式异常：keys={list(j.keys())}")

                if not embs or any(e is None for e in embs):
                    raise RuntimeError("[LightRAGMemory] ark_multimodal 未返回 embedding")

                arr = np.asarray(embs, dtype=np.float32)
                return arr

            embedding_func = EmbeddingFunc(
                embedding_dim=emb_dim,
                max_token_size=8192,
                model_name=emb_model,
                func=_ark_embed,
            )
        else:
            # openai / openai_compat：走 LightRAG 自带的 openai_embed
            _embed = partial(
                openai_embed.func,
                model=emb_model,
                base_url=str(emb_base_url) if emb_base_url else None,
                api_key=str(emb_api_key) if emb_api_key else None,
            )

            embedding_func = EmbeddingFunc(
                embedding_dim=emb_dim,
                max_token_size=getattr(openai_embed, "max_token_size", 8192),
                model_name=emb_model,
                func=_embed,
            )

        # -----------------------------
        # 2) LLM function（给 LightRAG 做实体/关系摘要用）
        # -----------------------------
        llm_model = str(llm_conf.get("model", "gpt-4o-mini"))
        llm_base_url = llm_conf.get("base_url")
        llm_api_key = llm_conf.get("api_key")
        llm_timeout = int(llm_conf.get("timeout", 120))

        async def _llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            # LightRAG 会额外塞一些 kwargs（如 _priority 等），这里直接透传
            return await openai_complete_if_cache(
                model=llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=str(llm_base_url) if llm_base_url else None,
                api_key=str(llm_api_key) if llm_api_key else None,
                timeout=llm_timeout,
                **kwargs,
            )

        # -----------------------------
        # 3) LightRAG instance
        # -----------------------------
        self._rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=embedding_func,
            llm_model_func=_llm_func,
            llm_model_name=llm_model,
            # 显式指定存储后端，避免版本差异导致默认值变化
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
        )
    def _ensure_ready(self) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self._rag.initialize_storages())
    # -----------------------------
    # BaseMemorySystem API
    # -----------------------------
    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        # metadata 这里先不强依赖；后续需要可写入 doc_id/chunk_id
        s = (data or "").strip()
        if not s:
            return
        self._buf.chunks.append(s)
        self._buf.full_parts.append(s)

    def build_index(self, doc_id: Optional[str] = None) -> None:
        self._ensure_ready()
        """
        非接口必需，但你的 ingest 脚本会用到。
        用 insert_custom_chunks 保持 chunks 与 memoRaxis 的 chunk_context 完全一致。
        """
        if not self._buf.chunks:
            logger.warning("[LightRAGMemory] build_index: buffer is empty, skip.")
            return

        full_text = "\n\n".join(self._buf.full_parts) if self._buf.full_parts else "\n\n".join(self._buf.chunks)

        # 同步接口（内部会 run_until_complete）
        self._rag.insert_custom_chunks(full_text=full_text, text_chunks=self._buf.chunks, doc_id=doc_id)

        self._buf.chunks.clear()
        self._buf.full_parts.clear()

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        直接走底层 chunks_vdb.query()，拿到 distance(=cosine similarity) 作为 score，
        这样 adaptor 日志里的 (score=...) 就不会是 N/A 了。
        """
        self._ensure_ready()

        q = (query or "").strip()
        if not q:
            return []

        k = max(int(top_k), 1)

        # 直接查向量库，返回包含 distance 字段
        loop = always_get_an_event_loop()
        try:
            results = loop.run_until_complete(self._rag.chunks_vdb.query(q, top_k=k))
        except Exception as e:
            logger.error(f"[LightRAGMemory] chunks_vdb.query failed: {e}")
            return []

        evidences: List[Evidence] = []
        for rank, r in enumerate(results[:k]):
            content = r.get("content", "") or ""
            chunk_id = r.get("id") or r.get("chunk_id") or ""
            file_path = r.get("file_path", "") or ""
            created_at = r.get("created_at", None)

            raw_score = r.get("distance", None)  # 注意：NanoVectorDBStorage 这里叫 distance，但语义是 cosine 相似度
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except Exception:
                score = 0.0

            evidences.append(
                Evidence(
                    content=content,
                    metadata={
                        "source": "LightRAG",
                        "rank": rank,
                        "score": score,          # ✅关键：给 adaptor 打日志用
                        "metric": "cosine_sim",  # 便于你后面排查
                        "chunk_id": chunk_id,
                        "file_path": file_path,
                        "created_at": created_at,
                    },
                )
            )

        return evidences

    def reset(self) -> None:
        """
        清空 LightRAG 工作区（用于重复实验）。
        注意：会删除 working_dir 下所有内容。
        """
        p = Path(self.working_dir)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        self._buf.chunks.clear()
        self._buf.full_parts.clear()