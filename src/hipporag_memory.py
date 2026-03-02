# -*- coding: utf-8 -*-
"""
HippoRAG 适配器（接入 memoRaxis 的统一 Memory 接口）

目标：
- 在 memoRaxis 的 BaseMemorySystem 接口下，封装 HippoRAG 的索引构建与检索
- 让现有 R1/R2/R3 推理范式（src/adaptors.py）可以无缝调用 HippoRAG 作为记忆后端

设计原则：
- 不改动 memoRaxis 现有 adaptor / benchmark 逻辑
- 通过独立的 build_index() 明确区分「ingest 阶段」与「infer 阶段」
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .config import get_config
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence


logger = get_logger()


def _add_hipporag_to_syspath(project_root: Path) -> None:
    """将 third_party/HippoRAG 的 src 路径加入 sys.path（兼容两种常见目录结构）"""
    candidates = [
        project_root / "third_party" / "HippoRAG" / "src",                 # third_party/HippoRAG/src
        project_root / "third_party" / "HippoRAG" / "HippoRAG" / "src",    # third_party/HippoRAG/HippoRAG/src
    ]

    for p in candidates:
        if p.exists() and (p / "hipporag").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
                logger.info("[HippoRAGMemory] Added HippoRAG src to sys.path: %s", p)
            return

    raise FileNotFoundError(
        "找不到 HippoRAG 的源码路径。请确认目录存在：\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}\n"
        "并且其中包含 hipporag/ 目录。"
    )


def _safe_rmtree(path: Path) -> None:
    """Windows 友好的递归删除（处理只读文件）"""
    if not path.exists():
        return

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)


class HippoRAGMemory(BaseMemorySystem):
    """HippoRAG 记忆体封装"""

    def __init__(
        self,
        index_dir: str,
        openie_mode: str = "online",
        force_rebuild: bool = False,
        top_k_default: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            index_dir: 每个 instance 的独立索引目录（强烈建议分开，避免互相污染）
            openie_mode: 'online' or 'offline'（本项目默认 'online'）
            force_rebuild: True 则删除旧索引并重建
            top_k_default: 默认返回证据数
            seed: 随机种子（保证一些内部采样稳定）
        """
        self._logger = get_logger()
        self._config = get_config()

        self.index_dir = Path(index_dir).resolve()
        self.openie_mode = openie_mode
        self.force_rebuild = force_rebuild
        self.top_k_default = top_k_default
        self.seed = seed

        # 1) 处理索引目录
        if self.force_rebuild:
            self._logger.warning("[HippoRAGMemory] force_rebuild=True, removing index_dir: %s", self.index_dir)
            _safe_rmtree(self.index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 2) 配置 OpenAI 环境变量（HippoRAG 内部使用 openai SDK）
        llm_conf = self._config.llm
        emb_conf = self._config.embedding

        api_key = llm_conf.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = llm_conf.get("base_url") or os.getenv("OPENAI_BASE_URL")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        # Embedding 如果单独配置，也同步给环境变量（很多 OpenAI-compatible 服务共用同一个 key/base_url）
        emb_key = emb_conf.get("api_key")
        emb_base_url = emb_conf.get("base_url")
        if emb_key:
            os.environ.setdefault("OPENAI_API_KEY", emb_key)
        if emb_base_url:
            os.environ.setdefault("OPENAI_BASE_URL", emb_base_url)

        # 3) 动态加入 third_party/HippoRAG 到 sys.path 并导入
        project_root = Path(__file__).resolve().parents[1]  # .../memoRaxis
        _add_hipporag_to_syspath(project_root)

        try:
            from hipporag import HippoRAG  # type: ignore
            from hipporag.utils.config_utils import BaseConfig  # type: ignore
        except Exception as e:
            raise ImportError(
                "HippoRAG 导入失败。常见原因：\n"
                "1) 未安装依赖：python-igraph / tqdm / numpy / openai / transformers 等\n"
                "2) third_party/HippoRAG 路径不正确\n"
                f"原始异常: {repr(e)}"
            ) from e

        # 4) 构建 HippoRAG 配置
        llm_model = llm_conf.get("model", "gpt-4o-mini")
        llm_base = llm_conf.get("base_url", None)

        emb_model = emb_conf.get("model", "text-embedding-3-small")
        emb_base = emb_conf.get("base_url", None)

        global_config = BaseConfig(
            llm_name=llm_model,
            llm_base_url=llm_base,
            embedding_model_name=emb_model,
            embedding_base_url=emb_base,
            openie_mode=self.openie_mode,
            save_dir=str(self.index_dir),
            force_index_from_scratch=self.force_rebuild,
            force_openie_from_scratch=self.force_rebuild,
            seed=self.seed,
        )

        # 5) 初始化 HippoRAG（若索引已存在且 force_* 为 False，会自动复用磁盘缓存）
        self._hippo = HippoRAG(global_config=global_config)

        # ingest 阶段缓冲区（存 chunks）
        self._buffer: List[str] = []

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        """添加记忆（暂存到 buffer，build_index 时一次性构建）"""
        if not isinstance(data, str) or not data.strip():
            return
        self._buffer.append(data)

    def build_index(self) -> None:
        """构建 HippoRAG 索引（ingest 阶段调用一次）"""
        if not self._buffer:
            self._logger.warning("[HippoRAGMemory] No chunks to index. build_index() skipped.")
            return

        self._logger.info("[HippoRAGMemory] Building index with %d chunks -> %s", len(self._buffer), self.index_dir)
        self._hippo.index(self._buffer)
        self._logger.info("[HippoRAGMemory] Index build done.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """检索，返回 Evidence 列表供 adaptor 使用"""
        k = top_k or self.top_k_default
        if not query:
            return []

        try:
            out = self._hippo.retrieve([query], num_to_retrieve=k)
        except TypeError:
            # 兼容 HippoRAG 可能的参数签名变化
            out = self._hippo.retrieve([query], k)

        # retrieve 可能返回 solutions 或 (solutions, meta)
        if isinstance(out, tuple) and len(out) == 2:
            solutions = out[0]
        else:
            solutions = out

        if not solutions:
            return []

        sol = solutions[0]
        docs = getattr(sol, "docs", []) or []
        scores = getattr(sol, "doc_scores", None)

        evidences: List[Evidence] = []
        for i, doc in enumerate(docs[:k]):
            score_val: float = 0.0
            if scores is not None:
                try:
                    score_val = float(scores[i])
                except Exception:
                    score_val = 0.0

            evidences.append(
                Evidence(
                    content=doc,
                    metadata={
                        "source": "HippoRAG",
                        "rank": i + 1,
                        "score": score_val,
                        "index_dir": str(self.index_dir),
                    },
                )
            )

        return evidences

    def reset(self) -> None:
        """清空 ingest buffer（不删除磁盘索引）"""
        self._buffer = []