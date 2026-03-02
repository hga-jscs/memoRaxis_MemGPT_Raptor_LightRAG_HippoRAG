# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests

from src.memory_interface import BaseMemorySystem, Evidence
from src.config import get_config
from scipy.spatial import distance
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel
class _NoQAModel(BaseQAModel):
    def answer_question(self, *args, **kwargs):
        return ""



class _CompatEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        conf = get_config().embedding
        self.provider = conf.get('provider', 'openai_compat')
        self.base_url = conf.get('base_url')
        self.api_key = conf.get('api_key')
        self.model = conf.get('model', 'text-embedding-3-small')

        if self.provider != 'ark_multimodal':
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = None

    def create_embedding(self, text: str):
        text = text.replace('\n', ' ')
        if self.provider == 'ark_multimodal':
            url = (self.base_url or '').rstrip('/') + '/embeddings/multimodal'
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + str(self.api_key)}
            payload = {'model': self.model, 'input': [{'type': 'text', 'text': text}]}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()['data']['embedding']
        resp = self._client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding


class _CompatSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        conf = get_config().llm
        from openai import OpenAI
        self._client = OpenAI(api_key=conf.get('api_key'), base_url=conf.get('base_url'))
        self.model = conf.get('model')

    def summarize(self, context, max_tokens=180):
        # 1) 防止 max_tokens 意外为 0/None
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 180
        if max_tokens < 32:
            max_tokens = 32

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a careful summarizer.'},
                {'role': 'user', 'content': f'Summarize the following. Keep key entities, dates, decisions:\n{context}'},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )

        # 2) 主路径：标准 OpenAI-compat
        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""

        # 3) 兼容路径：某些网关会把内容放在“分段 content”里
        #    这里用 model_dump/dict 把原始结构拿出来再捞一遍
        if not content.strip():
            try:
                raw = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
                msg = raw.get("choices", [{}])[0].get("message", {})
                c = msg.get("content", "")

                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    # 常见形态：[{type: "...", text: "..."}] 或类似字段
                    buf = []
                    for part in c:
                        if isinstance(part, dict):
                            buf.append(part.get("text") or part.get("content") or "")
                        elif isinstance(part, str):
                            buf.append(part)
                    content = "".join(buf)
            except Exception:
                pass

        # 4) 最终兜底：如果仍为空，用原文截断代替，确保树节点不为空
        if not content.strip():
            # 这里截断长度你可以调，先确保非空最重要
            content = (context or "")[:2000]

        return content


class RaptorTreeMemory(BaseMemorySystem):
    def __init__(self, tree_path: Optional[str] = None, tb_num_layers: int = 3):
        self._buffer: List[str] = []
        emb = _CompatEmbeddingModel()
        summ = _CompatSummarizationModel()

        self._config = RetrievalAugmentationConfig(
            embedding_model=emb,
            summarization_model=summ,
            qa_model=_NoQAModel(),
            tb_num_layers=tb_num_layers,
            tb_max_tokens=200,
            tb_summarization_length=120,
            tr_threshold=0.5,
            tr_top_k=5,
        )
        self._ra = RetrievalAugmentation(config=self._config, tree=tree_path)

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        self._buffer.append(data)

    def build_tree(self) -> None:
        if self._ra.tree is not None:
            return
        text = '\n\n'.join(self._buffer)
        self._ra.add_documents(text)

    def save_tree(self, path: str) -> None:
        self.build_tree()
        self._ra.save(path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        self.build_tree()
        context, layer_info = self._ra.retrieve(
            question=query,
            top_k=top_k,
            collapse_tree=True,
            return_layer_information=True,
        )

        # 用 RAPTOR 同一套 embedding 给 query 做向量（保证一致）
        q_emb = self._ra.retriever.create_embedding(query)
        emb_key = getattr(self._ra.retriever, "context_embedding_model", None) or "EMB"

        evidences: List[Evidence] = []
        for item in layer_info:
            idx = int(item["node_index"])
            layer = int(item["layer_number"])
            node = self._ra.tree.all_nodes[idx]
            node_text = node.text

        # 取节点 embedding
            node_emb = None
            try:
                if isinstance(node.embeddings, dict):
                    node_emb = node.embeddings.get(emb_key)
                    if node_emb is None and node.embeddings:
                        node_emb = next(iter(node.embeddings.values()))
                else:
                    node_emb = node.embeddings
            except Exception:
                node_emb = None

        # 计算 cosine distance，再映射到 [0,1] 的相似度 score
        # scipy cosine distance: d = 1 - cos_sim, d∈[0,2]
        # 映射：score = 1 - d/2 ∈ [0,1]
            if node_emb is not None:
                d = float(distance.cosine(q_emb, node_emb))
                score = float(1.0 - d / 2.0)
            else:
                d = None
                score = 0.0

            evidences.append(
                Evidence(
                    content=node_text,
                    metadata={
                        "source": "RAPTOR",
                        "node_index": idx,
                        "layer": layer,
                        "score": score,
                        "distance": d,
                        "score_source": "cosine_score=1-d/2",
                        "emb_key": emb_key,
                    },
                )
            )

        if not evidences and isinstance(context, str) and context.strip():
            evidences.append(
                Evidence(
                    content=context,
                    metadata={"source": "RAPTOR", "node_index": -1, "layer": -1, "score": 0.0, "score_source": "fallback_context"},
                )
            )

        return evidences

    def reset(self) -> None:
        self._buffer.clear()
        self._ra = RetrievalAugmentation(config=self._config, tree=None)


