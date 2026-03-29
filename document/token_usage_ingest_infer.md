# ingest / infer Token 消耗统计说明

## 结论
当前仓库**可以**对 ingest 和 infer 的 token 消耗进行统一统计，并且支持可视化调试输出（终端结构化打印）与落盘（JSON）。

## 本次实现内容

### 1) 统一 Token 统计器
新增 `src/token_tracker.py`：
- 支持阶段维度：`ingest` / `infer`
- 支持来源维度：`llm_chat` / `embedding` / `summarize` / `lightrag_internal_llm` 等
- 支持字段：
  - `call_count`
  - `prompt_tokens`
  - `completion_tokens`
  - `total_tokens`
  - `estimated_total_tokens`（用于网关未返回 usage 时）
- 支持调试输出：`print_token_usage_debug()`

### 2) infer 侧统计（LLM 生成）
在 `src/llm_interface.py` 中：
- `OpenAIClient.generate()` 优先使用 `response.usage` 记账
- 对不返回 usage 的兼容网关做估算记账
- `MockLLMClient` 也会记入估算 token

### 3) ingest 侧统计（embedding / summary / LightRAG 内部调用）
- `src/simple_memory.py`：embedding 调用记录 token
- `src/raptor_memory.py`：embedding + summarize 调用记录 token
- `src/lightrag_memory.py`：
  - LightRAG embedding 做估算记账
  - LightRAG 内部 llm 完成做估算记账

> 说明：LightRAG 上游接口通常不直接返回 usage，所以这里是**可解释估算值**，并显式计入 `estimated_total_tokens`。

### 4) 推理结果按“单题增量 token”统计
在 `src/adaptors.py` 中把 token 统计改成：
- 每次 `run()` 读取 `token_before` / `token_after`
- 返回本题增量 token（而非累计值）

### 5) 脚本级可视化输出与产物
已在以下脚本接入：
- `scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py`
- `scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py`

行为：
- ingest：结束后打印 token debug，并写入 `out/token_stats/ingest_lightrag_acc_ret_{instance}.json`
- infer：将 `token_usage` 写入最终结果 JSON，并打印 token debug

---

## 如何使用

### ingest 示例
```bash
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 850
```

### infer 示例
```bash
python scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit 5
```

执行后可查看：
- `out/token_stats/*.json`
- `out/acc_ret_results_*.json` 中的 `token_usage` 字段

---

## 调试输出解读
打印格式示例：
- `TOTAL`：总体调用次数和 token 总量
- `phase=ingest/infer`：阶段拆分
- 每个 source（如 `llm_chat`）展示该来源的调用和 token

当 `estimated_total_tokens > 0` 时，表示有部分来自估算（非 API 原生 usage）。

---

## 建议
若你希望把这套统计扩展到其它 `scripts/*_MAB/*` 脚本（例如 conflict/long_range/test_time），只需在脚本入口加入：
1. `reset_token_usage()`
2. `set_token_phase("ingest" or "infer")`
3. 结束时 `get_token_usage_snapshot()` + `print_token_usage_debug()`
