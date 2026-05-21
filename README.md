# Enterprise Knowledge Agent

面向企业知识库的可靠协作 Agent。项目围绕企业内部复杂业务文档问答与多步任务处理，重点实现可追溯回答、工具权限控制、长期记忆治理、Prompt Injection 防护和离线评测。

## 项目定位

传统 RAG 问答通常只关注“能不能检索到相关文档”。真实企业场景更复杂：业务问题会跨产品、销售、法务、财务、客服、安全和运营多个部门；答案还会受到版本差异、审批链路、合规话术、客户赔付和外部不可信材料影响。

本项目将知识库问答扩展为一个可靠 Agent Runtime：

- 基于 ReAct 工具调用循环组织任务执行。
- 使用 Todo List、步骤状态机和 trace 记录多步任务链路。
- 通过工具注册表、权限模式和 hook 机制约束工具调用。
- 将 RAG/Web 内容作为不可信证据处理，避免外部文本变成系统指令。
- 使用长期记忆类型、写入门控、冲突记录和召回预算降低记忆污染。
- 通过离线 eval 和单元测试验证检索、安全、引用和记忆策略。

## 核心能力

### 企业知识库问答

- 支持本地文档上传、分块、索引和混合检索。
- 检索链路包含 FAISS dense retrieval、BM25 sparse retrieval 和 RRF 融合。
- 回答会保留引用来源，便于追溯证据。
- 内置一套复杂企业知识库 fixture，用于验证跨部门业务问答。

### Reliable Agent Runtime

- 显式 `AgentRun`、`PlanStepState`、`ToolCallRecord` 和 `TraceEvent`。
- `RuntimeMode` 支持 explore、plan、execute、review。
- `PermissionMode` 支持 plan-only、read-only、ask、auto。
- Plan 会转为 Todo List，由 ReAct loop 按依赖和权限逐步执行。
- 每个节点、步骤、工具调用和最终回答都可以进入 trace。

### 工具治理

- `ToolRegistry` 统一维护工具名称、描述、风险等级、权限范围和 backend。
- `ToolDispatcher` 统一分发本地工具、MCP 工具和未来子 Agent 工具。
- 高风险工具支持确认策略。
- 未配置的 MCP 工具返回结构化 unavailable 状态，而不是静默失败。

### 安全防护

- RAG/Web 内容被包装为不可信上下文，只作为证据，不作为指令。
- Prompt Injection detector 会识别常见注入意图。
- 风险信号会沿工具调用链传播，并阻断由不可信上下文触发的高风险动作。
- Web fetch 覆盖 URL、重定向、响应类型和响应大小边界。

### 记忆系统

- 短期记忆支持滑动窗口和历史摘要压缩。
- 长期记忆区分 preference、constraint、research_summary、episode、hypothesis、conflict。
- Memory write gate 避免闲聊、澄清、低质量回答污染长期记忆。
- 显式偏好/约束抽取器只记录用户明确表达的长期偏好。
- 记忆冲突会进入 unresolved conflict，并支持用户确认采用旧记忆、采用新记忆或保留两者。
- 记忆召回预算按类型控制进入上下文的内容，避免单一类型挤占上下文。

### 离线评测

项目内置确定性 eval，尽量避免依赖外部模型和网络：

- security eval
- KB lexical retrieval eval
- RAG pipeline eval
- answer validation eval
- memory write eval
- memory conflict eval
- memory retrieval budget eval

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置模型 Key

```bash
export DEEPSEEK_API_KEY="your-key"
```

### 命令行使用

```bash
python cli.py chat
```

上传文档并进入对话：

```bash
python cli.py upload paper.pdf --chat
python cli.py upload notes.md --strategy structured
python cli.py index ./docs/
python cli.py status
```

### Web 服务

```bash
python main.py
```

默认访问：

```text
http://localhost:8000
```

API 示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "增长协作版低于85折需要哪些审批？", "session_id": "demo"}'
```

Trace 查询：

```bash
curl "http://localhost:8000/runs?limit=10"
curl "http://localhost:8000/runs/<run_id>"
```

## 评测与验证

运行全部离线评测：

```bash
python evals/run_all.py --no-report
```

运行单元测试：

```bash
python -m unittest discover -s tests
```

运行企业知识库复杂场景检索评测：

```bash
python evals/run_kb_eval.py \
  --kb-dir evals/fixtures/enterprise_knowledge_base \
  --cases evals/cases/enterprise_kb_cases.jsonl \
  --top-k 3 \
  --no-report
```

如果本地环境包含 FAISS，可以验证 RAG pipeline：

```bash
python evals/run_rag_pipeline_eval.py --no-report
```

## 企业知识库 Fixture

`evals/fixtures/enterprise_knowledge_base/` 提供一套虚构企业“云舟企服”的复杂业务文档，覆盖：

- 产品套餐
- 华东中小客户销售政策
- 法务合同合规
- 客户成功上线 SOP
- 客服 FAQ 与升级
- 安全与数据治理
- 竞品和客户反馈
- 财务账期与退款
- 版本冲突说明
- 运营上线 RACI
- 外部合作方恶意材料

这套 fixture 用来验证 Agent 是否能处理跨部门、跨版本、审批边界、合规承诺、上线阻断和 prompt injection 对抗等企业知识库真实难点。

## 项目结构

```text
.
├── agent/                  # Agent runtime、orchestrator、tool policy、hooks、trace
├── memory/                 # 短期记忆、长期记忆、写入门控、冲突和召回预算
├── rag/                    # 文档分块、混合检索、rerank、引用追踪、Web search
├── security/               # Prompt injection、URL safety、不可信上下文
├── evals/                  # 离线评测用例、fixture 和 runner
├── tests/                  # 单元测试和回归测试
├── static/                 # Web 前端
├── cli.py                  # CLI 入口
├── main.py                 # FastAPI 入口
└── config.py               # 全局配置
```

## 技术栈

- Python
- FastAPI
- FAISS
- rank-bm25
- sentence-transformers
- OpenAI-compatible LLM client
- SQLite trace store

