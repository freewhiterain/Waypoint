# 知行旅行助手

基于 LangGraph 的企业级智能旅行规划系统。

## 快速启动

### 第零步：安装依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`dev` 可选依赖包含 pytest；只部署服务可以去掉 `[dev]`。

### 第一步：配置环境变量

编辑 `.env` 文件，填入必要的 API Key：

```
DASHSCOPE_API_KEY=sk-你的千问APIKey   # 必填
LANGSMITH_API_KEY=你的LangSmithKey    # 推荐
AMAP_API_KEY=你的高德Key              # 第7章需要
TAVILY_API_KEY=你的TavilyKey          # 第7章需要
```

DashScope 注册：https://dashscope.aliyun.com/

### 第二步：启动 Docker 数据库

```powershell
# PostgreSQL (pgvector)
docker run -d `
  --name travel_postgres `
  -e POSTGRES_DB=ai_travel_db `
  -e POSTGRES_USER=travel_user `
  -e POSTGRES_PASSWORD=travel123456 `
  -p 5432:5432 `
  --restart unless-stopped `
  pgvector/pgvector:pg17

# Redis
docker run -d `
  --name travel_redis `
  -p 6379:6379 `
  --restart unless-stopped `
  redis:7-alpine redis-server --appendonly yes
```

### 第三步：初始化数据库

```powershell
python scripts/init_db.py
```

### 第四步：测试 LLM 连接

```powershell
python scripts/test_llm.py
```

### 第五步：运行对话测试

```powershell
python -m pytest -q
```

### 第六步：初始化 RAG（可选，需要 DashScope Key）

```powershell
python scripts/init_rag.py
```

### 第七步：启动 FastAPI 服务

```powershell
python app/main.py
# UI: http://localhost:18000/ui
# API docs: http://localhost:18000/docs
```

## 项目结构

```
app/
├── core/          # Checkpointer、Store
├── agents/        # 对话协调器、Supervisor 图、Planner、Workers
├── governance/    # 事件、审批、偏好、行程草稿仓库
├── rag/           # 混合检索、向量存储
├── mcp_core/      # MCP 服务器和客户端
├── api/           # FastAPI 路由
├── models/        # SQLAlchemy ORM 模型
└── schemas/       # Pydantic 请求/响应模型
scripts/           # 初始化和测试脚本
tests/             # 测试文件
data/documents/    # RAG 文档库（放 .md 攻略文件）
```

## 技术栈

LangGraph 1.0 · LangChain 1.0 · FastAPI · PostgreSQL 17 + pgvector · Redis · ChromaDB · Qwen-Max · SSE

## 当前实现状态

每轮消息先由 Main Agent 路由：旅行规划进入表单 Tool，开放旅行问题进入 RAG，其余消息直接对话。新会话会主动询问是否需要规划旅行。

- 表单通过 Tool Call / Tool Result 交互，必须收集目的地、出发日期和旅行天数，不使用默认日期或默认天数。
- 目的地未定时可暂停表单并请求 RAG 推荐；选择城市后恢复原表单状态。
- 只有三项必填字段完整后才调用 Supervisor。Supervisor 调度景点、天气、交通、住宿和美食五类 Worker。
- Worker 的实时数据源设计暂缓；未配置可用数据源时必须明确降级，不生成虚构事实。
- **当前数据阶段（Phase 2）：仅成都 + 本地模拟 Markdown。** `data/documents/{attractions,weather,transport,accommodation,food}/chengdu.md` 是唯一的知识来源，不接入天气、地图、航班、铁路、酒店或餐厅的真实 API。Markdown 只是知识库原料，不是 Worker 的直接输出：每个 Worker 先按目的地和职责类别调用 `LocalKnowledgeService.search_destination` 做 RAG 检索，再把证据交给 `analyze_worker_evidence` 做结构化分析；没有证据支持的候选会被丢弃，不允许编造价格、班次、营业状态或天气事实。`WorkerResult.is_mock=True` 标记结果来自本地模拟资料，前端和草稿会保留该标记与证据来源。真实数据源接入和新的检索模式属于后续阶段。
- **本地知识图谱（轻量 GraphRAG）**：离线脚本 `scripts/build_knowledge_graph.py`
  从本地 Markdown 资料中抽取实体和"位于/临近"关系，写入 `knowledge_entity`/
  `knowledge_relation` 两张表；`attractions`、`hotel` 两个 Worker 会额外查询
  这层关系图，把结果作为 `source_type="graph_relation"` 的证据与文档证据合并
  使用。抽取只在离线脚本里发生，未配置 LLM 时仅产出规则抽取结果，不阻塞任何
  请求路径；图为空或查询异常时该 Worker 的行为与没有图谱时完全一致。
  `weather`/`transport`/`food` 三个 Worker 暂未接入图证据。
- 行程草稿常驻（`models/draft.py` + `governance/drafts.py`）：每个会话一份可增量编辑的草稿，版本递增；正式行程仍需审批落库。
- Hybrid RAG：稳定文档/切片 ID、BM25、Dense、RRF、相关性重排和父文档回溯。
- Evidence：来源、URL、查询时间、有效期、置信度和冲突检查。
- MCP/API 可靠性：超时、有限重试、请求去重、TTL 缓存、熔断和明确降级。
- LangGraph Checkpointer、任务事件持久化（含 task_failed）和用户隔离。
- 用户明确批准后才写入长期偏好或覆盖正式行程。
- 不提供购票、预订、支付、退款、取消、改签或外部消息发送。

没有配置实时数据源时，系统会明确说明数据不可用，不生成虚构班次、价格、库存或天气。旧 Handoffs 状态机与交通 Subagents 已退役。

## 关键配置

复制 `.env.example` 为 `.env`，至少配置独立的 `JWT_SECRET_KEY` 和 `DASHSCOPE_API_KEY`。实时数据需要显式设置 `ENABLE_EXTERNAL_TOOLS=true`，并配置相应的 `AMAP_API_KEY`、`TAVILY_API_KEY` 或 MCP 服务地址。

### CrossEncoder 重排（可选）

默认用词频重叠的 `RelevanceReranker`。设置 `ENABLE_CROSS_ENCODER_RERANK=true` 后改用 `app/rag/reranker.py` 的 `CrossEncoderReranker`，对同义改写更敏感。`CROSS_ENCODER_MODEL` 默认 `BAAI/bge-reranker-base`（约 1.1GB，中文效果和体积较均衡）；也可换 `BAAI/bge-reranker-large`（约 2.2GB，更准更慢）。首次调用会从 HuggingFace 下载模型到本地缓存，需要网络。

验证：`RUN_CROSS_ENCODER_TESTS=1 python -m pytest tests/test_cross_encoder_real_model.py -v`。

## 主要接口

- `POST /api/v1/chat/stream/{conversation_id}`：兼容旧客户端的 SSE 对话。
- `POST /api/v1/tasks`：提交结构化旅行需求并执行 Supervisor 规划。
- `GET /api/v1/tasks/{task_id}`：查询任务状态。
- `GET /api/v1/tasks/{task_id}/events`：读取持久化任务事件。
- `POST /api/v1/approvals/{approval_id}/decision`：提交 `approve`、`edit` 或 `reject`。
- `POST /api/v1/preferences/proposals`：提出长期偏好写入申请。
- `POST /api/v1/itineraries/{conversation_id}/save`：提出正式行程保存/覆盖申请。

## 测试

```powershell
\.venv\Scripts\python.exe -m pytest -q
```

默认会跳过需要真实模型和网络的集成测试；设置 `RUN_EXTERNAL_TESTS=1` 后再运行这些测试。启动服务前先执行 `python scripts/init_db.py` 初始化业务表、Checkpointer、Store 和 pgvector 扩展。
