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
LLM_API_KEY=sk-你的模型APIKey         # 必填
LLM_MODEL=deepseek-v4-flash           # 默认值，按供应商调整
LLM_BASE_URL=https://api.deepseek.com # 默认值，OpenAI 兼容接口即可
LANGSMITH_API_KEY=你的LangSmithKey    # 推荐
ENABLE_EXTERNAL_TOOLS=true            # 打开后才会调用高德和联网搜索
AMAP_API_KEY=你的高德Key              # 天气、地点间耗时
TAVILY_API_KEY=你的TavilyKey          # 联网搜索
```

DeepSeek 注册：https://platform.deepseek.com/

对话模型只要求 OpenAI 兼容接口，换供应商改这三个变量即可，不用动代码。
注意 `LLM_DISABLE_THINKING`：DeepSeek 的 thinking 模式不支持强制 `tool_choice`，
而结构化输出正是靠它，保持 `true` 才能让路由、需求抽取、子 Agent 分析和行程合成
拿到结构化结果。

### 第一步之二：启动本地 Ollama（Embedding 用）

Embedding 走本地 Ollama，不依赖任何外部 API Key，也不受对话模型供应商影响：

```powershell
ollama pull qwen3-embedding:4b
```

未启动 Ollama 时 Dense 检索会退化成纯 BM25（有日志告警，不会中断请求）。

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

### 第六步：初始化 RAG（可选，需先启动 Ollama 并拉取 embedding 模型）

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
├── agents/        # Supervisor 图、子 Agent、排日程、交通耗时
├── services/      # 主 Agent（意图 + 槽位）、ReAct 问答、需求提取
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

LangGraph 1.0 · LangChain 1.0 · FastAPI · PostgreSQL 17 + pgvector · ChromaDB · DeepSeek（OpenAI 兼容）· 本地 Ollama Embedding · SSE

## 当前实现状态

### 对话：两条线路 + 三个槽位

每一轮消息先由主 Agent 调用一次模型，同时得到两样东西：这句话的意图，以及从这句话里能填的槽位。

- **三个必填槽位**：目的地、出发日期、天数。另外还会记人数、出发地、兴趣、饮食偏好。槽位存在会话上（`conversation.extra_info["trip_slots"]`），不管这一轮走哪条线路都会累积，前面聊到的信息在后面规划时都能用上。
- **模型填的值要过一遍代码校验**（`app/services/slots.py`）：目的地必须是用户说过的，或是用户明确接受的推荐；"国庆""春节"这类假期词不会被换算成日期；天数必须有"数字 + 时间单位"作依据。校验不过的值直接丢弃。
- **线路 1：问答**（开放问题、推荐目的地、单项研究）。走 ReAct 循环（`app/services/react_agent.py`），模型自己决定查什么，最多 4 轮工具调用。三个只读工具：查本地旅行资料、对某城市某方面做专项研究、查同城几个地点之间的交通耗时。
- **线路 2：规划**。用户想规划时：三个槽位有缺的，表单只问缺的那几项；三个都齐了，先显示"将为你规划：目的地 · 日期 · 天数"，用户确认后才开始。
- 没配 `LLM_API_KEY` 或模型出错时，意图判断退回关键词规则，槽位退回正则提取，线路 1 退回固定的检索问答。

### 规划：四个阶段

```
planner → dispatch ⇄ run_worker → route_planner → synthesize
```

1. **研究**。五个子 Agent（`app/agents/subagents/`）分两批跑：景点、住宿、美食、天气先跑；交通后跑，因为它要用前面查到的地点。每个子 Agent 先取证据，再让模型做结构化分析；证据不支持的地点整条丢弃，证据不支持的描述、价格、属性只去掉那一项。
2. **交通**。交通子 Agent 查的是城市内的交通：把景点、住宿、餐饮两两配对，算出每一段的耗时和距离。数据来源按顺序是：交通图谱（表 `travelroute`）→ 高德 → 按"是否同一个区"粗略判断。高德查到的结果会写回图谱，下次直接用。没有真实数据时只给"同区 / 跨区"的判断，不编分钟数。
3. **排日程**（`app/agents/itinerary_planner.py`）。模型从候选里按 id 选点、选落脚的住宿；分天和先后顺序由算法按交通耗时排；排完检查开放时间和每天的路上时间，不合格就把原因反馈给模型重选，最多 3 轮。没配模型时用规则选点。
4. **成文**。模型在排好的行程上改写每个时段的描述，不能更换地点或调换顺序。每天的天气直接取自天气证据，显示在当天行程的第一行。

### 数据来源

- **本地资料**：`data/documents/` 下的 Markdown，目前只有成都有完整资料，而且是开发用的模拟数据（结果里 `is_mock=True`）。其他城市没有景点资料，行程里会是"自由活动"。
- **高德**（需要 `ENABLE_EXTERNAL_TOOLS=true` 和 `AMAP_API_KEY`）：天气预报、地点之间的驾车耗时和距离。
- **联网搜索**（需要 `ENABLE_EXTERNAL_TOOLS=true` 和 `TAVILY_API_KEY`）：用户要求"深入研究"时使用。
- **本地知识图谱**：离线脚本 `scripts/build_knowledge_graph.py` 从本地资料里抽取实体和"位于 / 临近"关系，写入 `knowledgeentity`、`knowledgerelation` 两张表；景点和住宿两个子 Agent 会把这层关系作为额外证据。交通图谱 `travelroute` 的两端也是这里的实体。
- 没有可用数据源时，系统会明确说明查不到，不生成虚构的班次、价格、营业时间或天气。

### 其他

- 预算功能已整体移除，等重新设计。候选上的价格字段 `estimated_cost` 保留。
- 行程草稿常驻（`models/draft.py` + `governance/drafts.py`）：每个会话一份可增量编辑的草稿，版本递增；正式行程仍需审批落库。
- Hybrid RAG：稳定文档/切片 ID、BM25、Dense、RRF、相关性重排和父文档回溯。
- Evidence：来源、URL、查询时间、有效期、置信度和冲突检查。
- MCP/API 可靠性：超时、有限重试、请求去重、TTL 缓存、熔断和明确降级。
- LangGraph Checkpointer、任务事件持久化（含 task_failed）和用户隔离。
- 用户明确批准后才写入长期偏好或覆盖正式行程。
- 不提供购票、预订、支付、退款、取消、改签或外部消息发送。

## 关键配置

复制 `.env.example` 为 `.env`，至少配置独立的 `JWT_SECRET_KEY` 和 `LLM_API_KEY`。Embedding 走本地 Ollama（`EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`），不需要外部 Key。实时数据需要显式设置 `ENABLE_EXTERNAL_TOOLS=true`，并配置相应的 `AMAP_API_KEY`、`TAVILY_API_KEY` 或 MCP 服务地址。

### CrossEncoder 重排（可选）

默认用词频重叠的 `RelevanceReranker`。设置 `ENABLE_CROSS_ENCODER_RERANK=true` 后改用 `app/rag/reranker.py` 的 `CrossEncoderReranker`，对同义改写更敏感。`CROSS_ENCODER_MODEL` 默认 `BAAI/bge-reranker-base`（约 1.1GB，中文效果和体积较均衡）；也可换 `BAAI/bge-reranker-large`（约 2.2GB，更准更慢）。首次调用会从 HuggingFace 下载模型到本地缓存，需要网络。

验证：`RUN_CROSS_ENCODER_TESTS=1 python -m pytest tests/test_cross_encoder_real_model.py -v`。

## 主要接口

- `POST /api/v1/chat/stream/{conversation_id}`：SSE 对话。每一轮在这里判断意图、填槽位。
- `POST /api/v1/chat/tools/{call_id}/result`：提交需求表单或确认结果，触发规划并以 SSE 返回行程。
- `GET /api/v1/drafts/{conversation_id}`：读取会话当前的行程草稿。
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
