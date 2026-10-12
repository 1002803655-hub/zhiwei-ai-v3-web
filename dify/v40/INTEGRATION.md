# Lovis V4.0 实验集成（AL-01）

## 架构与职责

1. 独立 Streamlit 页面接收事件 JSON。
2. `line_fault_locator.locate_faults(...)` 在 Streamlit/Python 后端本地执行，唯一负责规则评分。
3. `lovis_v40/integration.py` 将候选、评分分解、关联部件、支持/矛盾事件、缺失数据和状态传给 Dify 管理Agent。
4. 管理Agent输出受限 JSON 路由决策。后端校验路由名称后，调用对应的独立 Dify 专家工作流；管理Agent不计算或重排分数。
5. 专家工作流返回结构化证据复核和检查建议，管理Agent再生成带不确定性的摘要。

评分是实验规则点数，不是概率。工位候选和共享控制器候选分开；部件作为工位候选的关联信息。没有独立现场验证时 `verified_fault` 必须保持 `false`。本次安全修订版本：定位参数集 `AL-01-exp-1.1`，事件格式 `lovis-v40-event-1.0`，Dify提示词契约 `v1.1`。PLC联锁是保护动作，不单独计作PLC本机异常；重复同设备同周期的本机异常点受配置上限约束。告警码必须与字典指定的设备和信号匹配。

## 导入Dify实验DSL

三份DSL均按当前Dify DSL导入器可接受的应用包顶层结构检查：`app`、`workflow`、`version`、`kind: app`；管理Agent采用 `advanced-chat`，两个专家采用 `workflow`。DSL版本为 `0.3.0`。这只能核对包结构与仓库节点配置，不能替代租户实际导入校验。模型节点目前填写 `openai / gpt-4o-mini`；因本会话无法访问用户Dify租户，不能核实该租户的供应商连接、模型可用性或套餐权限。导入时若该模型不在租户的已配置模型列表，需在新实验应用中改选租户可用模型并保存，不要覆盖正式应用。

依次将以下文件通过 Dify Studio 的 DSL 导入功能导入为**新应用**，不要选择覆盖现有应用：

- `management_agent.dsl.yml`：V4.0管理Agent实验应用。
- `robot_expert_workflow.dsl.yml`：机器人/真空夹爪专家工作流。
- `motion_control_expert_workflow.dsl.yml`：运动控制专家工作流。

导入后为每个应用选择当前租户已配置的模型供应商/模型，保存并使用平台测试运行验证各自的输入输出。机器人及运动控制工作流先提供可导入的结构化复核模板；需按现场维护规范补充专家知识和检查流程。压装与视觉分支目前仅预留配置键和Agent路由接口，尚无工作流DSL。

此DSL尚未在用户Dify租户导入或运行验证：当前会话未发现可访问的浏览器标签，浏览器扩展连接读取失败。现有Lovis应用也未能打开，因此本次没有导出现有DSL备份、创建平台副本、配置模型/API Key或发布任何应用。导入后请确认是新实验应用且旧应用未被覆盖。

## 配置后端环境变量

在运行 Streamlit 的服务器环境中配置（不要写入仓库或浏览器前端）：

- `DIFY_BASE_URL`：租户Dify API根地址。
- `LOVIS_V40_MANAGER_API_KEY`：独立管理Agent应用的API Key。
- `LOVIS_V40_ROBOT_EXPERT_API_KEY`：机器人专家工作流应用的API Key。
- `LOVIS_V40_MOTION_EXPERT_API_KEY`：运动控制专家工作流应用的API Key。
- 可选预留：`LOVIS_V40_PRESS_EXPERT_API_KEY`、`LOVIS_V40_VISION_EXPERT_API_KEY`。

应用 API Key 必须只保存在服务器环境/密钥管理器。客户端不接触Key。`DIFY_BASE_URL` 支持 HTTPS；只有 `localhost`/`127.0.0.1` 实验允许 HTTP。HTTP错误只记录状态码，不输出认证头或Key。

启动独立实验入口：

```powershell
streamlit run lovis_v40/streamlit_app.py
```

测试时选择默认T02模拟数据。平台调用要求管理应用可以通过 `/v1/chat-messages` 访问，专家工作流可通过 `/v1/workflows/run` 访问；需使用各自应用的API Key。没有配置Key时页面只返回本地定位结果和 `dify_not_configured` 状态，不声称已经连接。

事件字段契约为 `lovis-v40-event-1.0`（event_id、line_id、cycle_id、带时区ISO 8601 timestamp、device_id、signal、value、unit、quality、data_source，可选 alarm_code）；拓扑版本见 `topology.json` 的 `schema_version`，信号字典版本见 `locator_config.json` 的 `signal_dictionary_schema_version`。所有样例数据标注 `data_source: simulation`。

## T02端到端实验记录

本仓库包含 T02 模拟事件夹具 `tests/fixtures/t02_vacuum_gripper_simulation.json`。本地端到端协议测试使用替身Dify客户端验证：

- 管理Agent dispatch 决策为 `robot_expert`。
- 专家调用路由为 `robot_expert`，输入包含 locator 结构和 `grip`、`short` 等关联事件。
- 专家返回 `candidate_review`、观察和检查建议，`verified_fault=false`。
- 管理Agent摘要保留不确定性，`verified_fault=false`。
- 专家不可用会返回 `expert_unavailable`，不伪造专家结果；定位数据不足时不调用Dify。

这属于使用模拟Agent响应的端到端接口测试，不代表真实模型决策、Dify租户调用或专家知识已经通过平台验证。

## 回归状态

V3.1分支基础提交 `7528328` 的现有 GitHub Actions 第72次运行状态为成功，步骤包含依赖安装、Python编译和 `test_lovis_*.py` 测试。该工作流只监听 `lovis-v31-dev` 推送及针对 `main` 的PR；V4.0提交不会触发它。由于此环境无法克隆GitHub仓库，V3.1回归测试未在本次改动检出上重跑。勿将基础分支CI当作本次V4.0回归证据。

