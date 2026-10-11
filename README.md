# 智维AI V3.0 — Dify Agent 接入补丁

更新四个文件到 GitHub 仓库根目录：`app.py`, `dify_bridge.py`, `requirements.txt`；README 为说明，非必须上传。

本补丁基于上一版已完成 Random Forest 集成的 app.py 制作，不需要改动模型、特征算法、CWRU样本下载、其他文件。

## 部署

1. 先在 GitHub 仓库上传 `app.py`, `dify_bridge.py`, `requirements.txt`，覆盖已有同名文件。
2. Streamlit Cloud 的应用页面右下角 **Manage app** → 菜单 **Settings** → **Secrets**，添加两行 TOML 配置（如果有原配置，保留并在末尾追加）：

   ```toml
   DIFY_API_KEY = "app-这里填自己刚创建的完整密钥"
   DIFY_API_BASE = "https://api.dify.ai/v1"
   ```

3. 保存后等待云端重新部署/重启。请勿提交 `.streamlit/secrets.toml`、API密钥或含密钥的截图。
4. 打开公网地址、选择105.mat→开始分析→勾选同意传送脱敏摘要→点击“使用Dify生成专业解释”，确认出现实际生成的文本。
5. 下载HTML报告，检查是否包含AI解释段落；前往Dify日志核实本次是否确实使用了`motor_knowledge_search`。

## 隐私和可靠性

- 只发送提取的少量数值、投票结果和必要的来源类型标记；不发送原始文件、采样序列、文件名或SHA-256。
- 使用Dify Chatflow `POST /chat-messages`的streaming模式；服务端解析SSE，检测异常/结束。不会在网页或日志中打印API key。
- API失败不影响原有的波形、FFT、分类和报告下载。HTML报告包含AI解释（如果实际请求成功），文字经过HTML转义。
- Streamlit公开应用在服务端持有密钥，存在被外部用户反复发起请求造成费用的风险。当前示例仅做每个Streamlit会话最多3次软限制，**不是严格防刷机制**。参赛前应核查Dify配额/预算并监控用量；若流量较高，应使用有认证与全局限流的托管方案。
- 不保证Dify知识工具每次都实际执行；最终须以Dify应用日志和现场安全边界核验。
