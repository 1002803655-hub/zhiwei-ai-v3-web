# 智维AI V3.0 · 工业电机振动分析网页（第一阶段）

**目前仅支持真实振动信号时域/FFT分析与可追溯报告；尚无故障分类模型、没有Dify集成。**

## 在线部署

1. 把本目录内的文件（不是最外层文件夹）上传到公开GitHub仓库根目录。
2. 前往 <https://share.streamlit.io/>，登录GitHub，选择仓库的`main`分支及入口`app.py`。
3. 等待安装依赖、网页启动。**第一次访问会由服务器向CWRU官方链接获取4份MAT文件**，生成临时内置样本；可能需1分钟。
4. 浏览器里选内置示例或上传合法MAT文件，点击“开始分析”。若官方端点暂时不可用，上传MAT文件模式仍可使用。

> 不将原始CWRU数据或剪裁重封装数据提交到GitHub，避免在再分发许可未查明前公开重新分发。远程下载行为遵循官方站点要求，并标注数据来源；如官方站点限制自动下载，请遵守其要求并取得许可。

## 数据与能力边界

- CWRU: <https://engineering.case.edu/bearingdatacenter/download-data-file>
- 97.mat正常源记录48kHz；105/118/130.mat为12kHz，程序统一处理为12kHz。
- 本阶段`正常/内圈/滚动体/外圈`均为官方已知标签，不是模型的预测。
- 4096点窗口计算RMS、峭度、峰值因子及FFT，数值属于描述性统计，不是故障确诊/报警指标。
- 公开实验台数据，不代表现场工业电机诊断准确率。

## 本地启动（Python 3.10+）

`py -3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

`py -3 -m streamlit run app.py`

首次运行需要从学校网站下载原始MAT文件；已有的MAT文件也可以事先放进`data`目录，或者从网页直接上传MAT进行分析。

## 代码结构

- `app.py`: Streamlit页面、报告下载
- `signal_core.py`: 数据解析、采样率转换、FFT和时域特征
- `online_samples.py`: 可选的服务器端官方数据获取（不在GitHub托管数据）
- `prepare_demo.py`: 从官方MAT源制作暂存示例并记录SHA-256
- `requirements.txt`: 环境依赖
