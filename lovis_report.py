# -*- coding: utf-8 -*-
"""Portable single-file HTML evidence report for Lovis V3.1."""
from __future__ import annotations
import base64
import html
import io
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _chart_image(xs,ys,title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(8.3,2.9),dpi=110)
    ax.plot(xs,ys,linewidth=.75,color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
    fig.tight_layout()
    b=io.BytesIO()
    fig.savefig(b,format="png")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode("ascii")

def make_report(analysis:dict, latest_ai:str="", tool_events:list[str]|None=None)->bytes:
    meta=analysis["meta"]
    data=analysis["result"]
    pred=analysis.get("classification") or {}
    escape=lambda x: html.escape(str(x),quote=True)
    def row(k,v):
        return "<tr><th>"+escape(k)+"</th><td>"+escape(v)+"</td></tr>"
    metrics=[
        ("RMS（源信号单位）",data.get("rms_signal_units")),
        ("峭度（Pearson）",data.get("kurtosis_pearson")),
        ("峰值因子",data.get("crest_factor")),
        ("最大FFT幅值所在频率（Hz）",data.get("dominant_fft_hz")),
        ("模型预测（非确诊）",pred.get("predicted_label","不可用")),
        ("窗口投票数",json.dumps(pred.get("vote_counts",{}),ensure_ascii=False)),
        ("数据已知标签（非预测）",meta.get("label","未知")),
        ("原始采样率",meta.get("source_fs_hz","未知")),
        ("分析采样率",12000),
        ("模型SHA-256",pred.get("model_sha256","不可用")),
        ("原始MAT SHA-256",meta.get("raw_sha256","不提供")),
    ]
    rows="".join(row(k,round(float(v),6) if isinstance(v,(int,float,np.number)) else v) for k,v in metrics)
    im1=_chart_image(data["time_seconds"],data["waveform"],"Time-domain waveform","Time (s)","Amplitude (source units)")
    im2=_chart_image(data["fft_hz"],data["fft_amplitude"],"FFT amplitude spectrum","Frequency (Hz)","Amplitude")
    ai=escape(latest_ai or "本次未调用Dify；以下报告仍包含可复核的原始分析和模型输出。").replace("\n","<br>")
    events=", ".join(escape(t) for t in (tool_events or [])) or "未收到可直接核验的工具事件；以Dify运行日志为准。"
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智维AI Lovis · 诊断证据报告</title>
<style>body{{background:#f1f5f9;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#182946;margin:0}}
main{{max-width:920px;margin:auto;padding:28px}}section{{background:#fff;padding:22px;margin:16px 0;border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #e7eaf1;padding:8px;word-break:break-word}}
th{{width:39%}}img{{width:100%}}.note{{color:#5f6b7a}}</style></head><body><main>
<h1>⚙️ 智维AI V3.1 · Lovis 诊断证据报告</h1>
<p>UTC：{escape(datetime.now(timezone.utc).isoformat())}</p>
<section><h2>可复核的数值与模型结果</h2><table>{rows}</table></section>
<section><h2>Lovis辅助解释</h2><p>{ai}</p><p class="note">Dify API报告的工具事件：{events}</p></section>
<section><h2>真实采样窗口与频谱</h2><img alt="真实波形" src="data:image/png;base64,{im1}">
<img alt="FFT" src="data:image/png;base64,{im2}"></section>
<section class="note"><h2>适用范围与限制</h2>
<p>本研究原型基于CWRU人工轴承缺陷实验台。0HP演示文件来自训练源，不属于独立测试；
1–3HP的测试仅覆盖同台实验、有限源文件。窗口投票比例不是故障概率；
FFT最大幅值频率不等于轴承故障特征频率；未进行工业现场认证。分析幅值未校准为工程单位。
现场检修须由合格人员遵循停机与能源隔离规程执行。</p></section>
</main></body></html>"""
    return document.encode("utf-8")
