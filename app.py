# -*- coding: utf-8 -*-
"""智维AI V3.0 — reproducible vibration-signal demo web app (NOT a classifier)."""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from online_samples import ensure_demo_assets

from signal_core import (
    ANALYSIS_FS, DATA_SOURCE, WINDOW, extract_features, load_builtin_assets,
    read_mat, to_analysis_fs,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='智维AI V3.0 | 工业振动分析工作台', page_icon='⚙️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {max-width:1280px; padding-top:1.7rem; padding-bottom:2.5rem}
[data-testid="stMetric"] {background:#f7fafc;border-radius:11px;padding:12px;border:1px solid #e5eaf0}
h1 {letter-spacing:-.025em}
small, .subtle {color:#64748b}
.provenance {padding:12px 16px;background:#f7fafc;border:1px solid #e2e8f0;border-radius:11px}
</style>''', unsafe_allow_html=True)


def _chart(x, y, xlab: str, ylab: str, title: str, limit: int = 3000) -> go.Figure:
    n = len(x)
    factor = max(1, int(np.ceil(n / limit)))
    fig = go.Figure(go.Scatter(x=np.asarray(x)[::factor], y=np.asarray(y)[::factor], mode='lines', name=title,
                               line={'color':'#2563eb','width':1.15}))
    fig.update_layout(height=300, margin={'l':15,'r':15,'t':35,'b':15}, title={'text':title,'font':{'size':15}},
                      xaxis_title=xlab, yaxis_title=ylab, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    return fig


def _encode_matplotlib_graph(x, y, xlabel, ylabel, title) -> str:
    f, ax = plt.subplots(figsize=(9,3.0), dpi=110)
    ax.plot(x,y,linewidth=.8,color='#2563eb')
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(alpha=.23)
    f.tight_layout()
    out = io.BytesIO()
    f.savefig(out, format='png', bbox_inches='tight')
    plt.close(f)
    return base64.b64encode(out.getvalue()).decode('ascii')


def render_html_report(meta: dict, res: dict, offset: int) -> bytes:
    def clean(x): return html.escape(str(x),quote=True)
    # Matplotlib may lack CJK glyphs in Windows/Cloud environments; use English text
    # inside rendered PNGs. Chinese headings in the surrounding HTML remain intact.
    img1 = _encode_matplotlib_graph(res['time_seconds'],res['waveform'],
                                    'Time (s)','Signal amplitude (source units)','Time-domain waveform')
    img2 = _encode_matplotlib_graph(res['fft_hz'],res['fft_amplitude'],
                                    'Frequency (Hz)','Amplitude (source units)','FFT amplitude spectrum')
    values = [('RMS',res['rms_signal_units']),('峭度（Pearson）',res['kurtosis_pearson']),
              ('峰值因子',res['crest_factor']),('频谱最大幅值所在频率 (Hz)',res['dominant_fft_hz']),
              ('谱重心 (Hz)',res['spectral_centroid_hz'])]
    table = ''.join(f'<tr><td>{clean(k)}</td><td>{clean(round(v,6) if v is not None else "不适用")}</td></tr>' for k,v in values)
    tags = [('原始数据',meta.get('source_url','用户上传')),('输入名称',meta.get('name',meta.get('number','未命名'))),
            ('已知数据标签（非模型预测）',meta.get('label','用户自定义，未知标签')),
            ('原始采样率',f"{meta.get('source_fs_hz', '未知')} Hz"), ('分析采样率',f'{ANALYSIS_FS} Hz'),
            ('窗口起点（统一采样率）',offset), ('窗口长度',WINDOW),
            ('原文件SHA256',meta.get('raw_sha256','未记录')),
            ('演示片段SHA256',meta.get('derived_signal_sha256','未记录'))]
    tags_html = ''.join(f'<tr><td>{clean(k)}</td><td style="word-break:break-all">{clean(v)}</td></tr>' for k,v in tags)
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智维AI 振动分析报告</title><style>body{{font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#f1f5f9;color:#172b4d}}main{{max-width:970px;margin:auto;padding:32px}}article{{background:white;border:1px solid #e5e9f0;border-radius:12px;padding:25px;margin:18px 0}}h1{{font-size:25px}}h2{{font-size:19px}}img{{width:100%;max-width:900px}}table{{width:100%;border-collapse:collapse}}td{{padding:9px;border-bottom:1px solid #edf0f5}}td:first-child{{width:34%}}.note{{background:#fff8e8;padding:12px;border-radius:8px}}</style></head><body><main><h1>智维AI V3.0 · 振动信号分析报告</h1><p>生成时间（UTC）：{clean(datetime.now(timezone.utc).isoformat())}</p><article><h2>计算结果</h2><table>{table}</table><p>FFT采用去均值、周期Hann窗与单边幅值谱；均取{WINDOW}个点。</p></article><article><h2>真实采样片段时域波形</h2><img alt="波形" src="data:image/png;base64,{img1}"><h2>FFT频谱</h2><img alt="FFT" src="data:image/png;base64,{img2}"></article><article><h2>证据来源</h2><table>{tags_html}</table></article><article class="note"><strong>关键限制：</strong>这些数值属于描述性时域/频域特征，不是故障分类结果或设备报警值。网页内置数据的已知标签来自CWRU官方实验台文件，并非模型预测。没有训练分类器、没有验证故障准确率，亦不代表真实生产设备已确诊故障。振动数据原始幅值未作工程单位校准。</article></main></body></html>'''
    return doc.encode('utf-8')


st.title('⚙️ 智维AI V3.0 · 振动信号分析工作台')
st.caption('真实实验台数据 · 确定性信号计算 · 可追溯报告｜目前为网页第一阶段：尚未训练故障分类模型，也未接入Dify')

@st.cache_resource(show_spinner='首次加载演示数据：正在从CWRU官方地址获取真实测量记录，可能需要1分钟…')
def _load_demo():
    try:
        ensure_demo_assets(ROOT)
        return (*load_builtin_assets(ROOT), None)
    except Exception as exc:
        return None, {}, str(exc)

manifest, demo, demo_error = _load_demo()
if demo_error:
    st.warning('官方演示样本当前不可用。你仍可以使用左侧“上传MAT文件”分析真实数据。')
    if st.button('重试从CWRU官方获取内置样本'):
        _load_demo.clear()
        st.rerun()

with st.sidebar:
    st.subheader('分析数据')
    mode = st.radio('数据来源', ['内置真实演示样本','上传MAT文件'], horizontal=False)
    meta = None
    signal = None
    src_name = ''
    offset = 0
    if mode == '内置真实演示样本':
        if demo_error:
            st.warning('内置演示样本暂时无法从官方载入。请点击页面中的重试按钮，或使用“上传MAT文件”。')
            with st.expander('查看缺失原因'):
                st.code(demo_error)
        else:
            samples = manifest['samples']
            names = [f"{s['number']}.mat · {s['label']} · {s['load_hp']} HP" for s in samples]
            selected = st.selectbox('选择一组CWRU真实实验数据', options=range(len(names)), format_func=lambda i: names[i])
            meta = dict(samples[selected])
            signal = demo[str(meta['number'])]
            src_name = f"{meta['number']}.mat"
            st.info('表内类别是官方已知标签，并非模型预测。')
    else:
        upload = st.file_uploader('上传驱动端振动MAT文件', type=['mat'])
        src_fs = st.selectbox('原始采样率（必须与数据一致）', [12000,48000], format_func=lambda v: f'{v:,} Hz')
        if upload is not None:
            content = upload.getvalue()
            if len(content) > 25 * 1024 * 1024:
                st.error('文件超过25MB，本版不支持。')
            else:
                try:
                    x,key,rpm = read_mat(content)
                    signal = to_analysis_fs(x, src_fs)
                    src_name = upload.name
                    meta = {'name':upload.name,'source_url':'用户上传，来源未独立验证',
                            'source_fs_hz':src_fs,'analysis_fs_hz':ANALYSIS_FS,'rpm':rpm,
                            'raw_sha256':hashlib.sha256(content).hexdigest(),'label':'未知（不执行分类）',
                            'source_variable':key}
                    st.success(f'成功读取变量{key}，共{len(x):,}个原始采样点。')
                except Exception as exc:
                    st.error(f'MAT文件无法读取：{exc}')
    if signal is not None:
        window_count = len(signal)//WINDOW
        offset_index = st.slider('选择分析窗口（各窗口不重叠）',0,max(0,window_count-1),0)
        offset = offset_index*WINDOW
        st.caption(f'统一采样率12,000Hz；窗口{offset_index+1}/{window_count}，约0.341秒。')
    selection_identity = (mode, src_name, offset,
                          meta.get('derived_signal_sha256', meta.get('raw_sha256', '')) if meta else None)
    start = st.button('▶ 开始分析',type='primary',use_container_width=True,disabled=(signal is None))
    st.divider()
    st.markdown('**项目状态**')
    st.caption('✅ 真实数据读取及频谱计算')
    st.caption('✅ 证据与HTML报告下载')
    st.caption('⏳ 机器学习分类模型（待开发）')
    st.caption('⏳ Dify工业知识Agent集成（待开发）')

if start and signal is not None and meta is not None:
    try:
        window = np.asarray(signal[offset:offset+WINDOW],dtype=np.float64)
        result = extract_features(window)
        st.session_state['last_analysis'] = {'result':result,'meta':meta,'offset':offset,
                                             'source_name':src_name,'selection_identity':selection_identity}
    except Exception as exc:
        st.error(f'分析失败：{exc}')

last = st.session_state.get('last_analysis')
# Do not show stale calculation results after the user changes the selected
# sample, upload, or analysis window. The user must click Analyze again.
if last is not None and last.get('selection_identity') != selection_identity:
    st.session_state.pop('last_analysis', None)
    last = None
if last is None:
    st.info('请在左侧选择内置CWRU样本或上传MAT文件，然后点击“开始分析”。演示样本来自真实实验测量，不会使用生成的假波形。')
else:
    res, evidence, o = last['result'],last['meta'],last['offset']
    st.subheader(f"分析结果 · {last['source_name']}")
    c1,c2,c3,c4=st.columns(4)
    c1.metric('RMS（源信号单位）',f"{res['rms_signal_units']:.5f}")
    c2.metric('峭度（Pearson）',f"{res['kurtosis_pearson']:.3f}")
    c3.metric('峰值因子',f"{res['crest_factor']:.3f}" if res['crest_factor'] is not None else '不可计算')
    c4.metric('频谱最大峰值频率',f"{res['dominant_fft_hz']:.1f} Hz")
    left,right=st.columns(2)
    with left:
        st.plotly_chart(_chart(res['time_seconds'],res['waveform'],'时间 (秒)','源信号值','时域波形'),use_container_width=True)
    with right:
        st.plotly_chart(_chart(res['fft_hz'],res['fft_amplitude'],'频率 (Hz)','单边幅度','FFT频谱'),use_container_width=True)
    st.caption('计算方法：4096点；去均值；FFT采用周期Hann窗和单边幅度谱。峰值频率不等于轴承内/外圈故障特征频率。')
    t1,t2=st.tabs(['🔎 数据来源与证据','📄 下载可复核结果'])
    with t1:
        st.markdown('**数据与可追溯性**')
        if evidence.get('source_known_label_not_prediction'):
            st.info(f"公开实验台已知标签：{evidence.get('label')}。这是原始文件标签，不是模型预测结果。")
        elif evidence.get('label'):
            st.info(f"数据标签：{evidence.get('label')}。没有故障分类模型输出。")
        cols={
            '输入文件':last['source_name'],
            '数据来源':evidence.get('source_url','未知'),
            '原始采样率 (Hz)':evidence.get('source_fs_hz','未知'),
            '统一分析采样率 (Hz)':ANALYSIS_FS,
            '分析起点（统一采样率样本索引）':o,
            '窗口采样点':WINDOW,
            '原始MAT SHA-256':evidence.get('raw_sha256','上传文件未记录'),
            '演示片段 SHA-256':evidence.get('derived_signal_sha256','上传数据不适用'),
            '工况HP':evidence.get('load_hp','未提供'),
            'RPM元数据':evidence.get('rpm','未提供'),
        }
        st.table([{'字段':k,'内容':str(v)} for k,v in cols.items()])
        if evidence.get('source_url','').startswith('https://'):
            st.link_button('访问CWRU数据文件',evidence['source_url'])
        st.warning('本阶段只有信号处理与已知标签展示。尚未训练、验证故障分类器；没有工业报警阈值。公开实验台测试不等于工厂现场确诊。')
    with t2:
        report = render_html_report(evidence,res,o)
        st.download_button('⬇ 下载完整HTML分析报告',data=report,file_name='zhiwei_v30_vibration_report.html',mime='text/html',use_container_width=True)
        data = {k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in res.items() if not isinstance(v,np.ndarray)}
        data.update({'source_metadata':evidence,'window_offset':o,'window_samples':WINDOW,
                     'analysis_fs_hz':ANALYSIS_FS,'classification_status':'not_trained'})
        st.download_button('⬇ 下载结构化计算结果(JSON)',data=json.dumps(data,ensure_ascii=False,indent=2,default=str).encode('utf-8'),
                           file_name='zhiwei_v30_evidence.json',mime='application/json',use_container_width=True)
        st.caption('下载报告包含自带的图片，不依赖网页运行；结构化结果供进一步复核。')

st.divider()
st.caption(f'公开数据来源：CWRU Bearing Data Center · {DATA_SOURCE} ｜研究原型，不可替代现场工程安全判断。')
