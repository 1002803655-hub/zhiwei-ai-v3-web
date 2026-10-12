# -*- coding: utf-8 -*-
"""Lovis V3.1 experimental agent-first user interface.

Deploy separately from app.py until tested. Existing V3.0 remains the production app.
Dify Chatflow selects its own existing knowledge/trend/parameter tools. Vibration
analysis is a deterministic local backend called ONLY after user requests it.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import uuid
from pathlib import Path
from streamlit_local_storage import LocalStorage
from lovis_conversations import (
    STORAGE_KEY_V2, MAX_CHATS, new_store, active_chat, add_chat,
    switch_chat, rename_chat, delete_chat, sync_active, validate_store, to_json_bytes
)
from typing import Optional

import numpy as np
import streamlit as st
from scipy.io import loadmat

from online_samples import ensure_demo_assets
from signal_core import (
    ANALYSIS_FS, WINDOW, DATA_SOURCE, extract_features, load_builtin_assets,
    read_mat, to_analysis_fs,
)
from ml_inference import predict_signal, MODEL_SHA256
from dify_bridge import DifyBridgeError
from lovis_bridge import ask_lovis, combine_question_with_vibration, vibration_summary
from lovis_report import make_report

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="智维AI V3.1 | Lovis智能诊断Agent",
                   page_icon="🤖",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width:1450px; padding-top:1.25rem; padding-bottom:2rem}
.lovis-hero {background:linear-gradient(115deg,#eef5ff,#f7faff);border:1px solid #dbe8ff;
  border-radius:19px;padding:21px 25px;margin:5px 0 18px}
.lovis-hero h2 {font-size:1.75rem;margin:0 0 8px;color:#19365f}
.lovis-hero p {color:#4f6687;font-size:0.95rem;margin:0}
.small-tag {font-size:0.8rem;color:#506987}
[data-testid="stMetric"] {border:1px solid #e1e8f0;border-radius:10px;padding:9px}
</style>
""",unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在向CWRU官方获取演示振动数据……")
def load_demo_once():
    try:
        ensure_demo_assets(BASE)
        manifest,samples=load_builtin_assets(BASE)
        return manifest,samples,None
    except Exception as exc:
        return None,{},str(exc)


def read_uploaded_source(content:bytes, name:str):
    """Prefer standard reader; for multi-channel archived MAT pick filename-matched DE variable."""
    try:
        return read_mat(content)
    except ValueError as original:
        match=re.fullmatch(r"(\d{2,4})\.mat",str(name),re.I)
        if not match or "DE_time" not in str(original):
            raise
        mat=loadmat(io.BytesIO(content))
        target=f"X{int(match.group(1)):03d}_DE_time"
        de_keys=[k for k in mat if str(k).lower().endswith("_de_time")]
        if target not in de_keys:
            raise ValueError("无法按MAT编号定位唯一驱动端通道，请核查数据来源") from original
        values=np.asarray(mat[target])
        if values.dtype.kind not in "fi" or values.ndim > 2:
            raise ValueError("MAT驱动端通道数据格式错误")
        values=np.asarray(values,dtype=np.float64).reshape(-1)
        if len(values)<WINDOW or len(values)>10_000_000 or not np.isfinite(values).all():
            raise ValueError("振动数据长度或数值未通过校验")
        rpm_key=f"X{int(match.group(1)):03d}RPM"
        rpm=None
        if rpm_key in mat:
            try:
                maybe=float(np.asarray(mat[rpm_key]).reshape(-1)[0])
                if np.isfinite(maybe) and maybe>0:
                    rpm=maybe
            except (IndexError,TypeError,ValueError):
                pass
        return values,target,rpm


def apply_active_chat(chat:dict) -> None:
    """Load messages AND Dify identifiers; never leak context across chats."""
    st.session_state.lovis_messages = list(chat["messages"])
    st.session_state.lovis_conversation_id = chat["dify_conversation_id"]
    st.session_state.lovis_anon_user = chat["dify_user"]
    st.session_state.lovis_tool_events = list(chat["recent_tool_events"])
    st.session_state.lovis_trace_source = chat["recent_tool_trace_source"]
    st.session_state.pop("lovis_analysis",None)  # MAT data and waveforms are ephemeral.
    # Requiring an explicit new attachment prevents samples from one
    # motor being silently treated as part of a different diagnosis.
    st.session_state["lovis_data_source"]="暂不关联"
    st.session_state.pop("lovis_mat_upload",None)


def save_active_chat() -> None:
    sync_active(
        st.session_state.lovis_chat_store,
        st.session_state.get("lovis_messages",[]),
        st.session_state.get("lovis_anon_user",""),
        st.session_state.get("lovis_conversation_id",""),
        st.session_state.get("lovis_tool_events",[]),
        st.session_state.get("lovis_trace_source",""),
    )
    st.session_state.lovis_archive_dirty=True
    st.session_state.lovis_archive_restored=True


def restore_archive(archive:dict) -> None:
    st.session_state.lovis_chat_store=validate_store(archive)
    apply_active_chat(active_chat(st.session_state.lovis_chat_store))
    st.session_state.lovis_archive_restored=True


def activate_chat(chat_id:str) -> None:
    save_active_chat()
    chosen=switch_chat(st.session_state.lovis_chat_store,chat_id)
    apply_active_chat(chosen)
    st.session_state.lovis_archive_dirty=True
    st.rerun()


def create_chat() -> None:
    save_active_chat()
    created=add_chat(st.session_state.lovis_chat_store)
    apply_active_chat(created)
    st.session_state.lovis_archive_dirty=True
    st.rerun()



def wants_vibration_analysis(question:str)->bool:
    q=question.replace(" ","").lower()
    if any(p in q for p in ("不要分析","不需要分析","先别分析","暂不分析","无需分析")):
        return False
    if any(p in q for p in ("分析这份数据","分析当前数据","分析上传文件","分析所选数据",
                            "开始振动分析","振动信号分析","执行振动分析","运行振动分析",
                            "分析振动数据","振动故障分类","分析mat","分类这份")):
        return True
    return bool(re.search(r"(振动|轴承|波形|频谱|mat).{0,20}(分析|诊断|分类|检测|提取|识别)",q)
                or re.search(r"(分析|诊断|分类|检测).{0,15}(振动|轴承|波形|频谱|mat)",q))


def request_needs_previous_vibration(q:str)->bool:
    return any(x in q for x in ("振动","轴承","刚才","上面","当前","这份","这个","模型","分类",
                               "综合","温度","电流","负载","工况","故障"))


def local_analysis(signal:np.ndarray, meta:dict, offset:int, identity:str)->dict:
    segment=np.asarray(signal[offset:offset+WINDOW],dtype=np.float64)
    res=extract_features(segment)
    try:
        prediction=predict_signal(signal,max_windows=8)
        error=None
    except Exception as exc:
        prediction=None
        error=str(exc)
    return {"identity":identity,"meta":dict(meta),"result":res,
            "classification":prediction,"model_error":error,
            "offset":offset, "ai_answer":"", "tool_events":[]}


def show_evidence(analysis:dict, tool_events:list[str]):
    res=analysis["result"]
    pred=analysis.get("classification")
    st.markdown("#### 本轮分析结果")
    a,b=st.columns(2)
    a.metric("RMS（源信号单位）",f"{res['rms_signal_units']:.5f}")
    b.metric("峭度",f"{res['kurtosis_pearson']:.3f}")
    c,d=st.columns(2)
    c.metric("峰值因子",f"{res['crest_factor']:.3f}" if res["crest_factor"] is not None else "不适用")
    d.metric("FFT峰位置",f"{res['dominant_fft_hz']:.1f} Hz")
    if pred:
        st.markdown("**实验模型预测：** "+pred["predicted_label"])
        st.caption("投票："+str(pred.get("vote_counts",{}))+"；投票比例不是故障概率。")
    else:
        st.warning("模型不可用，本次只显示信号分析："+str(analysis.get("model_error","")))
    if analysis["meta"].get("source_known_label_not_prediction"):
        st.caption("数据集已知标签："+str(analysis["meta"].get("label","未知"))+"（不属于模型预测）")
    if analysis["meta"].get("is_training_source") or (
        analysis["meta"].get("source_known_label_not_prediction") and analysis["meta"].get("load_hp")==0
    ):
        st.warning("当前0HP演示样本属于模型训练来源，只能展示推理流程，不是独立测试。")
    import plotly.graph_objects as go
    with st.expander("查看真实振动波形与FFT频谱",expanded=False):
        wave=go.Figure(go.Scatter(
            x=res["time_seconds"][::2],y=res["waveform"][::2],
            mode="lines",line={"color":"#2563eb","width":1}))
        wave.update_layout(height=230,margin=dict(l=5,r=5,t=8,b=10),
                           xaxis_title="时间（秒）",yaxis_title="源信号")
        st.plotly_chart(wave,use_container_width=True)
        fft=go.Figure(go.Scatter(
            x=res["fft_hz"],y=res["fft_amplitude"],mode="lines",
            line={"color":"#0f766e","width":1}))
        fft.update_layout(height=230,margin=dict(l=5,r=5,t=8,b=10),
                          xaxis_title="频率（Hz）",yaxis_title="单边幅值")
        st.plotly_chart(fft,use_container_width=True)
        st.caption("FFT最大峰不等于轴承特征故障频率；不适用通用现场报警阈值。")
    with st.expander("下载本轮可追溯报告",expanded=False):
        report=make_report(analysis,analysis.get("ai_answer",""),tool_events)
        st.download_button("下载HTML报告",data=report,
                           file_name="lovis_v31_diagnostic_report.html",mime="text/html",
                           use_container_width=True)


if "lovis_chat_store" not in st.session_state:
    st.session_state.lovis_chat_store=new_store()
if "lovis_messages" not in st.session_state:
    apply_active_chat(active_chat(st.session_state.lovis_chat_store))
if "lovis_calls" not in st.session_state:
    st.session_state.lovis_calls=0  # Per active browser session; NOT reset when switching chats.

# Browser-local archive v2, with automatic upgrade from the previous single-chat
# storage key. Old key is left untouched as a backup.
try:
    lovis_storage=LocalStorage()
    if not st.session_state.get("lovis_archive_restored",False):
        stored_archive=lovis_storage.getItem(STORAGE_KEY_V2)
        if stored_archive is not None:
            try:
                # The browser key is unchanged: accept either older v1 or v2.
                import json
                source_schema=(json.loads(stored_archive) if isinstance(stored_archive,str)
                               else stored_archive).get("schema")
                restore_archive(stored_archive)
                if source_schema == 1:
                    st.session_state.lovis_archive_dirty=True
                    st.session_state.lovis_storage_note="旧版对话记录已自动迁移到多会话列表。"
            except (ValueError,TypeError,AttributeError) as exc:
                st.session_state.lovis_storage_note="已保存的会话记录有误："+str(exc)
                st.session_state.lovis_archive_restored=True
    if st.session_state.get("lovis_archive_dirty",False):
        # Build full validated JSON, then write one atomic archive.
        lovis_storage.setItem(STORAGE_KEY_V2,validate_store(st.session_state.lovis_chat_store))
        st.session_state.lovis_archive_dirty=False
except Exception as exc:
    st.session_state.lovis_storage_note=(
        "自动保存暂不可用（"+type(exc).__name__+"）；请在侧栏下载私密会话备份。"
    )

# Sidebar is optional context. It does not automatically call any algorithm.
current_signal:Optional[np.ndarray]=None
current_meta=None
offset=0
identity="none"
with st.sidebar:
    st.markdown("### 🤖 Lovis")
    st.caption("智维AI · 工业电机智能辅助诊断")
    if st.button("＋ 开始新对话",use_container_width=True):
        try:
            create_chat()
        except ValueError as exc:
            st.warning(str(exc))
    st.markdown("**最近**")
    archive=st.session_state.lovis_chat_store
    for ch in sorted(archive["chats"],key=lambda c:c["updated_at"],reverse=True):
        active = (ch["id"] == archive["active_id"])
        label = ("▸ " if active else "💬 ") + ch["title"]
        if st.button(label,key="lovis_select_"+ch["id"],
                     help="切换到此独立对话",use_container_width=True,
                     type="primary" if active else "secondary"):
            if not active:
                activate_chat(ch["id"])
    with st.expander("管理会话 / 备份与恢复",expanded=False):
        st.caption(f"本浏览器现有{len(archive['chats'])}/{MAX_CHATS}段独立对话。"
                   "每段对话分别保留Dify上下文。")
        current=active_chat(archive)
        title=st.text_input("当前对话名称",value=current["title"],
                            max_chars=40,key="lovis_title_"+current["id"])
        if st.button("保存名称",use_container_width=True):
            save_active_chat()
            rename_chat(archive,current["id"],title)
            st.rerun()
        confirm_delete=st.checkbox("确认删除当前对话（不可撤销）",
                                   key="lovis_delete_confirm_"+current["id"])
        if st.button("删除当前对话",disabled=not confirm_delete,
                     use_container_width=True):
            delete_chat(archive,current["id"])
            apply_active_chat(active_chat(archive))
            st.session_state.lovis_archive_dirty=True
            st.rerun()
        st.divider()
        st.caption("浏览器会自动保存，但清除站点数据、换设备或隐私模式可能导致记录丢失。")
        if st.session_state.get("lovis_storage_note"):
            st.warning(st.session_state["lovis_storage_note"])
        try:
            archive_bytes=to_json_bytes(archive)
            st.download_button(
                "⬇ 导出全部会话（私密JSON）",data=archive_bytes,
                file_name="lovis_all_chats_backup.json",mime="application/json",
                use_container_width=True,
            )
        except ValueError as exc:
            st.error("会话备份超出限制："+str(exc))
        backup=st.file_uploader("导入Lovis会话备份（支持旧版JSON）",
                                type=["json"],key="lovis_archive_import")
        if backup is not None:
            st.caption("新版多会话备份会替换当前全部会话；旧版单会话备份会新增一条对话。"
                       "请先导出当前全部会话。")
        overwrite=st.checkbox("确认导入或替换会话记录",
                              key="lovis_archive_import_confirm")
        if st.button("恢复备份",disabled=backup is None or not overwrite,
                     use_container_width=True):
            try:
                if backup.size>2_000_000:
                    raise ValueError("备份文件超过2MB")
                incoming=backup.getvalue().decode("utf-8")
                imported=validate_store(incoming)
                if imported.get("schema")!=2:
                    raise ValueError("备份格式错误")
                import json
                raw=json.loads(incoming)
                if raw.get("schema")==1:
                    if len(archive["chats"])>=MAX_CHATS:
                        raise ValueError("会话数量已满，先导出并清理旧会话")
                    new_item=imported["chats"][0]
                    existing={(c["dify_user"],c["dify_conversation_id"]) for c in archive["chats"]}
                    if (new_item["dify_user"],new_item["dify_conversation_id"]) in existing:
                        raise ValueError("这段旧对话已在当前会话列表中")
                    save_active_chat()
                    archive["chats"].insert(0,new_item)
                    archive["active_id"]=new_item["id"]
                    st.session_state.lovis_chat_store=archive
                    apply_active_chat(new_item)
                    st.session_state.lovis_archive_dirty=True
                else:
                    restore_archive(imported)
                    st.session_state.lovis_archive_dirty=True
                st.rerun()
            except (ValueError,TypeError,UnicodeDecodeError) as exc:
                st.error("导入失败："+str(exc))
        st.caption("备份含聊天原文和Dify会话标识。请勿分享给他人；"
                   "原始MAT和波形不会随会话记录保存。")
    st.divider()
    st.markdown("**关联数据（可选）**")
    source=st.radio("数据来源",["暂不关联","CWRU内置示例","上传MAT文件"],label_visibility="collapsed",key="lovis_data_source")
    if source=="CWRU内置示例":
        manifest,samples,err=load_demo_once()
        if err:
            st.warning("官方演示数据暂时不可用，可上传MAT文件。")
            if st.button("重新读取官方样本"):
                load_demo_once.clear()
                st.rerun()
        else:
            choices=manifest["samples"]
            index=st.selectbox(
                "选择实验记录",range(len(choices)),
                format_func=lambda i: str(choices[i]["number"])+".mat · "+choices[i]["label"])
            current_meta=dict(choices[index])
            current_signal=samples[str(current_meta["number"])]
            identity="demo-"+str(current_meta["number"])
            st.caption("所选类别为实验台标签，不是模型预测。")
    elif source=="上传MAT文件":
        f=st.file_uploader("选择振动MAT",type=["mat"],key="lovis_mat_upload")
        # CWRU Normal Baseline 97-100 is recorded at 48 kHz.
        # Filename is only a hint: non-official or modified uploads still require confirmation.
        known_cwru_normal=bool(f is not None and re.fullmatch(r"(97|98|99|100)\.mat",f.name,re.I))
        hz=st.selectbox("原始采样率（须与文件真实采样率一致）",[12000,48000],
                        index=1 if known_cwru_normal else 0,
                        format_func=lambda a:f"{a:,} Hz",
                        key="lovis_sample_rate_"+(f.name if f is not None else "none"))
        if known_cwru_normal:
            st.caption("检测到CWRU正常组文件名：已预选48 kHz；请核对来源后再分析。")
        if f is not None:
            content=f.getvalue()
            if len(content)>25*1024*1024:
                st.error("MAT文件不得大于25 MB。")
            else:
                try:
                    arr,channel,rpm=read_uploaded_source(content,f.name)
                    current_signal=to_analysis_fs(arr,hz)
                    sha=hashlib.sha256(content).hexdigest()
                    current_meta={
                        "name":f.name,"source_url":"用户上传，未独立验证",
                        "source_fs_hz":hz,"raw_sha256":sha,"rpm":rpm,
                        "source_variable":channel,"label":"未知（用户上传）",
                    }
                    identity="uploaded-"+sha+"-"+str(hz)
                    st.caption("读取成功；采样率由上传者确认。原文件不会发送给Dify。")
                except Exception as exc:
                    st.error("无法读取MAT："+str(exc))
    if current_signal is not None and len(current_signal)>=WINDOW:
        n_windows=len(current_signal)//WINDOW
        offset_idx=st.slider("时域分析窗口",0,n_windows-1,0)
        offset=offset_idx*WINDOW
        identity+="-offset-"+str(offset)
        st.info("仅关联数据；Lovis不会自动运行振动分类。请在对话中提出分析请求。")
    st.divider()
    st.caption("**云端与隐私**")
    approved=st.checkbox("同意将本轮文字提问和脱敏计算摘要发送至Dify Cloud",value=False)
    st.caption("不向Dify上传MAT、原始波形、文件名、SHA-256。")
    st.caption("公开实验台研究原型，不能代替现场故障确诊。")

try:
    api_key=str(st.secrets.get("DIFY_API_KEY","") or os.environ.get("DIFY_API_KEY","")).strip()
    api_base=str(st.secrets.get("DIFY_API_BASE","https://api.dify.ai/v1"))
except (FileNotFoundError,KeyError):
    api_key=os.environ.get("DIFY_API_KEY","").strip()
    api_base="https://api.dify.ai/v1"

st.markdown("""<div class="lovis-hero">
<h2>🤖 你好，我是 Lovis</h2>
<p>你的工业电机智能辅助诊断助手。直接描述问题；如果需要数据分析，可在左侧关联振动样本，
再通过对话让我开始分析。温度、电流和知识咨询也可以直接输入。</p></div>""",unsafe_allow_html=True)
st.caption("Lovis V3.1 开发预览 · 对话是主入口 · 结果按需展开 · 卡通形象待设计")
if not st.session_state.lovis_messages:
    with st.chat_message("assistant",avatar="🤖"):
        st.markdown("你好！我可以帮助分析轴承振动、温度电流变化、运行参数，以及查询工业电机维护知识。请描述现象，或先在侧边栏关联测量数据。必要信息不足时，我会先追问。")

example_text=None
if not st.session_state.lovis_messages:
    k1,k2,k3=st.columns(3)
    if k1.button("咨询轴承异响",use_container_width=True):
        example_text="电机运行中有轴承异响，但没有测量数据。应该先确认哪些信息？"
    if k2.button("分析所选振动",use_container_width=True):
        example_text="请分析当前选中的振动样本，并解释模型预测的证据和局限。"
    if k3.button("询问温升趋势",use_container_width=True):
        example_text="如果轴承温度连续上涨，要提供哪些时序数据才能进行趋势分析？"

for msg in st.session_state.lovis_messages:
    with st.chat_message(msg["role"],avatar="🤖" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("tools"):
            st.caption("Dify已核验工具调用："+", ".join(msg["tools"])+"｜来源："+msg.get("tool_trace_source","Dify工具记录"))

question=st.chat_input("向Lovis描述问题、追加运行信息，或请求分析已关联的振动数据…")
message=question or example_text

if message:
    message=str(message).strip()
    if len(message)>5000:
        st.error("请将本轮输入控制在5000字符内。")
    else:
        st.session_state.lovis_messages.append({"role":"user","content":message})
        local_msg=""
        current_analysis=st.session_state.get("lovis_analysis")
        if wants_vibration_analysis(message):
            if current_signal is None or current_meta is None:
                local_msg="尚未关联有效振动数据。请在左侧选择示例或上传MAT，再告诉我“分析这份振动数据”。"
            else:
                try:
                    current_analysis=local_analysis(current_signal,current_meta,offset,identity)
                    st.session_state.lovis_analysis=current_analysis
                    local_msg="网页后端已按你的请求实际完成振动信号分析与实验性Random Forest预测。"
                except Exception as exc:
                    local_msg="本地振动分析失败："+str(exc)
        elif current_analysis and current_analysis.get("identity")!=identity:
            current_analysis=None

        if not approved:
            answer=(local_msg+"\n\n" if local_msg else "")+"如需Lovis进行Dify多轮诊断，请先勾选左侧的文字与脱敏摘要发送同意选项。"
            st.session_state.lovis_messages.append({"role":"assistant","content":answer})
        elif not api_key:
            answer=(local_msg+"\n\n" if local_msg else "")+"Dify服务端密钥尚未配置。可以查看本地振动计算结果，但多轮Agent暂不可用。"
            st.session_state.lovis_messages.append({"role":"assistant","content":answer})
        elif st.session_state.lovis_calls>=6:
            st.session_state.lovis_messages.append({
                "role":"assistant","content":"本次浏览器会话已达6次Dify演示请求上限。可以继续浏览已有证据和报告；新的对话请在演示环境重启会话。"})
        elif local_msg.startswith("本地振动分析失败"):
            st.session_state.lovis_messages.append({"role":"assistant","content":local_msg})
        else:
            summary=None
            if current_analysis and current_analysis.get("identity")==identity and (
                wants_vibration_analysis(message) or request_needs_previous_vibration(message)
            ):
                summary=vibration_summary(current_analysis["meta"],current_analysis["result"],
                                           current_analysis.get("classification"))
            query=combine_question_with_vibration(message,summary)
            st.session_state.lovis_calls+=1
            with st.spinner("Lovis正在处理你的问题，并按需调用Dify现有工具……"):
                try:
                    response=ask_lovis(query,api_key,
                                       api_base=api_base,
                                       user=st.session_state.lovis_anon_user,
                                       conversation_id=st.session_state.lovis_conversation_id)
                    if response.get("conversation_id"):
                        st.session_state.lovis_conversation_id=response["conversation_id"]
                    events=response.get("tool_events",[])
                    trace_source=response.get("tool_trace_source","未返回工具记录")
                    st.session_state.lovis_tool_events=events
                    st.session_state.lovis_trace_source=trace_source
                    st.session_state.lovis_messages.append({
                        "role":"assistant","content":response["answer"],"tools":events,
                        "tool_trace_source":trace_source})
                    if current_analysis and current_analysis.get("identity")==identity:
                        current_analysis["ai_answer"]=response["answer"]
                        current_analysis["tool_events"]=events
                except DifyBridgeError as exc:
                    st.session_state.lovis_messages.append({
                        "role":"assistant","content":(local_msg+"\n\n" if local_msg else "")+"Dify调用失败："+str(exc)})
        save_active_chat()
        st.rerun()

# Compact, conditional evidence zone: do not make analytic widgets the homepage.
analysis=st.session_state.get("lovis_analysis")
if analysis and analysis.get("identity")==identity:
    st.divider()
    with st.container():
        st.markdown("### 📊 按需生成的分析证据")
        show_evidence(analysis,st.session_state.get("lovis_tool_events",[]))
if st.session_state.get("lovis_messages"):
    events=st.session_state.get("lovis_tool_events",[])
    with st.expander("🔎 Agent执行证据（只显示可核验事件）",expanded=False):
        st.caption("Dify内部知识、趋势和参数工具是否被调用，只以API工具事件和Dify日志为准；此处不展示模型隐含推理过程。")
        if analysis and analysis.get("identity")==identity:
            st.write("✅ 网页后端Python振动分析：用户请求触发；不是Dify工具调用。")
        if events:
            st.caption("证据来源："+st.session_state.get("lovis_trace_source","Dify提供的工具记录"))
            for n,event in enumerate(events,1):
                st.write(f"✅ Dify已记录的工具调用 {n}："+event)
        else:
            st.write("当前轮Dify服务API未提供可核验的工具记录（不等于没有执行）；请到Dify运行日志核对，网页不会凭回答文字伪造调用。")
st.caption("🔗 现有已验收的V3.0网页继续保持运行：https://zhiwei-ai-v3-web.streamlit.app")
