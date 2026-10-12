"""Standalone Streamlit entry point: streamlit run lovis_v40/streamlit_app.py"""
import json
from pathlib import Path

import streamlit as st

from line_fault_locator import locate_faults
from lovis_v40.integration import IntegrationError, analyze

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "v40"
TOPO = json.loads((CONFIG / "topology.json").read_text(encoding="utf-8"))
SIGNALS = json.loads((CONFIG / "signal_dictionary.json").read_text(encoding="utf-8"))
ALARMS = json.loads((CONFIG / "alarm_dictionary.json").read_text(encoding="utf-8"))
RULES = json.loads((CONFIG / "locator_config.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Lovis V4.0 实验诊断", page_icon="🛠️", layout="wide")
st.title("Lovis V4.0 · 生产线管理Agent实验")
st.caption("本地规则引擎负责定位排序；Dify管理Agent负责解释证据并选择专家工作流。实验评分不是故障概率，候选不是确诊。")
sample_path = ROOT / "tests" / "fixtures" / "t02_vacuum_gripper_simulation.json"
sample = json.loads(sample_path.read_text(encoding="utf-8"))
raw = st.text_area("事件列表 JSON", value=json.dumps(sample, ensure_ascii=False, indent=2), height=300)
if st.button("分析并调用管理Agent", type="primary"):
    try:
        events = json.loads(raw)
        if not isinstance(events, list):
            raise ValueError("输入必须是事件列表")
        result = analyze(events, TOPO, SIGNALS, ALARMS, RULES)
        st.subheader("结构化诊断与Agent决策")
        st.json(result)
        if result.get("status") == "dify_not_configured":
            st.info("本地定位已完成；未配置Dify管理Agent或专家工作流API密钥，平台调用未执行。")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        st.error(f"输入解析失败：{exc}")
    except IntegrationError as exc:
        st.error(f"Agent集成失败：{exc}")

