# -*- coding: utf-8 -*-
"""Lovis V3.1 browser history snapshot, validation, and portable export/import.

No raw MAT data, waveforms, uploaded filenames, API keys or SHA identifiers.
The Dify conversation_id and pseudonymous user ID are stored to resume multi-turn context.
Treat snapshots as private: possession can allow continuation of that Dify conversation.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

SCHEMA=1
STORAGE_KEY="lovis_v31_browser_history_v1"
MAX_MESSAGES=60
MAX_TEXT=10000
MAX_SNAPSHOT_BYTES=400000

def build_snapshot(messages, user, conversation_id, tool_events=None, trace_source="", calls=0):
    clean=[]
    for msg in messages[-MAX_MESSAGES:]:
        if not isinstance(msg,dict):
            continue
        role=msg.get("role")
        content=msg.get("content")
        if role not in ("user","assistant") or not isinstance(content,str):
            continue
        row={"role":role,"content":content[:MAX_TEXT]}
        tools=msg.get("tools",[])
        if isinstance(tools,list):
            row["tools"]=[x for x in tools[:12] if isinstance(x,str) and re.fullmatch(r"[\w./:-]{1,100}",x)]
        if isinstance(msg.get("tool_trace_source"),str):
            row["tool_trace_source"]=msg["tool_trace_source"][:120]
        clean.append(row)
    return {
        "schema":SCHEMA,
        "saved_at":datetime.now(timezone.utc).isoformat(),
        "messages":clean,
        "dify_user":str(user),
        "dify_conversation_id":str(conversation_id or ""),
        "recent_tool_events":list(tool_events or [])[:12],
        "recent_tool_trace_source":str(trace_source)[:120],
        "calls":int(calls) if isinstance(calls,int) else 0,
    }

def validate_snapshot(raw):
    if isinstance(raw,str):
        if len(raw.encode("utf-8"))>MAX_SNAPSHOT_BYTES:
            raise ValueError("对话文件超过大小限制")
        try:
            raw=json.loads(raw)
        except (TypeError,ValueError) as exc:
            raise ValueError("不是有效的Lovis会话JSON") from exc
    if not isinstance(raw,dict):
        raise ValueError("不是有效的Lovis会话对象")
    if raw.get("schema")!=SCHEMA:
        raise ValueError("不支持的会话文件版本")
    if len(json.dumps(raw,ensure_ascii=False).encode("utf-8"))>MAX_SNAPSHOT_BYTES:
        raise ValueError("会话记录过大")
    user=raw.get("dify_user")
    if not isinstance(user,str) or not re.fullmatch(r"lovis-[0-9a-f]{32}",user):
        raise ValueError("Dify访客标识不合法")
    conversation_id=raw.get("dify_conversation_id","")
    if not isinstance(conversation_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{0,128}",conversation_id):
        raise ValueError("Dify会话标识不合法")
    msgs=raw.get("messages")
    if not isinstance(msgs,list) or len(msgs)>MAX_MESSAGES:
        raise ValueError("对话条数超限")
    safe=[]
    for msg in msgs:
        if not isinstance(msg,dict) or msg.get("role") not in ("user","assistant"):
            raise ValueError("会话中存在不合法角色")
        content=msg.get("content")
        if not isinstance(content,str) or len(content)>MAX_TEXT:
            raise ValueError("消息过长或类型不正确")
        item={"role":msg["role"],"content":content}
        tools=msg.get("tools",[])
        if not isinstance(tools,list) or len(tools)>12 or any(
            not isinstance(t,str) or not re.fullmatch(r"[\w./:-]{1,100}",t) for t in tools
        ):
            raise ValueError("工具事件字段不正确")
        item["tools"]=tools
        if isinstance(msg.get("tool_trace_source"),str):
            item["tool_trace_source"]=msg["tool_trace_source"][:120]
        safe.append(item)
    events=raw.get("recent_tool_events",[])
    if not isinstance(events,list) or len(events)>12 or any(
        not isinstance(t,str) or not re.fullmatch(r"[\w./:-]{1,100}",t) for t in events
    ):
        raise ValueError("工具记录非法")
    calls=raw.get("calls",0)
    if not isinstance(calls,int) or not 0<=calls<=100000:
        raise ValueError("调用计数非法")
    return {
        "messages":safe,
        "user":user,
        "conversation_id":conversation_id,
        "tool_events":events,
        "trace_source":str(raw.get("recent_tool_trace_source",""))[:120],
        "calls":calls,
    }

def to_bytes(snapshot):
    return json.dumps(snapshot,ensure_ascii=False,indent=2).encode("utf-8")
