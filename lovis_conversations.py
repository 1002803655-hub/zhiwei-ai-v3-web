# -*- coding: utf-8 -*-
"""Lovis browser-local multi-conversation archive (v2).

Every chat has its own Dify user and conversation ID. Never store API keys,
uploaded MAT files, waveforms or locally computed evidence in this archive.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from lovis_history import validate_snapshot as validate_legacy

STORAGE_KEY_V2 = "lovis_v31_browser_history_v1"  # same browser slot; migrate schema 1 in place
MAX_CHATS = 24
MAX_MESSAGES = 60
MAX_TEXT = 20000  # Dify responses may exceed 10k characters
MAX_BYTES = 2_000_000
MAX_TITLE = 40
ID_RE = re.compile(r"[0-9a-f]{32}")
USER_RE = re.compile(r"lovis-[0-9a-f]{32}")
CONV_RE = re.compile(r"[A-Za-z0-9_-]{0,128}")
TOOL_RE = re.compile(r"[\w./:-]{1,100}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_chat(title: str = "新对话") -> dict:
    t = now_iso()
    return {
        "id": uuid.uuid4().hex,
        "title": normalize_title(title),
        "messages": [],
        "dify_user": "lovis-" + uuid.uuid4().hex,
        "dify_conversation_id": "",
        "recent_tool_events": [],
        "recent_tool_trace_source": "",
        "created_at": t,
        "updated_at": t,
    }


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        raise ValueError("会话标题不是字符串")
    text = " ".join(title.split()).strip()
    return text[:MAX_TITLE] or "新对话"


def auto_title(question: str) -> str:
    line = next((ln.strip() for ln in question.splitlines() if ln.strip()), "")
    # CSV starts are poor conversation labels.
    if line.lower().startswith(("time,current_", "time_series_data", "{")):
        return "工业电机数据分析"
    return normalize_title(line)


def new_store() -> dict:
    chat = make_chat()
    return {"schema": 2, "active_id": chat["id"], "chats": [chat]}


def active_chat(store: dict) -> dict:
    for chat in store["chats"]:
        if chat["id"] == store["active_id"]:
            return chat
    raise ValueError("找不到正在使用的会话")


def add_chat(store: dict) -> dict:
    if len(store["chats"]) >= MAX_CHATS:
        raise ValueError(f"最多保留{MAX_CHATS}段会话。请导出备份并删除不再需要的会话。")
    chat = make_chat()
    store["chats"].insert(0, chat)
    store["active_id"] = chat["id"]
    return chat


def switch_chat(store: dict, chat_id: str) -> dict:
    if not any(chat["id"] == chat_id for chat in store["chats"]):
        raise ValueError("无法切换到不存在的会话")
    store["active_id"] = chat_id
    return active_chat(store)


def rename_chat(store: dict, chat_id: str, title: str) -> str:
    name = normalize_title(title)
    for chat in store["chats"]:
        if chat["id"] == chat_id:
            chat["title"] = name
            chat["updated_at"] = now_iso()
            return name
    raise ValueError("会话不存在")


def delete_chat(store: dict, chat_id: str) -> None:
    if not any(chat["id"] == chat_id for chat in store["chats"]):
        raise ValueError("会话不存在")
    store["chats"] = [ch for ch in store["chats"] if ch["id"] != chat_id]
    if not store["chats"]:
        store["chats"] = [make_chat()]
    if store["active_id"] == chat_id:
        store["active_id"] = store["chats"][0]["id"]


def sync_active(
    store: dict, messages: list[dict], user: str, conversation_id: str,
    tools: list[str] | None = None, trace_source: str = ""
) -> None:
    chat = active_chat(store)
    if not USER_RE.fullmatch(user) or not CONV_RE.fullmatch(conversation_id):
        raise ValueError("会话标识无效")
    chat["messages"] = messages[-MAX_MESSAGES:]
    chat["dify_user"] = user
    chat["dify_conversation_id"] = conversation_id
    chat["recent_tool_events"] = list(tools or [])[:12]
    chat["recent_tool_trace_source"] = str(trace_source)[:120]
    if chat["title"] == "新对话":
        first = next((msg["content"] for msg in chat["messages"]
                      if isinstance(msg,dict) and msg.get("role") == "user"),None)
        if first:
            chat["title"] = auto_title(first)
    chat["updated_at"] = now_iso()


def from_legacy(raw) -> dict:
    validated = validate_legacy(raw)
    result = new_store()
    chat = active_chat(result)
    chat.update({
        "title": auto_title(next((m["content"] for m in validated["messages"]
                                  if m["role"] == "user"), "已恢复的对话")),
        "messages": validated["messages"],
        "dify_user": validated["user"],
        "dify_conversation_id": validated["conversation_id"],
        "recent_tool_events": validated["tool_events"],
        "recent_tool_trace_source": validated["trace_source"],
    })
    return result


def validate_store(raw) -> dict:
    """Strictly validate untrusted JSON before putting it in session state."""
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_BYTES:
            raise ValueError("历史记录超过浏览器容量限制")
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("会话JSON格式无效") from exc
    if not isinstance(raw, dict):
        raise ValueError("历史记录格式不正确")
    if raw.get("schema") == 1:
        return from_legacy(raw)
    if raw.get("schema") != 2:
        raise ValueError("不支持的会话文件版本")
    if len(json.dumps(raw, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        raise ValueError("历史记录超过大小限制")
    chats = raw.get("chats")
    if not isinstance(chats, list) or not 1 <= len(chats) <= MAX_CHATS:
        raise ValueError("会话数量不合法")
    ids: set[str] = set()
    cleaned = []
    for ch in chats:
        if not isinstance(ch, dict) or not isinstance(ch.get("id"), str) or not ID_RE.fullmatch(ch["id"]):
            raise ValueError("非法会话标识")
        if ch["id"] in ids:
            raise ValueError("存在重复会话标识")
        ids.add(ch["id"])
        title = ch.get("title")
        if not isinstance(title, str) or len(title) > MAX_TITLE*2:
            raise ValueError("会话标题无效")
        user = ch.get("dify_user")
        conv = ch.get("dify_conversation_id", "")
        if not isinstance(user, str) or not USER_RE.fullmatch(user):
            raise ValueError("Dify用户标识无效")
        if not isinstance(conv, str) or not CONV_RE.fullmatch(conv):
            raise ValueError("Dify会话ID无效")
        msgs = ch.get("messages")
        if not isinstance(msgs, list) or len(msgs) > MAX_MESSAGES:
            raise ValueError("对话消息数超过限制")
        clean_msgs = []
        for item in msgs:
            if not isinstance(item,dict) or item.get("role") not in ("user","assistant"):
                raise ValueError("存在非法角色")
            content = item.get("content")
            if not isinstance(content,str) or len(content)>MAX_TEXT:
                raise ValueError("会话消息无效")
            tools=item.get("tools",[])
            if not isinstance(tools,list) or len(tools)>12 or any(
                not isinstance(t,str) or not TOOL_RE.fullmatch(t) for t in tools
            ):
                raise ValueError("非法工具记录")
            message={"role":item["role"],"content":content,"tools":tools}
            if isinstance(item.get("tool_trace_source"),str):
                message["tool_trace_source"]=item["tool_trace_source"][:120]
            clean_msgs.append(message)
        events=ch.get("recent_tool_events",[])
        if not isinstance(events,list) or len(events)>12 or any(
            not isinstance(t,str) or not TOOL_RE.fullmatch(t) for t in events
        ):
            raise ValueError("工具事件不合法")
        for key in ("created_at","updated_at"):
            if not isinstance(ch.get(key),str) or len(ch[key])>70:
                raise ValueError("会话时间无效")
        # Whitelist the archive format; imported JSON cannot inject unrelated
        # local/private fields into browser storage or subsequent backups.
        cleaned.append({
            "id":ch["id"],
            "title":normalize_title(title),
            "messages":clean_msgs,
            "dify_user":user,
            "dify_conversation_id":conv,
            "recent_tool_events":events,
            "recent_tool_trace_source":str(ch.get("recent_tool_trace_source",""))[:120],
            "created_at":ch["created_at"],
            "updated_at":ch["updated_at"],
        })
    active=raw.get("active_id")
    if active not in ids:
        raise ValueError("当前会话标识无法匹配")
    return {"schema":2,"active_id":active,"chats":cleaned}


def to_json_bytes(store: dict) -> bytes:
    checked=validate_store(store)
    return json.dumps(checked, ensure_ascii=False,indent=2).encode("utf-8")
