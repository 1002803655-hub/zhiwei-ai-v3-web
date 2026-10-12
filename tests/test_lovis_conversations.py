# -*- coding: utf-8 -*-
"""Independent conversation histories and migration regressions."""
import json
import unittest
from lovis_history import build_snapshot
from lovis_conversations import (
    active_chat, add_chat, auto_title, delete_chat, from_legacy, make_chat,
    MAX_CHATS, new_store, rename_chat, switch_chat, sync_active,
    to_json_bytes, validate_store,
)

class MultiConversationTests(unittest.TestCase):
    def test_two_chats_never_share_dify_context(self):
        archive = new_store()
        a=active_chat(archive)
        id_a, user_a = a["id"], a["dify_user"]
        sync_active(archive,[{"role":"user","content":"电机A发生振动异常"}],
                    user_a,"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    ["motor_knowledge_search"],"Dify实时API事件")
        b=add_chat(archive)
        self.assertNotEqual(b["dify_user"],user_a)
        self.assertEqual(b["dify_conversation_id"],"")
        sync_active(archive,[{"role":"user","content":"电机B的温度升高"}],
                    b["dify_user"],"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    ["motor_trend_analysis_v23"],"Dify实时API事件")
        back=switch_chat(archive,id_a)
        self.assertEqual(back["dify_user"],user_a)
        self.assertEqual(back["dify_conversation_id"],"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        self.assertIn("电机A",back["messages"][0]["content"])
        self.assertEqual(back["recent_tool_events"],["motor_knowledge_search"])
        valid=validate_store(to_json_bytes(archive).decode("utf-8"))
        self.assertEqual(len(valid["chats"]),2)
        self.assertEqual(active_chat(valid)["id"],id_a)
        again=switch_chat(valid,b["id"])
        self.assertEqual(again["dify_conversation_id"],"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        self.assertIn("电机B",again["messages"][0]["content"])

    def test_auto_titles_and_rename(self):
        store=new_store()
        item=active_chat(store)
        sync_active(store,[{"role":"user","content":"轴承温度一直升高，帮我分析\n更多信息"}],
                    item["dify_user"],"")
        self.assertEqual(item["title"],"轴承温度一直升高，帮我分析")
        self.assertEqual(rename_chat(store,item["id"],"  电机问题   记录 A "), "电机问题 记录 A")
        sync_active(store,[{"role":"user","content":"其它问题"}],item["dify_user"],"")
        self.assertEqual(item["title"],"电机问题 记录 A")
        self.assertEqual(auto_title("time,current_a,current_b,current_c,bearing_temp"),"工业电机数据分析")

    def test_legacy_v1_backup_import(self):
        backup=build_snapshot([{"role":"user","content":"我昨天讨论的电机M-01"}],
                              "lovis-"+"a"*32,"de7e8bac-2ace-4ec5-b62b-14bb179266a9")
        upgraded=validate_store(json.dumps(backup))
        self.assertEqual(upgraded["schema"],2)
        self.assertIn("电机M-01",active_chat(upgraded)["title"])
        self.assertEqual(active_chat(upgraded)["dify_conversation_id"],backup["dify_conversation_id"])

    def test_archive_rejects_foreign_chat_id(self):
        data=new_store()
        data["active_id"]="f"*32
        with self.assertRaises(ValueError):
            validate_store(data)

    def test_archive_rejects_leakage_and_duplicate_chat_id(self):
        data=new_store()
        extra=make_chat()
        extra["id"]=data["chats"][0]["id"]
        data["chats"].append(extra)
        with self.assertRaises(ValueError):
            validate_store(data)

    def test_add_delete_and_cap(self):
        data=new_store()
        initial=active_chat(data)["id"]
        b=add_chat(data)
        self.assertEqual(active_chat(data)["id"],b["id"])
        delete_chat(data,b["id"])
        self.assertEqual(active_chat(data)["id"],initial)
        delete_chat(data,initial)
        self.assertEqual(len(data["chats"]),1)
        self.assertNotEqual(active_chat(data)["id"],initial)
        for _ in range(MAX_CHATS-1):
            add_chat(data)
        with self.assertRaises(ValueError):
            add_chat(data)

    def test_full_backup_excludes_secrets_and_raw_attachment(self):
        data=new_store()
        item=active_chat(data)
        sync_active(data,[{"role":"user","content":"轴承异响"}],
                    item["dify_user"],"")
        serialized=to_json_bytes(data)
        self.assertNotIn(b"DIFY_API_KEY",serialized)
        self.assertNotIn(b"file_uploader",serialized)
        self.assertNotIn(b"raw_sha256",serialized)
        self.assertNotIn(b"waveform",serialized)

if __name__ == "__main__":
    unittest.main()
