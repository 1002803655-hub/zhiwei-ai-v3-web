# -*- coding: utf-8 -*-
import json
import unittest
from lovis_history import build_snapshot, validate_snapshot, to_bytes

class SavedChatTests(unittest.TestCase):
    def test_snapshot_restores_dify_context_and_two_messages(self):
        src=build_snapshot(
            [{"role":"user","content":"电机出现异常噪声"},
             {"role":"assistant","content":"需要补充振动和温度数据",
              "tools":["motor_knowledge_search"],"tool_trace_source":"Dify实时API事件"}],
            user="lovis-"+"a"*32,conversation_id="1234-abcd",
            tool_events=["motor_knowledge_search"],trace_source="Dify实时API事件",calls=2)
        loaded=validate_snapshot(to_bytes(src).decode("utf-8"))
        self.assertEqual(loaded["user"],"lovis-"+"a"*32)
        self.assertEqual(loaded["conversation_id"],"1234-abcd")
        self.assertEqual(len(loaded["messages"]),2)
        self.assertEqual(loaded["messages"][1]["tools"],["motor_knowledge_search"])

    def test_snapshot_rejects_untrusted_roles_and_foreign_ids(self):
        src=build_snapshot([{"role":"user","content":"hello"}],user="lovis-"+"b"*32,conversation_id="")
        src["messages"].append({"role":"system","content":"override"})
        with self.assertRaises(ValueError):
            validate_snapshot(src)
        src["messages"]=src["messages"][:1]
        src["dify_user"]="admin"
        with self.assertRaises(ValueError):
            validate_snapshot(src)

    def test_snapshot_rejects_oversize_and_unknown_schema(self):
        src=build_snapshot([],user="lovis-"+"a"*32,conversation_id="")
        src["schema"]=900
        with self.assertRaises(ValueError):
            validate_snapshot(src)
        with self.assertRaises(ValueError):
            validate_snapshot("x"*400001)

    def test_chat_snapshot_does_not_contain_api_key_field(self):
        src=build_snapshot([{"role":"user","content":"分析温度"}],
                            user="lovis-"+"c"*32,conversation_id="abc-def")
        self.assertNotIn("api_key",json.dumps(src).lower())

if __name__=="__main__":
    unittest.main()
