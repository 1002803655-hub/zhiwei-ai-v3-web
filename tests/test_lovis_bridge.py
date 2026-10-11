# -*- coding: utf-8 -*-
"""Safe mocked tests. Do not call live Dify or download CWRU datasets."""
import json
import unittest
from lovis_bridge import ask_lovis, combine_question_with_vibration, vibration_summary


class FakeResponse:
    status_code=200
    headers={"Content-Type":"text/event-stream"}
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def iter_lines(self, decode_unicode=True):
        for obj in [
            {"event":"agent_thought","tool":"motor_knowledge_search","thought":"NEVER EXPOSE THIS","conversation_id":"conv-123"},
            {"event":"message","answer":"诊断应结合现场数据。","conversation_id":"conv-123"},
            {"event":"message_end","message_id":"msg-456","conversation_id":"conv-123"},
        ]:
            yield "data: "+json.dumps(obj,ensure_ascii=False)


class FakeClient:
    payload=None
    def post(self,url,headers=None,json=None,**kwargs):
        self.payload=json
        assert url=="https://api.dify.ai/v1/chat-messages"
        assert headers["Authorization"]=="Bearer TEST_ONLY"
        return FakeResponse()


class BridgeTests(unittest.TestCase):
    def test_sse_roundtrip_and_context(self):
        fake=FakeClient()
        result=ask_lovis("你好","TEST_ONLY",user="visitor-1",conversation_id="",session=fake)
        self.assertEqual(result["conversation_id"],"conv-123")
        self.assertEqual(result["tool_events"],["motor_knowledge_search"])
        self.assertNotIn("NEVER EXPOSE",str(result))
        self.assertEqual(result["message_id"],"msg-456")
        ask_lovis("请继续","TEST_ONLY",user="visitor-1",
                  conversation_id=result["conversation_id"],session=fake)
        self.assertEqual(fake.payload["conversation_id"],"conv-123")
        self.assertEqual(fake.payload["query"],"请继续")

    def test_summary_redacts_source(self):
        meta={"name":"secret_filename.mat","raw_sha256":"PRIVATEHASH","source_url":"file:///private/secret",
              "source_fs_hz":12000}
        data={"rms_signal_units":0.22,"kurtosis_pearson":4.3,"crest_factor":5,
              "dominant_fft_hz":1300}
        pred={"status":"predicted","predicted_label":"内圈缺陷","vote_counts":{"inner_race":8}}
        summ=vibration_summary(meta,data,pred)
        self.assertNotIn("secret_filename",summ)
        self.assertNotIn("PRIVATEHASH",summ)
        self.assertNotIn("/private",summ)
        self.assertIn("inner_race",summ)
        prompt=combine_question_with_vibration("解释当前样本",summ)
        self.assertIn("振动结果",prompt)
        self.assertIn("用户本轮问题",prompt)


if __name__=="__main__":
    unittest.main()
