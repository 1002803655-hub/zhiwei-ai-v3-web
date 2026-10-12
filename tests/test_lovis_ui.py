# -*- coding: utf-8 -*-
"""Mock browser storage API exactly as streamlit-local-storage 0.0.25 exposes it."""
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from lovis_history import build_snapshot


class FakeBrowserStorage:
    saved = {}
    def __init__(self, key="storage_init"):
        pass
    def getItem(self, itemKey):
        # Deliberately no **kwargs: catches unsupported getItem(...,key="x").
        return self.saved.get(itemKey)
    def setItem(self, itemKey, itemValue, key="set"):
        self.saved[itemKey] = itemValue


def app_path():
    return str(Path(__file__).resolve().parents[1] / "lovis_app.py")


class HomeTests(unittest.TestCase):
    def setUp(self):
        FakeBrowserStorage.saved = {}

    def test_home_without_data_or_secrets(self):
        with patch("streamlit_local_storage.LocalStorage", FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
        self.assertEqual(len(app.exception),0,
                         "Streamlit raised: "+str(app.exception))
        self.assertTrue(any("Lovis" in item.value for item in app.markdown))
        self.assertTrue(len(app.chat_input)>=1)
        self.assertFalse(any("浏览器自动保存暂不可用" in w.message for w in app.warning))

    def test_sidebar_can_create_and_switch_independent_chats(self):
        with patch("streamlit_local_storage.LocalStorage", FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
            self.assertEqual(len(app.exception),0)
            original=app.session_state["lovis_chat_store"]["active_id"]
            create=next(btn for btn in app.button if btn.label=="＋ 开始新对话")
            app=create.click().run()
            self.assertEqual(len(app.exception),0)
            archive=app.session_state["lovis_chat_store"]
            self.assertEqual(len(archive["chats"]),2)
            second=archive["active_id"]
            self.assertNotEqual(original,second)
            choose=next(btn for btn in app.button if btn.key=="lovis_select_"+original)
            app=choose.click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.session_state["lovis_chat_store"]["active_id"],original)
            self.assertEqual(app.session_state["lovis_conversation_id"],"")

    def test_each_chat_has_ellipsis_rename_and_delete_menu(self):
        # Verify history actions are attached to their own chat ID, not a
        # global management panel (which could delete the wrong conversation).
        with patch("streamlit_local_storage.LocalStorage", FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
            self.assertEqual(len(app.exception),0)
            first=app.session_state["lovis_chat_store"]["active_id"]
            app=next(b for b in app.button if b.label=="＋ 开始新对话").click().run()
            self.assertEqual(len(app.exception),0)
            second=app.session_state["lovis_chat_store"]["active_id"]
            self.assertNotEqual(first,second)
            rename_ids={b.key for b in app.button if b.key and
                        b.key.startswith("lovis_menu_rename_")}
            delete_ids={b.key for b in app.button if b.key and
                        b.key.startswith("lovis_menu_delete_")}
            self.assertEqual(rename_ids,{"lovis_menu_rename_"+first,
                                         "lovis_menu_rename_"+second})
            self.assertEqual(delete_ids,{"lovis_menu_delete_"+first,
                                         "lovis_menu_delete_"+second})
            # Clicking the action must point to the selected, non-active chat.
            app=next(b for b in app.button if b.key=="lovis_menu_rename_"+first).click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.session_state["lovis_action_target"],
                             {"mode":"rename","id":first})
            self.assertEqual(app.session_state["lovis_chat_store"]["active_id"],second)

    def test_rename_from_ellipsis_does_not_switch_other_chat(self):
        with patch("streamlit_local_storage.LocalStorage",FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
            first=app.session_state["lovis_chat_store"]["active_id"]
            app=next(b for b in app.button if b.label=="＋ 开始新对话").click().run()
            second=app.session_state["lovis_chat_store"]["active_id"]
            app=next(b for b in app.button if b.key=="lovis_menu_rename_"+first).click().run()
            self.assertEqual(len(app.exception),0)
            edit=next(t for t in app.text_input if t.label=="对话名称")
            app=edit.set_value("一号电机历史分析").run()
            self.assertEqual(len(app.exception),0)
            app=next(b for b in app.button if b.label=="保存名称").click().run()
            self.assertEqual(len(app.exception),0)
            archive=app.session_state["lovis_chat_store"]
            self.assertEqual(archive["active_id"],second)
            name=next(c["title"] for c in archive["chats"] if c["id"]==first)
            self.assertEqual(name,"一号电机历史分析")
            self.assertNotIn("lovis_action_target",app.session_state._state)

    def test_deletion_requires_confirmation_and_preserves_other_chat(self):
        with patch("streamlit_local_storage.LocalStorage",FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
            first=app.session_state["lovis_chat_store"]["active_id"]
            app=next(b for b in app.button if b.label=="＋ 开始新对话").click().run()
            second=app.session_state["lovis_chat_store"]["active_id"]
            app=next(b for b in app.button if b.key=="lovis_menu_delete_"+first).click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state["lovis_chat_store"]["chats"]),2)
            self.assertEqual(app.session_state["lovis_action_target"],
                             {"mode":"delete","id":first})
            confirm=[b for b in app.button if b.label=="确认删除"]
            self.assertEqual(len(confirm),1)
            app=confirm[0].click().run()
            self.assertEqual(len(app.exception),0)
            remaining=app.session_state["lovis_chat_store"]
            self.assertEqual(len(remaining["chats"]),1)
            self.assertEqual(remaining["active_id"],second)
            self.assertEqual(remaining["chats"][0]["id"],second)

    def test_reopening_page_restores_browser_chat(self):
        from lovis_history import STORAGE_KEY
        backup=build_snapshot(
            [{"role":"user","content":"实验：轴承有异响"},
             {"role":"assistant","content":"建议记录转速与振动数据"}],
            "lovis-"+"b"*32,"11df34a5-30be-4be5-8732-1e7a1ffdd002"
        )
        FakeBrowserStorage.saved[STORAGE_KEY]=backup
        with patch("streamlit_local_storage.LocalStorage",FakeBrowserStorage):
            app=AppTest.from_file(app_path(),default_timeout=25).run()
        self.assertEqual(len(app.exception),0,
                         "Streamlit raised: "+str(app.exception))
        transcript=[m.value for m in app.markdown]
        self.assertTrue(any("实验：轴承有异响" in v for v in transcript))
        self.assertTrue(any("建议记录转速与振动数据" in v for v in transcript))
        self.assertEqual(app.session_state["lovis_conversation_id"],backup["dify_conversation_id"])

if __name__=="__main__":
    unittest.main()
