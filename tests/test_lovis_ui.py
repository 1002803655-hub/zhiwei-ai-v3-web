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
