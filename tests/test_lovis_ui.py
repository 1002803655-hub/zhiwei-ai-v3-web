# -*- coding: utf-8 -*-
"""Smoke check the Lovis home page without calling any external API."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class HomeTests(unittest.TestCase):
    def test_home_without_data_or_secrets(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / "lovis_app.py"),default_timeout=35).run()
        self.assertEqual(len(app.exception),0,
                         "Streamlit raised a runtime exception: "+str(app.exception))
        self.assertTrue(any("Lovis" in item.value for item in app.markdown))
        self.assertTrue(len(app.chat_input)>=1)

if __name__=="__main__":
    unittest.main()
