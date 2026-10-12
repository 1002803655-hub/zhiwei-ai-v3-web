"""Reproduce Streamlit Cloud launching the V4 app from its subdirectory."""
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StreamlitEntrypointTests(unittest.TestCase):
    def test_entrypoint_imports_project_modules_with_only_app_dir_on_path(self):
        # Streamlit is stubbed so this test checks startup imports without requiring
        # a browser or the Streamlit server. Imports run in a clean child process.
        script = r'''
import pathlib, runpy, sys, types
root = pathlib.Path(sys.argv[1]).resolve()
entry_dir = root / "lovis_v40"
sys.path[:] = [str(entry_dir)] + [p for p in sys.path if str(root) not in p]
st = types.ModuleType("streamlit")
st.set_page_config = lambda **kwargs: None
st.title = lambda *args, **kwargs: None
st.caption = lambda *args, **kwargs: None
st.text_area = lambda label, value="", **kwargs: value
st.button = lambda *args, **kwargs: False
sys.modules["streamlit"] = st
namespace = runpy.run_path(str(entry_dir / "streamlit_app.py"), run_name="__main__")
assert callable(namespace["analyze"])
assert isinstance(namespace["IntegrationError"], type)
assert sys.path.index(str(root)) < sys.path.index(str(entry_dir))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script, str(ROOT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"entrypoint import failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

