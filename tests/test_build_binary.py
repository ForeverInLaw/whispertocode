import importlib
from pathlib import Path
import shutil
import types
import unittest
from unittest import mock


build_binary = importlib.import_module("build_binary")


class BuildBinaryTests(unittest.TestCase):
    def test_build_uses_noconsole_on_windows(self):
        root = (Path.cwd() / ".tmp_build_binary_test").resolve()
        dist_dir = root / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "riva-ptt.exe").write_bytes(b"binary")
        try:
            with (
                mock.patch(
                    "build_binary.parse_args",
                    return_value=types.SimpleNamespace(name="riva-ptt", artifact_tag="windows"),
                ),
                mock.patch("build_binary.Path") as mock_path,
                mock.patch("build_binary.shutil.rmtree"),
                mock.patch("build_binary.shutil.copy2"),
                mock.patch("build_binary.subprocess.run") as mock_run,
                mock.patch("build_binary.os.name", "nt"),
                mock.patch("build_binary.platform.system", return_value="Windows"),
                mock.patch("builtins.print"),
            ):
                mock_path.return_value.resolve.return_value.parent = root
                result = build_binary.main()
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(result, 0)
        cmd = mock_run.call_args.kwargs.get("args", mock_run.call_args.args[0])
        self.assertIn("--noconsole", cmd)
        self.assertIn("PySide6.QtCore", cmd)
        self.assertEqual(cmd[-1], "run_whispertocode.py")


if __name__ == "__main__":
    unittest.main()
