import importlib
import unittest
from unittest import mock


class PackageEntrypointTests(unittest.TestCase):
    def test_package_main_delegates_to_app_main(self):
        main_module = importlib.import_module("whispertocode.__main__")
        with mock.patch("whispertocode.__main__.app_main", return_value=7):
            code = main_module.main()
        self.assertEqual(code, 7)

    def test_package_main_guard_uses_system_exit(self):
        main_module = importlib.import_module("whispertocode.__main__")
        fake_globals = {
            "__name__": "__main__",
            "main": mock.Mock(return_value=3),
            "SystemExit": SystemExit,
            "__builtins__": __builtins__,
        }
        with self.assertRaises(SystemExit) as exc:
            exec("raise SystemExit(main())", fake_globals)
        self.assertEqual(exc.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
