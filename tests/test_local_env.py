import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import local_env  # noqa: E402


class TestLocalEnv(unittest.TestCase):
    def test_save_and_load_dotenv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            game_dir = Path(tmp) / "xbox_copy"
            game_dir.mkdir()
            (game_dir / "MicrosoftGame.Config").write_text("<Game/>", encoding="utf-8")
            (game_dir / "levels.rpf").write_bytes(b"RPF9")

            local_env.save_dotenv({"GTA6_GAME_DIR": str(game_dir)}, env_path=env_path)
            loaded = local_env.load_dotenv(env_path=env_path)
            self.assertEqual(loaded["GTA6_GAME_DIR"], str(game_dir))
            self.assertEqual(loaded["GTA6_DATA_DIR"], "local-data")
            self.assertEqual(loaded["GTA6_WORK_DIR"], "work")

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GTA6_GAME_DIR", None)
                report = local_env.check_environment(env_path=env_path)
                info = report["game_dir"]
                self.assertTrue(info["configured"])
                self.assertTrue(info["exists"])
                self.assertTrue(info["has_microsoft_game_config"])
                self.assertEqual(info["rpf_count"], 1)

    def test_unconfigured_game_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GTA6_GAME_DIR", None)
                report = local_env.check_environment(env_path=env_path)
                self.assertFalse(report["game_dir"]["configured"])
                self.assertFalse(report["game_dir"]["exists"])


if __name__ == "__main__":
    unittest.main()
