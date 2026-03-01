import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    NEMOTRON_REASONING_BUDGET_DEFAULT,
    NEMOTRON_REASONING_BUDGET_MAX,
    NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT,
    NEMOTRON_REASONING_PRINT_LIMIT_MAX,
)

DEFAULT_RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
DEFAULT_RIVA_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
DEFAULT_NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


@dataclass(frozen=True)
class AppSettings:
    nvidia_api_key: str = ""
    riva_server: str = DEFAULT_RIVA_SERVER
    riva_function_id: str = DEFAULT_RIVA_FUNCTION_ID
    nemotron_base_url: str = DEFAULT_NEMOTRON_BASE_URL
    nemotron_model: str = DEFAULT_NEMOTRON_MODEL
    nemotron_temperature: float = 1.0
    nemotron_top_p: float = 1.0
    nemotron_max_tokens: int = 16384
    nemotron_reasoning_budget: int = NEMOTRON_REASONING_BUDGET_DEFAULT
    nemotron_reasoning_print_limit: int = NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT
    nemotron_enable_thinking: bool = True


def get_config_dir() -> Path:
    if os.name == "nt":
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "WhisperToCode"
        return Path.home() / "AppData" / "Roaming" / "WhisperToCode"

    if os.name == "posix" and sys_platform_startswith("darwin"):
        return Path.home() / "Library" / "Application Support" / "WhisperToCode"

    xdg = os.getenv("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "whispertocode"
    return Path.home() / ".config" / "whispertocode"


def get_config_path() -> Path:
    return get_config_dir() / "config.json"


def load_config_json() -> dict[str, Any]:
    path = get_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_config_json(settings: AppSettings) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, ensure_ascii=False, indent=2)


def load_env_fallback(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = env if env is not None else os.environ
    keys = [
        "NVIDIA_API_KEY",
        "RIVA_SERVER",
        "RIVA_FUNCTION_ID",
        "NEMOTRON_BASE_URL",
        "NEMOTRON_MODEL",
        "NEMOTRON_TEMPERATURE",
        "NEMOTRON_TOP_P",
        "NEMOTRON_MAX_TOKENS",
        "NEMOTRON_REASONING_BUDGET",
        "NEMOTRON_REASONING_PRINT_LIMIT",
        "NEMOTRON_ENABLE_THINKING",
    ]
    result: dict[str, str] = {}
    for key in keys:
        raw = source.get(key)
        if raw is None:
            continue
        result[key] = str(raw).strip()
    return result


def resolve_settings(config_json: Mapping[str, Any], env_map: Mapping[str, str]) -> AppSettings:
    cfg = dict(config_json or {})
    env = dict(env_map or {})

    def _pick_str(cfg_key: str, env_key: str, default: str) -> str:
        cfg_value = cfg.get(cfg_key)
        if isinstance(cfg_value, str):
            stripped = cfg_value.strip()
            if stripped:
                return stripped
        env_value = env.get(env_key)
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()
        return default

    def _pick_float(cfg_key: str, env_key: str, default: float) -> float:
        cfg_value = cfg.get(cfg_key)
        if isinstance(cfg_value, (int, float)):
            return float(cfg_value)
        cfg_text = cfg_value if isinstance(cfg_value, str) else None
        if cfg_text is not None:
            try:
                return float(cfg_text.strip())
            except ValueError:
                pass
        env_text = env.get(env_key)
        if isinstance(env_text, str):
            try:
                return float(env_text.strip())
            except ValueError:
                pass
        return default

    def _pick_int(cfg_key: str, env_key: str, default: int) -> int:
        cfg_value = cfg.get(cfg_key)
        if isinstance(cfg_value, int):
            return cfg_value
        cfg_text = cfg_value if isinstance(cfg_value, str) else None
        if cfg_text is not None:
            try:
                return int(cfg_text.strip())
            except ValueError:
                pass
        env_text = env.get(env_key)
        if isinstance(env_text, str):
            try:
                return int(env_text.strip())
            except ValueError:
                pass
        return default

    def _pick_bool(cfg_key: str, env_key: str, default: bool) -> bool:
        cfg_value = cfg.get(cfg_key)
        if isinstance(cfg_value, bool):
            return cfg_value
        if isinstance(cfg_value, str):
            parsed = _parse_bool(cfg_value)
            if parsed is not None:
                return parsed
        env_text = env.get(env_key)
        if isinstance(env_text, str):
            parsed = _parse_bool(env_text)
            if parsed is not None:
                return parsed
        return default

    reasoning_budget = _pick_int(
        "nemotron_reasoning_budget",
        "NEMOTRON_REASONING_BUDGET",
        NEMOTRON_REASONING_BUDGET_DEFAULT,
    )
    reasoning_budget = max(0, min(reasoning_budget, NEMOTRON_REASONING_BUDGET_MAX))

    reasoning_print_limit = _pick_int(
        "nemotron_reasoning_print_limit",
        "NEMOTRON_REASONING_PRINT_LIMIT",
        NEMOTRON_REASONING_PRINT_LIMIT_DEFAULT,
    )
    reasoning_print_limit = max(
        0,
        min(reasoning_print_limit, NEMOTRON_REASONING_PRINT_LIMIT_MAX),
    )

    return AppSettings(
        nvidia_api_key=_pick_str("nvidia_api_key", "NVIDIA_API_KEY", ""),
        riva_server=_pick_str("riva_server", "RIVA_SERVER", DEFAULT_RIVA_SERVER),
        riva_function_id=_pick_str(
            "riva_function_id",
            "RIVA_FUNCTION_ID",
            DEFAULT_RIVA_FUNCTION_ID,
        ),
        nemotron_base_url=_pick_str(
            "nemotron_base_url",
            "NEMOTRON_BASE_URL",
            DEFAULT_NEMOTRON_BASE_URL,
        ),
        nemotron_model=_pick_str(
            "nemotron_model",
            "NEMOTRON_MODEL",
            DEFAULT_NEMOTRON_MODEL,
        ),
        nemotron_temperature=_pick_float(
            "nemotron_temperature",
            "NEMOTRON_TEMPERATURE",
            1.0,
        ),
        nemotron_top_p=_pick_float("nemotron_top_p", "NEMOTRON_TOP_P", 1.0),
        nemotron_max_tokens=_pick_int(
            "nemotron_max_tokens",
            "NEMOTRON_MAX_TOKENS",
            16384,
        ),
        nemotron_reasoning_budget=reasoning_budget,
        nemotron_reasoning_print_limit=reasoning_print_limit,
        nemotron_enable_thinking=_pick_bool(
            "nemotron_enable_thinking",
            "NEMOTRON_ENABLE_THINKING",
            True,
        ),
    )


def sys_platform_startswith(prefix: str) -> bool:
    return os.sys.platform.startswith(prefix)


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
