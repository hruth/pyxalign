from typing import Optional

_default_dialog_dir: Optional[str] = None


def set_default_dialog_dir(path: Optional[str]) -> None:
    global _default_dialog_dir
    _default_dialog_dir = path


def get_default_dialog_dir() -> str:
    return _default_dialog_dir or ""
