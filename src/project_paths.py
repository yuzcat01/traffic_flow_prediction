from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root(project_root: Optional[PathLike] = None) -> Path:
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root).resolve()


def resolve_project_path(path: PathLike, project_root: Optional[PathLike] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_project_root(project_root) / candidate).resolve()


def to_project_relative_path(path: PathLike, project_root: Optional[PathLike] = None) -> str:
    resolved = resolve_project_path(path, project_root=project_root)
    root = get_project_root(project_root)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)
