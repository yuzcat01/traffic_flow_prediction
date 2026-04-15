import os
import sys
from pathlib import Path


def setup_project_paths():
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Keep relative config/data paths stable no matter where command is launched.
    os.chdir(project_root)

