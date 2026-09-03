from pathlib import Path

from app.utils import escape_milvus_string_utils


expected_root = Path(__file__).resolve().parent.parent
module_path = Path(escape_milvus_string_utils.__file__).resolve()
assert module_path == expected_root / "app" / "utils" / "escape_milvus_string_utils.py"
print(f"app import succeeded; module path: {module_path}")
