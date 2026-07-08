"""
AST-level verification of the updated website/main.py.
Checks that all expected endpoints and models are present.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
main_file = ROOT / "website" / "main.py"

src = main_file.read_text()
tree = ast.parse(src)

# ── Collect all async function definitions (FastAPI route handlers) ──────────
functions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}

# ── Collect all class definitions (Pydantic models, etc.) ───────────────────
classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

# ── Check imports ────────────────────────────────────────────────────────────
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            imports.append(alias.name)

REQUIRED_FUNCTIONS = {
    "_make_strategy",
    "_save_svd_previews",
    "segment_images",
    "update_mask",
    "resegment_images",
    "svd_preview",
    "calculate_volume",
}

REQUIRED_CLASSES = {
    "UpdateMaskRequest",
    "ResegmentRequest",
    "SVDPreviewRequest",
    "CalculateVolumeRequest",
}

REQUIRED_IMPORTS = {"SVDCodesStrategy"}

errors = []

for fn in REQUIRED_FUNCTIONS:
    if fn in functions:
        print(f"  ✓ function  {fn}")
    else:
        print(f"  ✗ MISSING function  {fn}")
        errors.append(fn)

for cls in REQUIRED_CLASSES:
    if cls in classes:
        print(f"  ✓ class     {cls}")
    else:
        print(f"  ✗ MISSING class     {cls}")
        errors.append(cls)

for imp in REQUIRED_IMPORTS:
    if imp in imports:
        print(f"  ✓ import    {imp}")
    else:
        print(f"  ✗ MISSING import    {imp}")
        errors.append(imp)

# ── Check ResegmentRequest has alpha and view_type fields ────────────────────
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "ResegmentRequest":
        field_names = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_names.append(stmt.target.id)
        for field in ("alpha", "view_type"):
            if field in field_names:
                print(f"  ✓ field     ResegmentRequest.{field}")
            else:
                print(f"  ✗ MISSING field ResegmentRequest.{field}")
                errors.append(f"ResegmentRequest.{field}")

if errors:
    print(f"\n❌ {len(errors)} check(s) failed: {errors}", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll checks passed ✓")
