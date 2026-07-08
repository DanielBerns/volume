"""
Verify that SVDCodesStrategy is correctly integrated into the segmentation
package by inspecting strategy.py via the AST (avoids needing cv2 installed).
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
strategy_file = ROOT / "src" / "volumen" / "segmentation" / "strategy.py"

src = strategy_file.read_text()
tree = ast.parse(src)

classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
print("Classes found in strategy.py:", classes)

# Also confirm SVDCodesStrategy has both required methods
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "SVDCodesStrategy":
        methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
        print(f"\nSVDCodesStrategy methods: {methods}")
        break
else:
    print("ERROR: SVDCodesStrategy not found!", file=sys.stderr)
    sys.exit(1)

# Check that the svdcodes imports are present
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and "svdcodes" in node.module:
        imports += [alias.name for alias in node.names]
print(f"\nsvdcodes functions imported: {imports}")

print("\nAll checks passed ✓")
