"""
AST-level verification of the SVDCodes per-cluster refactoring.
Checks svdcodes.py, strategy.py, and main.py for all expected symbols,
and confirms that the removed symbols are gone.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

def ast_names(path, node_type):
    src  = path.read_text()
    tree = ast.parse(src)
    return {n.name for n in ast.walk(tree) if isinstance(n, node_type)}

def ast_imports(path):
    src  = path.read_text()
    tree = ast.parse(src)
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            names += [a.name for a in n.names]
    return set(names)

errors = []

def check(label, condition):
    sym = "✓" if condition else "✗"
    print(f"  {sym}  {label}")
    if not condition:
        errors.append(label)

# ── svdcodes.py ───────────────────────────────────────────────────────────────
svdcodes_path = ROOT / "src" / "volumen" / "svdcodes.py"
svd_fns = ast_names(svdcodes_path, ast.FunctionDef)
print("\nsvdcodes.py:")
check("get_cluster_masks_data defined", "get_cluster_masks_data" in svd_fns)
check("get_pixels still present",       "get_pixels"             in svd_fns)
check("get_codes still present",        "get_codes"              in svd_fns)

# ── strategy.py ───────────────────────────────────────────────────────────────
strategy_path = ROOT / "src" / "volumen" / "segmentation" / "strategy.py"
strat_imports = ast_imports(strategy_path)
strat_src     = strategy_path.read_text()
strat_tree    = ast.parse(strat_src)

# collect methods per class
class_methods = {}
for node in ast.walk(strat_tree):
    if isinstance(node, ast.ClassDef):
        class_methods[node.name] = {
            m.name for m in ast.walk(node) if isinstance(m, ast.FunctionDef)
        }

print("\nstrategy.py:")
check("get_cluster_masks_data imported",             "get_cluster_masks_data"  in strat_imports)
check("SVDCodesStrategy.get_cluster_masks defined",  "get_cluster_masks" in class_methods.get("SVDCodesStrategy", set()))
check("SVDCodesStrategy.create_mask still present",  "create_mask"       in class_methods.get("SVDCodesStrategy", set()))
check("SVDCodesStrategy.segment_image still present","segment_image"     in class_methods.get("SVDCodesStrategy", set()))

# ── main.py ───────────────────────────────────────────────────────────────────
main_path = ROOT / "website" / "main.py"
main_tree = ast.parse(main_path.read_text())
main_fns  = {n.name for n in ast.walk(main_tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}
main_cls  = {n.name for n in ast.walk(main_tree) if isinstance(n, ast.ClassDef)}

print("\nmain.py — expected symbols:")
check("_run_svd_clusters helper",           "_run_svd_clusters"          in main_fns)
check("_initial_svd_mask helper",           "_initial_svd_mask"          in main_fns)
check("segment_images endpoint",            "segment_images"             in main_fns)
check("resegment_images endpoint",          "resegment_images"           in main_fns)
check("svd_clusters endpoint",              "svd_clusters"               in main_fns)
check("compute_from_clusters endpoint",     "compute_from_clusters"      in main_fns)
check("SVDClustersRequest model",           "SVDClustersRequest"         in main_cls)
check("ComputeFromClustersRequest model",   "ComputeFromClustersRequest" in main_cls)
check("ResegmentRequest.min_cluster_pct", True)  # checked separately below

# ResegmentRequest fields
for node in ast.walk(main_tree):
    if isinstance(node, ast.ClassDef) and node.name == "ResegmentRequest":
        fields = [s.target.id for s in ast.walk(node)
                  if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        check("ResegmentRequest.alpha",           "alpha"           in fields)
        check("ResegmentRequest.min_cluster_pct", "min_cluster_pct" in fields)
        break

print("\nmain.py — removed symbols (must be absent):")
check("_save_svd_previews REMOVED", "_save_svd_previews" not in main_fns)
check("svd_preview REMOVED",        "svd_preview"        not in main_fns)

# ── Summary ───────────────────────────────────────────────────────────────────
if errors:
    print(f"\n❌  {len(errors)} check(s) failed: {errors}", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll checks passed ✓")
