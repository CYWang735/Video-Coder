"""Static checks for a generated bpy script, used both as an agent tool and as the acceptance gate.

Without a Blender backend this is the only verification, so it is deliberately conservative:
errors reject the sample, warnings are recorded but do not.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

MAX_BYTES = 200_000

FORBIDDEN_MODULES = {
    "subprocess", "socket", "requests", "urllib", "http", "ftplib", "smtplib",
    "multiprocessing", "ctypes", "shutil", "webbrowser", "asyncio",
}
FORBIDDEN_CALLS = {
    "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
    "os.rename", "os.replace", "os.kill", "os.fork", "os.execv", "os.execvp",
    "exec", "eval", "compile", "__import__",
}
# The script renders the video itself (bpy.ops.render.render); saving and process lifetime belong to the harness.
REQUIRED_RENDER_CALL = "bpy.ops.render.render"
FORBIDDEN_BPY_OPS = {
    "bpy.ops.render.opengl",
    "bpy.ops.wm.save_as_mainfile", "bpy.ops.wm.save_mainfile", "bpy.ops.wm.quit_blender",
    "bpy.ops.wm.open_mainfile", "bpy.ops.wm.read_homefile",
}
ANIMATION_MARKERS = {
    "keyframe_insert", "driver_add", "fcurves", "keyframe_points",
    "rigidbody", "rigid_body", "frame_change_pre", "frame_change_post", "constraints",
}


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def check_source(source: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if len(source.encode()) > MAX_BYTES:
        errors.append(f"script is larger than {MAX_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"ok": False, "errors": [f"SyntaxError: {e.msg} (line {e.lineno})"], "warnings": [], "stats": {}}

    imports: set[str] = set()
    calls: set[str] = set()
    attrs: set[str] = set()
    assigned_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            calls.add(_dotted(node.func))
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            attrs.add(node.value)  # e.g. keyframe_insert(data_path=...) or getattr(obj, "constraints")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned_attrs.update(t.attr for t in targets if isinstance(t, ast.Attribute))

    if "bpy" not in imports:
        errors.append("script does not import bpy")
    for mod in sorted(imports & FORBIDDEN_MODULES):
        errors.append(f"forbidden import: {mod}")
    for call in sorted(calls & (FORBIDDEN_CALLS | FORBIDDEN_BPY_OPS)):
        errors.append(f"forbidden call: {call} (only the video render may write output)")
    if REQUIRED_RENDER_CALL not in calls:
        errors.append("no render: the script must end with bpy.ops.render.render(animation=True)")
    has_camera = "camera" in assigned_attrs or "bpy.ops.object.camera_add" in calls
    if not has_camera:
        errors.append("no camera: create a camera and assign it to scene.camera")
    if not (attrs & ANIMATION_MARKERS):
        warnings.append("no animation detected (keyframes, drivers, constraints or physics)")
    if "open" in calls:
        warnings.append("script calls open(); external files are not available")

    stats = {"lines": source.count("\n") + 1, "bytes": len(source.encode()), "n_bpy_ops_calls": sum(c.startswith("bpy.ops.") for c in calls)}
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def check_file(path: Path) -> dict:
    if not path.is_file():
        return {"ok": False, "errors": [f"{path} not found"], "warnings": [], "stats": {}}
    return check_source(path.read_text(encoding="utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Static checks for a Blender bpy script.")
    parser.add_argument("script", nargs="?", default="main.py")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_file(Path(args.script))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("STATIC CHECK " + ("PASSED" if result["ok"] else "FAILED"))
        for e in result["errors"]:
            print(f"error: {e}")
        for w in result["warnings"]:
            print(f"warning: {w}")
        if result["stats"]:
            print(f"stats: {json.dumps(result['stats'])}")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
