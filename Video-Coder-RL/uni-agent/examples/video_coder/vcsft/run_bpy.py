"""Execute a generated script with the `bpy` module and print a JSON scene summary.

Run with the bpy environment (Python 3.11 + bpy 4.5), not the project venv:
    LD_LIBRARY_PATH=$WORK/envs/bpy45-libs/lib $WORK/envs/bpy45/bin/python vcsft/run_bpy.py main.py
The script runs in --cwd (default: a fresh temporary directory); exit code 0 = executed without exception.
Normally called through vcsft.blender_exec, which sets up the environment, timeout and video checks.
"""

import argparse
import json
import os
import runpy
import sys
import tempfile
import time
import traceback
from pathlib import Path


def summarize(bpy, samples: int) -> dict:
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    objects = list(scene.objects)
    by_type: dict[str, int] = {}
    for o in objects:
        by_type[o.type] = by_type.get(o.type, 0) + 1
    animated = sorted(o.name for o in objects if o.animation_data and o.animation_data.action)
    start, end = scene.frame_start, scene.frame_end
    frames = sorted({start, end, *[round(start + i * (end - start) / (samples - 1)) for i in range(samples)]})

    meshes = [o for o in objects if o.type == "MESH"]
    positions: dict[str, list] = {}
    out_of_view: dict[str, list[int]] = {}
    n_faces = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for f in frames:
        scene.frame_set(f)
        for o in meshes:
            loc = o.matrix_world.translation
            if o.name in animated or (o.parent and o.parent.name in animated):
                positions.setdefault(o.name, []).append([f, *[round(v, 3) for v in loc]])
            if scene.camera:
                p = world_to_camera_view(scene, scene.camera, loc)
                if not (0 <= p.x <= 1 and 0 <= p.y <= 1 and p.z > 0):
                    out_of_view.setdefault(o.name, []).append(f)
    for o in meshes:
        n_faces += len(o.evaluated_get(depsgraph).data.polygons)
    return {
        "blender": bpy.app.version_string,
        "frame_range": [start, end],
        "fps": scene.render.fps,
        "camera": scene.camera.name if scene.camera else None,
        "n_objects": len(objects),
        "objects_by_type": by_type,
        "n_faces_evaluated": n_faces,
        "animated_objects": animated,
        "sampled_frames": frames,
        "moving_mesh_positions": positions,
        "mesh_origins_out_of_camera_view": out_of_view,
    }


def install_guards(bpy) -> None:
    """Turn the most common hard crash into a Python error with a line number.

    Mesh.from_pydata does not check indices: an edge or face that references a vertex that does not exist
    corrupts Blender's memory and kills the process (SIGABRT/SIGSEGV) with no traceback.
    """
    original = bpy.types.Mesh.from_pydata

    def checked_from_pydata(self, vertices, edges, faces, *args, **kwargs):
        n = len(vertices)
        for kind, items, min_len in (("edge", edges, 2), ("face", faces, 3)):
            for i, item in enumerate(items):
                idx = list(item)
                if len(idx) < min_len:
                    raise ValueError(f"from_pydata: {kind} {i} has {len(idx)} vertices, needs at least {min_len}")
                bad = [v for v in idx if not (isinstance(v, int) and 0 <= v < n)]
                if bad:
                    raise ValueError(f"from_pydata: {kind} {i} uses vertex index {bad[0]!r}, "
                                     f"but the mesh has only {n} vertices (valid indices 0..{n - 1})")
                if len(set(idx)) != len(idx):
                    raise ValueError(f"from_pydata: {kind} {i} repeats a vertex: {tuple(idx)}")
        return original(self, vertices, edges, faces, *args, **kwargs)

    bpy.types.Mesh.from_pydata = checked_from_pydata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script")
    p.add_argument("--samples", type=int, default=6, help="frames sampled for motion/visibility checks")
    p.add_argument("--cwd", help="directory to run the script in (it renders its output here)")
    p.add_argument("--result", help="write the JSON result to this file instead of stdout")
    args = p.parse_args()
    script = Path(args.script).resolve()

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=False)  # same starting point as `blender --factory-startup`
    install_guards(bpy)
    result = {"script": str(script)}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(args.cwd).resolve() if args.cwd else Path(tmp)
        os.chdir(cwd)
        before = set(os.listdir(cwd))
        t0 = time.time()
        try:
            runpy.run_path(str(script), run_name="__main__")
            result["ok"] = True
        except BaseException:  # noqa: BLE001  (SystemExit from the script counts as failure too)
            result["ok"] = False
            result["traceback"] = traceback.format_exc(limit=8)
        result["exec_seconds"] = round(time.time() - t0, 2)
        result["files_written"] = sorted(set(os.listdir(cwd)) - before)
        if result["ok"]:
            try:
                result["scene"] = summarize(bpy, args.samples)
            except Exception:  # noqa: BLE001  (a broken scene must not hide the execution result)
                result["scene_error"] = traceback.format_exc(limit=3)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.result:
        Path(args.result).write_text(text)
    else:
        print(text)
    # Skip bpy teardown: after a failed script it can hang on leaked data until the harness timeout.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
