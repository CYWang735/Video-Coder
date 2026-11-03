"""Summarize a generation/evaluation run: per-task outcome, failure reason and tool usage, grouped by difficulty.

    .venv/bin/python -m vcsft.summarize runs/baseline_qwen35_9b [--json out.json]
"""

import argparse
import json
from pathlib import Path

from vcsft.export import normalize


def failure_reason(result: dict, task_dir: Path) -> str:
    if result["status"] == "accepted":
        return ""
    if result.get("error"):
        return f"{result['exit_status']}: {result['error'][:120]}"
    if result["exit_status"] != "Submitted":
        return result["exit_status"]  # LimitsExceeded / RepeatedFormatError / TimeExceeded
    if not result["static_check"]["ok"]:
        return "static: " + "; ".join(result["static_check"]["errors"])[:160]
    judge = result.get("judge")
    if judge and (result.get("execution") or {}).get("ok"):
        return f"judge {judge.get('verdict')} score={judge.get('score')}: {judge.get('summary', judge.get('error', ''))}"[:220]
    report_path = task_dir / "final_run" / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        tb = (report.get("traceback") or "").strip().splitlines()
        detail = tb[-1] if tb else report.get("error") or ""
        return ("exec: " + "; ".join(report.get("problems", [])) + (f" | {detail}" if detail else ""))[:220]
    return "rejected"


def tool_usage(traj: dict) -> dict:
    messages = normalize(traj["messages"])
    commands, outputs = [], {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            try:
                commands.append((tc["id"], json.loads(tc["function"]["arguments"]).get("command", "")))
            except (json.JSONDecodeError, TypeError):
                commands.append((tc["id"], ""))
        if m.get("role") == "tool":
            content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            outputs[m.get("tool_call_id")] = content
    runs = [(cid, c) for cid, c in commands if "run_blender" in c]
    return {
        "n_run_blender": len(runs),
        "n_run_blender_ok": sum("RUN OK" in outputs.get(cid, "") for cid, _ in runs),
        "n_check_script": sum("check_script" in c for _, c in commands),
        "n_commands": len(commands),
    }


def summarize(run_dir: Path) -> list[dict]:
    rows = []
    for result_path in sorted((run_dir / "tasks").glob("*/result.json")):
        task_dir = result_path.parent
        result = json.loads(result_path.read_text())
        task = json.loads((task_dir / "task.json").read_text())
        traj = json.loads((task_dir / "traj.json").read_text()) if (task_dir / "traj.json").is_file() else {"messages": []}
        rows.append({
            "task_id": result["task_id"],
            "difficulty": task["meta"].get("difficulty", "-"),
            "status": result["status"],
            "exit_status": result["exit_status"],
            "n_steps": result["n_steps"],
            "n_format_errors": result["n_format_errors"],
            "seconds": result["seconds"],
            **tool_usage(traj),
            "reason": failure_reason(result, task_dir),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--json", help="also write rows + aggregates to this file")
    args = p.parse_args()
    rows = summarize(Path(args.run))
    if not rows:
        raise SystemExit(f"no results in {args.run}")

    print(f"{'task':24} {'diff':6} {'status':9} {'steps':>5} {'runs':>5} {'ok':>3} {'fmt':>3}  reason")
    for r in rows:
        print(f"{r['task_id']:24} {r['difficulty']:6} {r['status']:9} {r['n_steps']:5d} {r['n_run_blender']:5d} "
              f"{r['n_run_blender_ok']:3d} {r['n_format_errors']:3d}  {r['reason']}")

    groups = {}
    for r in rows:
        for key in ("all", r["difficulty"]):
            groups.setdefault(key, []).append(r)
    aggregates = {}
    print()
    for key, rs in groups.items():
        n = len(rs)
        acc = sum(r["status"] == "accepted" for r in rs)
        aggregates[key] = {
            "n": n,
            "accepted": acc,
            "accept_rate": round(acc / n, 3),
            "mean_steps": round(sum(r["n_steps"] for r in rs) / n, 1),
            "tasks_with_a_successful_run_blender": sum(r["n_run_blender_ok"] > 0 for r in rs),
            "tasks_with_format_errors": sum(r["n_format_errors"] > 0 for r in rs),
        }
        a = aggregates[key]
        print(f"{key:7} accepted {acc}/{n} ({a['accept_rate']:.0%})  mean steps {a['mean_steps']}  "
              f"format-error tasks {a['tasks_with_format_errors']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
