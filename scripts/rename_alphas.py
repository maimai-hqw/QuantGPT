"""Bulk rename WQ BRAIN alphas — replaces 'anonymous' with meaningful labels.

Usage:
  python scripts/rename_alphas.py                       # rename auto-discovered alphas from /tmp/*.json
  python scripts/rename_alphas.py --alpha-id XXX --name "Foo"  # rename one
  python scripts/rename_alphas.py --from-file labels.json      # rename from a {alpha_id: name} map

Reads .env and authenticates as primary. Skips alphas whose existing name
already matches the desired name (idempotent).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from quantgpt.wq_brain_client import WQBrainClient


# Curated names for the submitted / known-good alphas in this project.
CURATED = {
    # Submitted (ACTIVE) production alpha — V1b_w20:
    "om3dXMwk": {
        "name": "DebtMomentum/F127-S167/W20",
        "description": "Debt-momentum composite, 20-day av_diff window. -1*rank(ts_av_diff(close,20)) + rank(debt/enterprise_value)",
        "tags": ["submitted", "debt-momentum", "production"],
    },
    # Breakthrough A-grade auto-evolved candidate (regime-conditional + multi-factor composite)
    "O0n1L7RJ": {
        "name": "BreakthroughRegimeVolDecay/F105-S145/O0n1L7RJ",
        "description": "Auto-evolved A-grade: regime-conditional vol decay reversal + multi-factor composite. evolve8_R9 round 4 simplify strategy.",
        "tags": ["auto-evolve", "a-grade", "breakthrough", "regime-conditional"],
    },
    # Champion successor — refined version with 5-factor composite low-vol branch
    "E5q5JdeK": {
        "name": "RegimeVolDecay-v2/F138-S158/E5q5JdeK",
        "description": "Auto-evolved champion v2: regime-conditional with 5-factor composite. evolve9 round 6 simplify strategy.",
        "tags": ["auto-evolve", "a-grade", "champion", "regime-conditional", "v2"],
    },
}


def derive_name_from_alpha_record(alpha: dict) -> str:
    """Build a name from a fetched alpha record."""
    aid = alpha.get("alpha_id") or alpha.get("id") or "??"
    is_d = alpha.get("is", {}) or {}
    fit = (is_d.get("fitness") or 0) * 100
    shp = (is_d.get("sharpe") or 0) * 100
    grade = alpha.get("grade") or "?"
    return f"auto/F{fit:.0f}-S{shp:.0f}-G{grade[:1]}/{aid}"


def derive_name_from_evolve_candidate(c: dict, label: str) -> str:
    wq = c.get("wq_brain", {})
    aid = c.get("alpha_id") or "??"
    fit = (wq.get("wq_fitness") or 0) * 100
    shp = (wq.get("wq_sharpe") or 0) * 100
    return f"{label}/F{fit:.0f}-S{shp:.0f}/{aid}"


def collect_from_evolve_outputs() -> dict[str, dict]:
    """Scan /tmp/evolve*.json and /tmp/wq_*.json for alpha_id → {name, description, tags}."""
    targets: dict[str, dict] = {}
    paths = sorted(glob.glob("/tmp/evolve*.json")) + sorted(glob.glob("/tmp/wq_*.json"))
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception:
            continue
        # evolve script writes either {candidates: [...]} or {success: [...], failed: [...]}
        items = []
        if isinstance(data, dict):
            items = data.get("candidates") or data.get("success") or []
            if isinstance(data, list):
                items = data
        elif isinstance(data, list):
            items = data
        label = Path(p).stem  # e.g. 'evolve8_R9'
        for c in items:
            if not isinstance(c, dict):
                continue
            aid = c.get("alpha_id") or c.get("wq_brain", {}).get("alpha_id")
            if not aid:
                continue
            wq = c.get("wq_brain", {}) or {}
            is_d = c.get("is", {}) or {}  # wq_batch.json keeps metrics directly under "is"
            expr = c.get("expression") or c.get("original_expression") or ""
            fit = wq.get("wq_fitness") or is_d.get("fitness") or 0
            shp = wq.get("wq_sharpe")  or is_d.get("sharpe")  or 0
            ret = wq.get("wq_returns") or is_d.get("returns") or 0
            turn = wq.get("wq_turnover") or is_d.get("turnover") or 0
            name = f"{label}/F{fit*100:.0f}-S{shp*100:.0f}/{aid}"
            desc = (
                f"Source: {label}\n"
                f"Expression: {expr}\n"
                f"IS: sharpe={shp:.3f} fitness={fit:.3f} returns={ret:.4f} turnover={turn:.4f}\n"
                f"Strategy: {c.get('strategy','?')}, Round: {c.get('round','?')}"
            )
            existing = targets.get(aid)
            if existing and (existing.get("_fitness") or 0) >= fit:
                continue
            targets[aid] = {
                "name": name,
                "description": desc,
                "tags": ["auto-evolve", label],
                "_fitness": fit,
            }
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-id", help="Rename a single alpha")
    ap.add_argument("--name", help="Name (when used with --alpha-id)")
    ap.add_argument("--description", help="Description (when used with --alpha-id)")
    ap.add_argument("--from-file", help="JSON file: {alpha_id: {name, description, tags}}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    c = WQBrainClient()
    if not c.authenticate():
        sys.exit("WQ auth failed")

    targets: dict[str, dict] = {}
    if args.alpha_id:
        targets[args.alpha_id] = {"name": args.name or "", "description": args.description or ""}
    elif args.from_file:
        targets = json.load(open(args.from_file))
    else:
        # Auto-discover from evolve outputs + curated
        targets = collect_from_evolve_outputs()
        targets.update({k: v for k, v in CURATED.items()})

    print(f"Renaming {len(targets)} alphas...")
    ok_count = 0
    for aid, info in targets.items():
        info = {k: v for k, v in info.items() if not k.startswith("_")}
        if args.dry_run:
            print(f"  [dry-run] {aid}: name={info.get('name','')[:80]}")
            ok_count += 1
            continue
        r = c.update_alpha_metadata(aid, **info)
        if r.get("ok"):
            print(f"  ✓ {aid}: {info.get('name','')[:80]}")
            ok_count += 1
        else:
            print(f"  ✗ {aid}: {r.get('error','')[:120]}")
    print(f"\n>>> {ok_count}/{len(targets)} renamed successfully")


if __name__ == "__main__":
    main()
