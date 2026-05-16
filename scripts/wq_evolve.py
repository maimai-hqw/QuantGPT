"""WQ BRAIN-aware evolutionary factor mining.

Wires the existing trajectory + meta-evolution + mutation engines from
quantgpt.iteration to a WQ BRAIN evaluator (instead of local hs300 backtest).

Each candidate is evaluated by wq_brain_client.simulate() and scored on:
  score = 70 * wq_fitness + 30 * (hard_pass_rate)
where hard_pass_rate excludes SELF_CORRELATION (lazy-computed at submit time).

Usage:
  python scripts/wq_evolve.py --rounds 5 --seed-expr "<expression>"

The script:
  1. Loads .env, authenticates WQ BRAIN
  2. Loads knowledge base from docs/research_notes/knowledge/
  3. Reads existing ACTIVE alphas (orthogonality constraint)
  4. For each round: trajectory → meta-strategy → mutation/crossover/explore
     → LLM generates expression → WQ-mode parser validates → WQ simulate → score
  5. Emits sorted candidates as JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Bootstrap path + .env
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from quantgpt.crossover_engine import build_crossover_prompt, extract_top_segments
from quantgpt.iteration import (
    _build_explore_prompt,
    _call_llm,
    _normalize_expression,
    is_duplicate_expression,
)
from quantgpt.llm_service import OPERATORS_DOC
from quantgpt.meta_evolution import EvolutionStrategy, select_strategy
from quantgpt.mutation_engine import MutationEngine
from quantgpt.trajectory_analyzer import analyze_trajectory
from quantgpt.wq_brain_client import WQBrainClient


WQ_PARAMS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
}


# Local parser names → WQ FASTEXPR canonical names.
# Our parser is lenient and accepts local-canonical (no ts_ prefix); WQ requires ts_ form.
# Applied as whole-word regex substitution before sending to WQ.
LOCAL_TO_WQ_OPS = [
    ("decay_linear", "ts_decay_linear"),  # WQ canonical has ts_ prefix
    ("ts_std", "ts_std_dev"),             # WQ name is ts_std_dev
    ("sign_power", "signed_power"),       # WQ uses 'signed_power' with 'ed'
    ("product", "ts_product"),            # WQ has ts_ prefix
]


def to_wq_fastexpr(expr: str) -> str:
    import re as _re
    out = expr
    out = _re.sub(r"\bwhere\b", "if_else", out)
    # WQ uses `cap` not `market_cap` (the latter shows up in some local docs)
    out = _re.sub(r"\bmarket_cap\b", "cap", out)
    for local, wq in LOCAL_TO_WQ_OPS:
        out = _re.sub(rf"(?<![a-zA-Z_])(?<!ts_){_re.escape(local)}\b", wq, out)
    return out

# Already-simulated alphas (orthogonality constraint to avoid SC failure / dups)
# Kept in sync with WQ BRAIN platform listing. Top-of-list = most recent / highest fitness.
EXISTING_ALPHAS = [
    # ACTIVE on platform 2026-05-15/16 (codex gpt-5.5 xhigh discoveries)
    "scale(group_zscore(-0.5*ts_sum(returns,4)-0.3*ts_mean(returns,16)-0.2*ts_av_diff(close,12),subindustry)+0.28*group_zscore(log(sales/enterprise_value),subindustry))",  # 6XRRLpKG ACTIVE F=1.18 SR=2.01
    "scale(0.42*zscore(ts_rank(close,140)-ts_rank(close,45))+0.28*zscore(ts_delta(close,20)/close-ts_delta(close,80)/close)+0.18*zscore(ts_av_diff(close,60)))",  # 3qEz7KRN ACTIVE F=1.02 SR=1.90
    "-1 * zscore(ts_decay_linear(ts_mean(returns, 5), 3))",  # P0nXleYE ACTIVE F=1.14 SR=1.52
    # gn4/push round near-misses (Fit 0.96 LOW_FITNESS fail, but structurally similar - keep in orthog list)
    "group_zscore(-ts_decay_linear(returns,7),subindustry)+0.25*group_zscore(sales/cap,industry)+0.14*group_zscore(-debt/sales,industry)",  # 1YopNxPQ F=0.96
    # New A-grade family discovered 2026-05-13/14: regime-switch + sales/EV value + vwap-close microstructure
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.28,-zscore(ts_mean(returns,22)),0.55*zscore(log(sales/enterprise_value))-0.35*zscore(debt/enterprise_value)+0.65*zscore(ts_mean((vwap-close)/close,22))-0.15*zscore(ts_mean(volume/adv20,80))))",  # E5qb2bd0 F=1.30 SR=1.50 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.3,-zscore(ts_mean(returns,25)),0.5*zscore(log(sales/enterprise_value))-0.3*zscore(debt/enterprise_value)+0.6*zscore(ts_mean((vwap-close)/close,25))-0.2*zscore(rank(volume))))",  # O0nkVl9b F=1.18 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.25,-zscore(ts_mean(returns,25)),0.5*zscore(log(sales/enterprise_value))-0.3*zscore(debt/enterprise_value)+0.6*zscore(ts_mean((vwap-close)/close,25))-0.2*zscore(rank(volume))))",  # omnvpRgJ F=1.16 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.32,-zscore(ts_mean(returns,23)),0.55*zscore(log(sales/enterprise_value))-0.35*zscore(debt/enterprise_value)+0.6*zscore(ts_mean((vwap-close)/close,23))-0.15*zscore(ts_sum(volume/adv20,23))))",  # wp51Qj55 F=1.13 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.3,-zscore(ts_mean(returns,20)),0.5*zscore(log(sales/enterprise_value))-0.3*zscore(debt/enterprise_value)+0.6*zscore(ts_mean((vwap-close)/close,20))-0.2*zscore(rank(volume))))",  # XgkP3n65 F=1.12 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.34,-zscore(ts_mean(returns,22)),0.6*zscore(log(sales/enterprise_value))-0.4*zscore(debt/enterprise_value)+0.55*zscore(ts_decay_linear((vwap-close)/close,32))-0.25*zscore(ts_rank(volume,110))+0.2*zscore(ts_mean(log(high/low),25))))",  # xAeQnb1p F=1.03 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.35,-zscore(ts_sum(returns,20)),0.6*zscore(log(sales/enterprise_value))-0.4*zscore(debt/sales)+0.7*zscore(ts_mean((vwap-close)/close,30))-0.2*zscore(rank(volume))))",  # vR59PodA F=1.02 ✓
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.3,-zscore(ts_decay_linear(returns,25)),0.5*zscore(log(sales/enterprise_value))-0.3*zscore(debt/enterprise_value)+0.6*zscore(ts_decay_linear((vwap-close)/close,25))-0.2*zscore(ts_rank(volume,80))))",  # 3qE3OpJO F=1.01 ✓
    # Reversal family (ACTIVE on platform)
    "rank(-1 * ts_delta(close, 5) / close)",  # j2ndvlnO base, Fit 0.78 SR 1.50
    "scale(rank(-ts_delta(close,5)/close) * rank(ts_decay_linear(volume/adv20,10)))",  # P0nXleYE variant template - ACTIVE on platform
    # Pre-existing platform alphas (Fitness ~1.19, sales/cap value branch)
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.35,-zscore(ts_decay_linear(returns,20)),zscore(log(sales/cap))-0.5*zscore(debt/sales)+0.75*zscore(ts_decay_linear((vwap-close)/close,30))))",  # zq5MvXPd F=1.19
    "scale(if_else(zscore(ts_std_dev(returns,60))>0.35,-zscore(ts_decay_linear(returns,20)),1.0*zscore(log(sales/cap))-0.5*zscore(debt/cap)+0.75*zscore(ts_decay_linear((vwap-close)/close,30))-0.4*zscore(ts_mean((high-low)/close,20))))",  # qMn8QzYA F=1.19
    # Earlier baseline alphas (kept to prevent regression)
    "-1 * rank(ts_av_diff(close, 10)) + rank(debt / enterprise_value)",  # F1
    "-1 * rank(ts_decay_linear(close / vwap, 10))",                      # F2
    "-1 * rank(ts_decay_linear(returns * volume / adv20, 5))",           # F3
    "-1 * rank(ts_av_diff(close, 20)) + rank(debt / enterprise_value)",  # V1b_w20
]

KB_DIR = ROOT / "docs" / "research_notes" / "knowledge"


WQ_SYSTEM_PROMPT = """你是 WorldQuant BRAIN 因子专家。生成的表达式将提交到 WQ BRAIN USA TOP3000 上模拟。

## ✅ 允许的算子（WQ FASTEXPR canonical 名字 — 严格使用）
截面: rank, zscore, scale, group_rank, group_zscore
单元数学: abs, sign, log, sqrt, signed_power, power（注意是 signed_power 不是 sign_power）
时序: ts_mean, ts_std_dev, ts_max, ts_min, ts_sum, ts_shift, ts_delta, ts_rank, ts_argmax, ts_argmin, ts_av_diff, ts_corr, ts_cov, ts_decay_linear, ts_product, ts_kurtosis, ts_skewness, ts_regression, ts_ir, ts_backfill
条件: if_else (不是 where!), trade_when
二元: max, min, ts_min, ts_max

## ❌ 严禁使用（会被 WQ 直接拒绝）
- tanh, sigmoid, exp, ema, sma, wma, rsi, macd, obv, atr, boll_*
- decay_linear（必须写 ts_decay_linear）
- ts_std（必须写 ts_std_dev）
- sign_power（必须写 signed_power）
- product（必须写 ts_product）
- where（必须写 if_else）
- **ts_shift, ts_min, ts_max, ts_ir, ts_skewness, ts_cov（用户 WQ 账号权限拒绝）**
  - 需要 lag 时用 ts_delta(x, N) / x 替代（不要 ts_shift）
  - 需要极值时用 ts_argmax/ts_argmin 或 ts_rank 比较替代

## ✅ 允许的变量（仅这些）
价量: open, high, low, close, volume, vwap, returns
ADV: 仅 adv20（adv5 / adv10 / adv60 等都不存在！）
基本面/市值: cap, market_cap, debt, enterprise_value, sales

## ❌ 严禁使用的变量
roe, debt_ratio, equity_multiplier, pe, pb, ps, np_margin, gp_margin, yoy_ni, yoy_equity, asset_turnover, cfo_to_np, current_ratio 等（全是本地 fallback 字段，WQ 没有）
adv5, adv10, adv60, adv120 等（WQ 只有 adv20）

## 输出格式（强制）
- 只输出 1 行可执行的因子表达式
- 不要 markdown / 代码块 / 反引号 / 引号 / 注释 / 解释
- 不要换行
- 嵌套 ≤ 8 层，长度 ≤ 300 字符
"""


def load_knowledge_base() -> str:
    chunks = []
    if not KB_DIR.exists():
        return ""
    for sub in ("rules", "findings", "failures"):
        d = KB_DIR / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.md")):
            chunks.append(f"### {sub}/{f.stem}\n\n" + f.read_text()[:1500])
    return "\n\n---\n\n".join(chunks)


def validate_wq(expr: str) -> str | None:
    """Validate expression in WQ mode — catches adv5/adv10/local-only operators."""
    try:
        from quantgpt.expression_parser import parse_expression
        parse_expression(expr, mode="wq")
        return None
    except Exception as e:
        return str(e)


def wq_evaluate(client: WQBrainClient, expression: str) -> dict:
    """Run WQ simulate, normalize result to iteration-compatible metrics dict."""
    pre_err = validate_wq(expression)
    if pre_err:
        return {
            "expression": expression, "status": "invalid",
            "error": f"WQ-mode parser rejected: {pre_err}",
            "score": 0, "grade": "F",
            "metrics": {"backtest_summary": {}, "report_metrics": {}},
        }

    wq_expression = to_wq_fastexpr(expression)
    if wq_expression != expression:
        print(f"    [translated] {wq_expression}", flush=True)
    res = client.simulate(wq_expression, **WQ_PARAMS)
    if res.get("ok"):
        res["original_expression"] = expression
        res["wq_expression"] = wq_expression
        # Auto-name the alpha on WQ so it shows up with a meaningful label
        alpha_id = res.get("alpha_id")
        if alpha_id:
            is_d = res.get("is", {})
            fit = (is_d.get("fitness") or 0) * 100
            shp = (is_d.get("sharpe") or 0) * 100
            label = os.environ.get("WQ_EVOLVE_LABEL", "auto")
            name = f"{label}/F{fit:.0f}-S{shp:.0f}/{alpha_id}"
            description = (
                f"Auto-evolution candidate.\n"
                f"Expression: {expression}\n"
                f"WQ Expression: {wq_expression}\n"
                f"IS: sharpe={is_d.get('sharpe')}, fitness={is_d.get('fitness')}, "
                f"returns={is_d.get('returns')}, turnover={is_d.get('turnover')}\n"
                f"Settings: {WQ_PARAMS}"
            )
            try:
                client.update_alpha_metadata(
                    alpha_id, name=name, description=description, tags=["auto-evolve", label],
                )
            except Exception as e:
                print(f"    [rename failed] {e}", flush=True)
    if not res.get("ok"):
        return {
            "expression": expression, "status": "failed",
            "error": res.get("error", "")[:300],
            "score": 0, "grade": "F",
            "metrics": {"backtest_summary": {}, "report_metrics": {}},
        }

    is_d = res.get("is", {})
    sharpe = is_d.get("sharpe", 0) or 0
    fitness = is_d.get("fitness", 0) or 0
    returns = is_d.get("returns", 0) or 0
    turnover = is_d.get("turnover", 0) or 0
    checks = is_d.get("checks", [])

    hard_checks = [c for c in checks if c.get("name") != "SELF_CORRELATION"]
    hard_pass = sum(1 for c in hard_checks if c.get("result") == "PASS")
    hard_total = max(len(hard_checks), 1)
    pass_rate = hard_pass / hard_total

    score = round(min(100, max(0, 70 * fitness + 30 * pass_rate)), 1)
    grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"

    return {
        "expression": expression,
        "status": "success",
        "alpha_id": res.get("alpha_id"),
        "score": score,
        "grade": grade,
        "wq_brain": {
            "wq_sharpe": sharpe, "wq_fitness": fitness,
            "wq_returns": returns, "wq_turnover": turnover,
            "submittable": fitness >= 1.0 and 0.01 <= turnover <= 0.7 and pass_rate == 1.0,
            "wq_rating": grade,
        },
        "checks": [{"name": c.get("name"), "result": c.get("result"),
                    "value": c.get("value"), "limit": c.get("limit")} for c in checks],
        "metrics": {
            "backtest_summary": {
                "long_short_sharpe": sharpe,
                "long_short_annual": returns,
                "turnover": turnover,
                "wq_fitness": fitness,
                "ic_mean": 0, "rank_ic_mean": 0,
                "ic_ir": 0, "ic_win_rate": 0.5,
                "monotonicity_score": pass_rate,
                "spread": 1.0 if sharpe > 0 else -1.0,
            },
            "report_metrics": {"sharpe": sharpe, "cagr": returns},
        },
    }


def build_user_prompt(strategy: EvolutionStrategy, trajectory: list[dict],
                      seed_expr: str, all_seen: list[str], iter_idx: int,
                      direction: str | None, knowledge: str) -> str:
    traj_metrics = analyze_trajectory(trajectory)
    current_score = trajectory[-1]["score"]

    if strategy == EvolutionStrategy.RECOMBINE:
        segments = extract_top_segments(trajectory)
        if len(segments) >= 2:
            _, prompt = build_crossover_prompt(segments, seed_expr, current_score, OPERATORS_DOC)
        else:
            strategy = EvolutionStrategy.EXPLORE

    if strategy == EvolutionStrategy.EXPLORE:
        prompt = _build_explore_prompt(
            trajectory[-1]["expression"], current_score,
            trajectory[-1].get("metrics", {}),
            all_seen, iter_idx, "wq_evolve", direction,
        )

    elif strategy in (EvolutionStrategy.EXPLOIT, EvolutionStrategy.SIMPLIFY):
        base_expr = traj_metrics.best_expression or seed_expr
        base_metrics = next(
            (t.get("metrics", {}) for t in trajectory if t["expression"] == base_expr),
            trajectory[0].get("metrics", {}),
        )
        engine = MutationEngine(base_expr, base_metrics, traj_metrics.best_score)
        _, prompt = engine.build_mutation_prompt(OPERATORS_DOC)

    # Append constraints: knowledge + existing alphas + trajectory dedup
    extra = []
    if knowledge:
        extra.append("\n## Knowledge Base — 必须遵守")
        extra.append(knowledge[:4000])
    extra.append("\n## 已提交 alphas — 必须正交（避免 SELF_CORRELATION ≥ 0.7）")
    for e in EXISTING_ALPHAS:
        extra.append(f"- {e}")
    extra.append("\n## 不要重复（本 trajectory 内已尝试）")
    for e in all_seen[-15:]:
        extra.append(f"- {e}")
    if direction:
        extra.append(f"\n## 用户指定方向\n{direction}")

    return prompt + "\n" + "\n".join(extra)


def evolve(seed_expr: str, n_rounds: int, direction: str | None,
           seed_metrics: dict | None = None) -> tuple[list[dict], list[dict]]:
    print(f"[wq_evolve] seed: {seed_expr}", flush=True)
    print(f"[wq_evolve] rounds: {n_rounds}, direction: {direction or '(none)'}", flush=True)

    knowledge = load_knowledge_base()
    print(f"[wq_evolve] knowledge base: {len(knowledge)} chars", flush=True)

    client = WQBrainClient()
    if not client.authenticate():
        raise RuntimeError("WQ auth failed")

    # Bootstrap trajectory with seed
    if seed_metrics is None:
        print("[wq_evolve] evaluating seed...", flush=True)
        seed_metrics = wq_evaluate(client, seed_expr)
        if seed_metrics["status"] != "success":
            raise RuntimeError(f"seed evaluation failed: {seed_metrics.get('error')}")

    trajectory: list[dict] = [{
        "expression": seed_expr,
        "score": seed_metrics["score"],
        "metrics": seed_metrics["metrics"],
        "strategy": "parent",
    }]
    all_seen: list[str] = [seed_expr] + EXISTING_ALPHAS
    candidates: list[dict] = []

    system_prompt = WQ_SYSTEM_PROMPT

    for i in range(n_rounds):
        traj_metrics = analyze_trajectory(trajectory)
        nesting = sum(1 for c in trajectory[-1]["expression"] if c == "(")
        strategy = select_strategy(traj_metrics, trajectory[-1]["score"], nesting)

        print(f"\n[round {i+1}/{n_rounds}] strategy={strategy.value} "
              f"current_score={trajectory[-1]['score']:.1f} "
              f"best={traj_metrics.best_score:.1f} "
              f"diversity={traj_metrics.exploration_diversity:.2f}", flush=True)

        # Generate candidate expression with retries
        candidate_expr = None
        for retry in range(4):
            user_prompt = build_user_prompt(
                strategy, trajectory, seed_expr, all_seen, i, direction, knowledge,
            )
            temp = 1.2 if strategy == EvolutionStrategy.EXPLORE else 0.9
            try:
                expr = _call_llm(system_prompt, user_prompt, temperature=temp)
            except Exception as e:
                print(f"  LLM error (retry {retry+1}): {e}", flush=True)
                continue
            if not expr:
                continue
            if is_duplicate_expression(expr, all_seen):
                print(f"  duplicate (retry {retry+1}): {expr[:80]}", flush=True)
                continue
            err = validate_wq(expr)
            if err:
                print(f"  WQ-mode invalid (retry {retry+1}): {err[:100]} | {expr[:80]}", flush=True)
                continue
            candidate_expr = expr
            break

        if not candidate_expr:
            print("  ✗ failed to generate valid candidate after 4 retries", flush=True)
            continue

        all_seen.append(candidate_expr)
        print(f"  candidate: {candidate_expr}", flush=True)

        result = wq_evaluate(client, candidate_expr)
        result["strategy"] = strategy.value
        result["round"] = i + 1
        candidates.append(result)

        if result["status"] == "success":
            wq = result["wq_brain"]
            mark = "✓" if wq["submittable"] else "·"
            print(f"  → {mark} score={result['score']:.1f} grade={result['grade']} "
                  f"fitness={wq['wq_fitness']:.2f} sharpe={wq['wq_sharpe']:.2f} "
                  f"ret={wq['wq_returns']*100:.2f}% turn={wq['wq_turnover']*100:.1f}% "
                  f"alpha_id={result.get('alpha_id')}", flush=True)
        else:
            print(f"  → ✗ {result['status']}: {result.get('error','')[:120]}", flush=True)

        trajectory.append({
            "expression": candidate_expr,
            "score": result["score"],
            "metrics": result.get("metrics", {}),
            "strategy": strategy.value,
        })

        time.sleep(3)  # WQ rate limit cushion

    return candidates, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-expr", default="-1 * rank(ts_av_diff(close, 20)) + rank(debt / enterprise_value)",
                    help="Starting expression (default: V1b_w20)")
    ap.add_argument("--rounds", type=int, default=5, help="Number of evolution rounds")
    ap.add_argument("--direction", default=None,
                    help="Optional direction hint, e.g. '加入波动率结构'")
    ap.add_argument("--out", default="/tmp/wq_evolve.json")
    args = ap.parse_args()

    candidates, trajectory = evolve(args.seed_expr, args.rounds, args.direction)

    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

    print("\n\n=== FINAL CANDIDATES (sorted by score) ===")
    print(f"{'Score':>5} {'Grade':>5} {'Fitness':>7} {'Sharpe':>7} {'Returns':>8} {'Turn':>7} {'Strategy':<10} alpha_id    Expression")
    print("-" * 160)
    for r in candidates:
        wq = r.get("wq_brain", {})
        if r.get("status") == "success":
            print(f"{r['score']:>5.1f} {r['grade']:>5} {wq['wq_fitness']:>7.3f} "
                  f"{wq['wq_sharpe']:>7.3f} {wq['wq_returns']*100:>7.2f}% "
                  f"{wq['wq_turnover']*100:>6.2f}% {r.get('strategy','?'):<10} "
                  f"{r.get('alpha_id') or '-':<11} {r['expression']}")
        else:
            print(f"{'  -':>5} {'F':>5} {'-':>7} {'-':>7} {'-':>8} {'-':>7} {r.get('strategy','?'):<10} -           [{r.get('status')}] {r['expression'][:80]}")

    with open(args.out, "w") as f:
        json.dump({
            "candidates": candidates,
            "trajectory": trajectory,
            "params": WQ_PARAMS,
            "seed": args.seed_expr,
            "rounds": args.rounds,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n>>> saved {args.out}")


if __name__ == "__main__":
    main()
