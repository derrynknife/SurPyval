"""The study's table, from the summary JSON files."""

import json
import os

D = os.path.dirname(os.path.abspath(__file__)) + os.sep
inside = total = 0
for tag in [
    "linear",
    "exp",
    "loglink",
    "quad",
    "loglink_gh11",
    "exp_gh11",
    "loglink_n1200",
]:
    s = json.load(open(D + f"summary_{tag}.json"))
    print(
        f"\n{tag}: {s['fitted']}/{s['reps']} fitted, "
        f"{s['verified']} verified, "
        f"fit time total {s['fit_time_total']:.0f} s "
        f"(mean {s['fit_time_mean']:.1f} s), "
        f"wall {s['wall_time_this_run']:.0f} s, {s['date']}"
    )
    print(
        "| parameter | truth | bias (MC se) | coverage (MC se) | sd/se "
        "| flag |"
    )
    print("|---|---|---|---|---|---|")
    for r in s["rows"]:
        flag = []
        if not 0.92 <= r["coverage"] <= 0.98:
            flag.append("coverage")
        if abs(r["bias_z"]) > 2:
            flag.append(f"bias {r['bias_z']:+.1f} MC se")
        if tag in ("linear", "exp", "loglink", "quad"):
            total += 1
            inside += 0.92 <= r["coverage"] <= 0.98
        b = (
            f"{100 * r['bias']:+.2f}% ({100 * r['bias_mcse']:.2f}%)"
            if r["relative"]
            else f"{r['bias']:+.4f} ({r['bias_mcse']:.4f}) abs"
        )
        cov = f"{r['coverage']:.3f} ({r['coverage_mcse']:.3f})"
        print(
            f"| {r['name']} | {r['truth']:.4g} | {b} | {cov} "
            f"| {r['sd_over_se']:.2f} | {', '.join(flag)} |"
        )
print("\ncoverage inside 0.92-0.98:", inside, "of", total)
