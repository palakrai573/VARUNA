"""
Full benchmark: ClimateUNet vs persistence / persistence-of-anomaly / climatology
on the held-out TEST years, in real units, with per-lead-day skill.

Produces:
  outputs/eval_metrics.json   machine-readable results
  outputs/skill_curves.png    RMSE & ACC vs lead day
  console table               headline numbers for the deck
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models import baseline, dataset as D  # noqa: E402
from models.forecast import Forecaster  # noqa: E402
from evaluation import metrics as M  # noqa: E402


def run(max_windows=None, stride=1):
    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    splits = D.split_indices(dates)
    test_idx = splits["test"][::stride]
    if max_windows:
        test_idx = test_idx[:max_windows]
    print(f"[eval] evaluating {len(test_idx)} test windows", flush=True)

    fc = Forecaster()
    w = M._wstats(landmask, grid["lat"])
    obs_arr = {v: obs[v].values for v in C.VARIABLES}

    methods = ["ai", "poa", "persistence", "climatology"]
    # accumulators: method -> var -> lead -> list
    acc = {m: {v: {"se": np.zeros(C.HORIZON), "ae": np.zeros(C.HORIZON),
                   "acc": np.zeros(C.HORIZON), "n": np.zeros(C.HORIZON)}
               for v in C.VARIABLES} for m in methods}

    for t in test_idx:
        t = int(t)
        ai = fc.predict(cube, t, carr, std, dates)["frames"]
        poa = baseline.persistence_of_anomaly(obs, carr, dates, t)
        per = baseline.persistence(obs, dates, t)
        cli = baseline.climatology_forecast(carr, dates, t)
        preds = {"ai": ai, "poa": poa, "persistence": per, "climatology": cli}

        for lead in range(C.HORIZON):
            ti = t + lead
            if ti >= len(dates):
                break
            doy = min(int(dates[ti].dayofyear), 365)
            for v in C.VARIABLES:
                truth = obs_arr[v][ti]
                cl = carr[v][doy - 1]
                for m in methods:
                    p = preds[m][v][lead]
                    acc[m][v]["se"][lead] += M.wrmse(p, truth, w) ** 2
                    acc[m][v]["ae"][lead] += M.wmae(p, truth, w)
                    acc[m][v]["acc"][lead] += M.wacc(p, truth, cl, w)
                    acc[m][v]["n"][lead] += 1

    # reduce
    results = {}
    for m in methods:
        results[m] = {}
        for v in C.VARIABLES:
            n = np.maximum(acc[m][v]["n"], 1)
            results[m][v] = {
                "rmse": np.sqrt(acc[m][v]["se"] / n).tolist(),
                "mae": (acc[m][v]["ae"] / n).tolist(),
                "acc": (acc[m][v]["acc"] / n).tolist(),
            }

    # skill vs references
    for v in C.VARIABLES:
        rmse_ai = np.array(results["ai"][v]["rmse"])
        results["ai"][v]["skill_vs_persistence"] = [
            M.skill(rmse_ai[k], results["persistence"][v]["rmse"][k]) for k in range(C.HORIZON)]
        results["ai"][v]["skill_vs_poa"] = [
            M.skill(rmse_ai[k], results["poa"][v]["rmse"][k]) for k in range(C.HORIZON)]

    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(C.OUTPUTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    _plot(results)
    return results


def _print_table(results):
    units = {"rain": "mm", "tmax": "C", "tmin": "C"}
    print("\n================ TEST-SET SKILL (real units) ================")
    for v in C.VARIABLES:
        print(f"\n--- {v} ({units[v]}) ---")
        print(f"{'lead':>4} | {'AI RMSE':>8} {'POA RMSE':>8} {'Pers RMSE':>9} "
              f"{'AI ACC':>7} | {'skill/pers':>10} {'skill/poa':>10}")
        for k in range(C.HORIZON):
            ai = results["ai"][v]
            print(f"{k+1:>4} | {ai['rmse'][k]:8.2f} {results['poa'][v]['rmse'][k]:8.2f} "
                  f"{results['persistence'][v]['rmse'][k]:9.2f} {ai['acc'][k]:7.3f} | "
                  f"{ai['skill_vs_persistence'][k]*100:9.1f}% {ai['skill_vs_poa'][k]*100:9.1f}%")


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads = np.arange(1, C.HORIZON + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, v in enumerate(C.VARIABLES):
        ax = axes[0, j]
        for m, lab in [("ai", "ClimateUNet"), ("poa", "Persist-anom"),
                       ("persistence", "Persistence"), ("climatology", "Climatology")]:
            ax.plot(leads, results[m][v]["rmse"], marker="o", label=lab)
        ax.set_title(f"{v} RMSE"); ax.set_xlabel("lead day"); ax.legend(fontsize=7)
        ax2 = axes[1, j]
        ax2.plot(leads, results["ai"][v]["acc"], marker="o", color="#FF7B00")
        ax2.axhline(0.6, ls="--", color="gray", lw=1)
        ax2.set_title(f"{v} ACC (AI)"); ax2.set_xlabel("lead day"); ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUTS_DIR, "skill_curves.png"), dpi=120)


if __name__ == "__main__":
    run()
