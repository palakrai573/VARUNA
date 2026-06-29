"""
Fill DECK.md placeholders with the real evaluation numbers -> outputs/DECK_final.md.
Run after evaluation:  python tools/build_deck.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


def main():
    ev = json.load(open(os.path.join(C.OUTPUTS_DIR, "eval_metrics.json")))
    ai = ev["ai"]
    pers_skill = sum(ai[v]["skill_vs_persistence"][0] for v in C.VARIABLES) / 3 * 100
    repl = {
        "{RAIN_RMSE}": f"{ai['rain']['rmse'][0]:.1f}",
        "{TMAX_RMSE}": f"{ai['tmax']['rmse'][0]:.2f}",
        "{TMAX_ACC}": f"{ai['tmax']['acc'][0]:.2f}",
        "{TMIN_RMSE}": f"{ai['tmin']['rmse'][0]:.2f}",
        "{TMIN_ACC}": f"{ai['tmin']['acc'][0]:.2f}",
        "{SKILL_PERS}": f"{pers_skill:.0f}",
    }
    text = open(os.path.join(C.ROOT, "DECK.md")).read()
    for k, v in repl.items():
        text = text.replace(k, v)
    out = os.path.join(C.OUTPUTS_DIR, "DECK_final.md")
    open(out, "w", encoding="utf-8").write(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
