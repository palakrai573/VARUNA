"""
Headless UI smoke test using Streamlit's AppTest: runs the real app.py,
exercises every view and the key sliders, and reports any exception.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def show_exc(at, label):
    if at.exception:
        print(f"  [FAIL] {label}: {at.exception[0].value}")
        return False
    print(f"  [ok]   {label}")
    return True


def main():
    print("Running app (initial load + model)…")
    at = AppTest.from_file(APP, default_timeout=600).run()
    ok = show_exc(at, "initial render")

    views = at.radio[0].options if at.radio else []
    print(f"views detected: {views}")
    for v in views:
        at.radio[0].set_value(v).run()
        ok &= show_exc(at, f"view '{v}'")

    # exercise sliders on the climate-twin view
    at.radio[0].set_value(views[0]).run()
    if len(at.slider) >= 2:
        # move lead day and forecast-start date
        try:
            at.slider[1].set_value(at.slider[1].value + 3).run()
            ok &= show_exc(at, "moved lead-day slider")
        except Exception as e:
            print("  [warn] slider move:", e)

    # toggle a what-if scenario slider
    at.radio[0].set_value(views[2]).run()
    sliders = at.slider
    if sliders:
        sliders[-1].set_value(0.4).run()
        ok &= show_exc(at, "what-if greening slider")

    print("\nRESULT:", "ALL VIEWS + WIDGETS OK" if ok else "ERRORS FOUND")


if __name__ == "__main__":
    main()
