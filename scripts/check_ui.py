"""Headless smoke check for the Streamlit UI.

Runs the app through Streamlit's AppTest harness and prints what rendered.
Useful in CI, and much faster than clicking through a browser.

    python scripts/check_ui.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.disable(logging.CRITICAL)

from streamlit.testing.v1 import AppTest  # noqa: E402

SAMPLE = "vs previous years"


def main() -> int:
    app = AppTest.from_file(str(ROOT / "src" / "ui" / "app.py"), default_timeout=300)
    app.run()

    if app.exception:
        print("FAIL on load:", app.exception)
        return 1
    print(f"loaded OK - {len(app.button)} buttons, {len(app.tabs)} tabs")
    print(f"metrics on load : {[(m.label, m.value) for m in app.metric]}")
    print(f"charts on load  : {len(app.get('plotly_chart'))}")

    matches = [b for b in app.button if SAMPLE in b.label]
    if not matches:
        print(f"FAIL: no sample button matching {SAMPLE!r}")
        print("      available:", [b.label for b in app.button])
        return 1

    # Clicking parks the question in session state; the rerun consumes it.
    matches[0].click().run()
    if app.exception:
        print("FAIL after sample click:", app.exception)
        return 1

    messages = app.session_state.get("messages", [])
    print(f"chat messages   : {len(messages)}")
    if len(messages) < 2:
        print("FAIL: the question did not produce an answer")
        return 1

    reply = messages[-1]["content"]
    print(f"charts after    : {len(app.get('plotly_chart'))}")
    print(f"dataframes      : {len(app.dataframe)}")
    print("\nreply:")
    print("   " + reply[:300].replace("\n", " "))

    if "Generated without" in reply or "set GROQ_API_KEY" in reply:
        print("\nFAIL: the status footnote leaked into the answer text")
        return 1

    # Extra turns are the regression test for StreamlitDuplicateElementId:
    # every chat message renders its own charts, so without unique keys the
    # second answer crashes the page.
    #
    # Note what this cannot check: the transcript is wrapped in a
    # fixed-height scrolling container so st.chat_input stays on screen.
    # AppTest has no layout, so if that container is ever removed these
    # assertions still pass while the input walks off the bottom of the
    # page. Verify placement in a real browser after touching the chat tab.
    for follow_up in ("and what about Pune?", "will it rain there this week?"):
        app.chat_input[0].set_value(follow_up).run()
        if app.exception:
            print(f"\nFAIL on follow-up {follow_up!r}:", app.exception)
            return 1

    messages = app.session_state.get("messages", [])
    print(f"\nafter 3 turns   : {len(messages)} messages, "
          f"{len(app.get('plotly_chart'))} charts, no duplicate-ID crash")
    if len(messages) < 6:
        print(f"FAIL: expected 6 messages after 3 turns, got {len(messages)}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
