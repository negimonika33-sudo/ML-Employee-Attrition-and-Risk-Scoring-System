"""Start the Workforce Assistant with one Python command."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "Streamlit is not installed. Run: "
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    app_path = Path(__file__).resolve().with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ]
    return streamlit_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
