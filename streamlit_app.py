"""
streamlit_app.py
Default Streamlit Cloud entrypoint for APEX QUANT 999 & Chartink Pro Suite.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if DASHBOARD_PATH not in sys.path:
    sys.path.insert(0, DASHBOARD_PATH)

dashboard_file = os.path.join(DASHBOARD_PATH, "app.py")
with open(dashboard_file, "r", encoding="utf-8") as f:
    app_code = f.read()

exec(compile(app_code, dashboard_file, "exec"), globals())
