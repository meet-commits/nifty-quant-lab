"""
src/alert_service.py
Purpose: Dispatches automated daily quantitative research alerts via Telegram.
Strictly simulates research tracking - NO order placement.
"""

import os
import requests
from dotenv import load_dotenv

# Load .env variables
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(ENV_PATH)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def send_morning_quant_alert(
    predicted_prob_up: float,
    vix_change_pct: float,
    sp500_change_pct: float,
    crude_change_pct: float
) -> bool:
    """
    Evaluates model conviction and dispatches a formatted alert.
    """
    # Conviction rules
    if predicted_prob_up >= 0.65:
        bias_title = "🟢 STRONG BULLISH (CALL BIAS)"
        model_action = "Simulated Action: ATM Call Paper Tracking"
    elif predicted_prob_up <= 0.35:
        bias_title = "🔴 STRONG BEARISH (PUT BIAS)"
        model_action = "Simulated Action: ATM Put Paper Tracking"
    else:
        bias_title = "⚪ NEUTRAL / NO TRADE (NOISE BAND)"
        model_action = "Simulated Action: Sitting on 100% Cash. Zero trade taken."

    message = (
        f"📊 *NIFTY 50 QUANT LAB - MORNING SCANNER*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Signal State:** {bias_title}\n"
        f"**Model Probability:** `{predicted_prob_up * 100:.1f}%`\n"
        f"**Guidance:** {model_action}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Macro Drivers:**\n"
        f"• S&P 500 Futures: `{sp500_change_pct:+.2f}%`\n"
        f"• India VIX Change: `{vix_change_pct:+.2f}%`\n"
        f"• Crude Oil: `{crude_change_pct:+.2f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ *Disclaimer:* Purely algorithmic research & paper tracking. Not financial advice."
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[ALERT ENGINE - CONSOLE OUTPUT (Telegram credentials not set)]")
        print(message)
        print("💡 Hint: Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env to receive mobile alerts.\n")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=8)
        if response.status_code == 200:
            print("✅ Morning Quant Alert delivered to Telegram successfully!")
            return True
        else:
            print(f"⚠️ Telegram API response: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to reach Telegram API: {e}")
        return False

if __name__ == "__main__":
    send_morning_quant_alert(
        predicted_prob_up=0.71,
        vix_change_pct=-2.4,
        sp500_change_pct=+0.55,
        crude_change_pct=-0.80
    )
