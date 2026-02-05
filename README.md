# Telegram Trading Bot - Cloud Deployment

AI stock analysis bot for Telegram. Deploys to Railway, Render, or Heroku.

## Quick Deploy to Railway/Render

1. **Upload this entire folder** to GitHub (new repo)
2. **Connect to Railway** https://railway.app
3. **Add environment variable**: `BOT_TOKEN=your_telegram_token`
4. **Deploy!** Bot runs 24/7

## Get BOT_TOKEN

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the token BotFather gives you
4. Use it as `BOT_TOKEN` environment variable

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set token (Windows)
$env:BOT_TOKEN="your_token_here"

# Run bot
python main.py
```

## What it does

- Enter your trading capital
- Send any Indian stock symbol (e.g., RELIANCE, TCS)
- Get dual-engine AI trading signals
- Engine 1: Quick profits (2-5 days)
- Engine 2: Big runners (2-4 weeks)

Bot runs 24/7 on cloud! 🚀
