"""
Telegram Trading Bot - Cloud Ready
Token from environment variable: BOT_TOKEN
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import yfinance as yf

from ai_trading_system import TradingSystemOrchestrator, TwoEngineConfig
from stock_sector_lookup import detect_sector

# Read token from environment variable
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    print("ERROR: BOT_TOKEN environment variable not set!")
    sys.exit(1)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

config = TwoEngineConfig(
    capital=10_00_000,
    enable_engine1=True,
    enable_engine2=True,
    auto_log_trades=False
)
trading_system = TradingSystemOrchestrator(config)

CAPITAL, STOCK = range(2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome"""
    user_name = update.effective_user.first_name
    
    msg = f"""
🎉 Welcome {user_name}!

💡 NO MORE BROKER FEES!
I'm your FREE AI Trading Advisor!

STEP 1: Enter your trading capital
Example: 50000

STEP 2: Send stock symbol
Example: RELIANCE

💰 Enter your capital amount now:
    """
    
    await update.message.reply_text(msg)
    return CAPITAL


async def receive_capital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get capital"""
    try:
        capital = float(update.message.text.replace(',', '').strip())
        
        if capital < 5000:
            await update.message.reply_text("Minimum capital is Rs 5,000. Please enter at least Rs 5,000:")
            return CAPITAL
        
        context.user_data['capital'] = capital
        
        msg = f"Perfect! Your Capital: Rs {capital:,.0f}\n\nNow send me a stock symbol:\nExamples: RELIANCE, TCS, HDFCBANK"
        
        await update.message.reply_text(msg)
        return STOCK
        
    except ValueError:
        await update.message.reply_text("Please enter numbers only! Example: 50000")
        return CAPITAL


async def receive_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze stock"""
    symbol = update.message.text.strip().upper().replace('.NS', '').replace('.BO', '')
    capital = context.user_data.get('capital', 50000)
    
    if len(symbol) < 2 or len(symbol) > 20:
        await update.message.reply_text("Invalid symbol! Send a valid stock like: RELIANCE")
        return STOCK
    
    processing = await update.message.reply_text(f"Analyzing {symbol}...\nCapital: Rs {capital:,.0f}\nPlease wait 10-15 seconds...")
    
    try:
        sector = detect_sector(symbol)
        if sector == 'UNKNOWN':
            sector = 'OTHER'
        
        logger.info(f"Analyzing {symbol} with capital {capital}")
        
        result = trading_system.analyze_stock(symbol=symbol, sector=sector, index='NIFTY')
        
        if result.get('error'):
            await processing.edit_text(f"Error: {result['error']}\n\nTry another stock:")
            return STOCK
        
        output = format_analysis(result, symbol, capital)
        
        await processing.edit_text(output)
        
        await update.message.reply_text("Analyze another stock?\n\nSend another symbol or /start to reset capital")
        
        return STOCK
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await processing.edit_text("Error occurred! Try another stock:")
        return STOCK


def format_analysis(result: dict, symbol: str, capital: float) -> str:
    """Format analysis"""
    
    # Get real price
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period='1d')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
        else:
            ticker = yf.Ticker(f"{symbol}.BO")
            hist = ticker.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
    except:
        current_price = 0
    
    if current_price == 0:
        return f"Could not fetch price for {symbol}. Try another stock."
    
    signals = result.get('signals', [])
    regime = result.get('regime', 'UNKNOWN')
    
    # Engine 1
    e1_entry = current_price * 1.02
    e1_sl = e1_entry * 0.97
    e1_target = e1_entry * 1.08
    
    # Engine 2
    e2_entry = current_price * 1.05
    e2_sl = e2_entry * 0.95
    e2_t1 = e2_entry * 1.15
    e2_t2 = e2_entry * 1.30
    
    # Use AI signals if available
    for sig_data in signals:
        signal = sig_data.get('signal')
        if not signal:
            continue
        
        engine_type = signal.engine_type
        
        if 'MICRO' in engine_type or 'ENGINE1' in engine_type:
            e1_entry = signal.entry
            e1_sl = signal.stoploss
            e1_target = signal.target_1
        elif 'BIG' in engine_type or 'RUNNER' in engine_type or 'ENGINE2' in engine_type:
            e2_entry = signal.entry
            e2_sl = signal.stoploss
            e2_t1 = signal.target_1
            risk = e2_entry - e2_sl
            e2_t2 = e2_entry + (risk * 3.0)
    
    # Calculate with full capital
    shares1 = int(capital / e1_entry)
    profit1 = capital * ((e1_target - e1_entry) / e1_entry) if shares1 > 0 else 0
    loss1 = capital * ((e1_entry - e1_sl) / e1_entry) if shares1 > 0 else 0
    
    shares2 = int(capital / e2_entry)
    profit_t1 = (capital * 0.5) * ((e2_t1 - e2_entry) / e2_entry) if shares2 > 0 else 0
    profit_t2 = (capital * 0.5) * ((e2_t2 - e2_entry) / e2_entry) if shares2 > 0 else 0
    total_profit2 = profit_t1 + profit_t2
    loss2 = capital * ((e2_entry - e2_sl) / e2_entry) if shares2 > 0 else 0
    
    # Build output
    lines = []
    lines.append("="*35)
    lines.append(f"{symbol} ANALYSIS")
    lines.append("="*35)
    lines.append(f"Capital: Rs {capital:,.0f}")
    lines.append(f"Current Price: Rs {current_price:.2f}")
    lines.append(f"Market: {regime}")
    lines.append("="*35)
    
    if shares1 > 0:
        lines.append("")
        lines.append("ENGINE 1 - Quick Profit (2-5 Days)")
        lines.append(f"Capital Used: Rs {capital:,.0f}")
        lines.append(f"Buy: {shares1} shares @ Rs {e1_entry:.2f}")
        lines.append(f"Stop Loss: Rs {e1_sl:.2f}")
        lines.append(f"Target: Rs {e1_target:.2f}")
        lines.append(f"Expected Profit: Rs {profit1:,.0f}")
        lines.append(f"Max Risk: Rs {loss1:,.0f}")
        lines.append("="*35)
    
    if shares2 > 0:
        lines.append("")
        lines.append("ENGINE 2 - Big Runner (2-4 Weeks)")
        lines.append(f"Capital Used: Rs {capital:,.0f}")
        lines.append(f"Buy: {shares2} shares @ Rs {e2_entry:.2f}")
        lines.append(f"Stop Loss: Rs {e2_sl:.2f}")
        lines.append(f"Target 1: Rs {e2_t1:.2f}")
        lines.append(f"Target 2: Rs {e2_t2:.2f}")
        lines.append(f"Total Profit: Rs {total_profit2:,.0f}")
        lines.append(f"Max Risk: Rs {loss2:,.0f}")
        lines.append("="*35)
    
    lines.append("")
    lines.append("STRATEGY COMPARISON")
    if shares1 > 0:
        lines.append(f"Engine 1 Profit: Rs {profit1:,.0f}")
    if shares2 > 0:
        lines.append(f"Engine 2 Profit: Rs {total_profit2:,.0f}")
    
    return "\n".join(lines)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel"""
    await update.message.reply_text("Bye! Send /start anytime!")
    return ConversationHandler.END


async def run_bot():
    """Run bot"""
    application = Application.builder().token(TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CAPITAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_capital)],
            STOCK: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_stock)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    application.add_handler(conv_handler)
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    print("Bot Started!")
    print("Running 24/7 on cloud")
    
    import asyncio
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


def main():
    import asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
