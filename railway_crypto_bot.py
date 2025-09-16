import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
import ccxt
from openai import OpenAI
import telegram
from telegram.ext import Application, CommandHandler
import schedule
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoTradingBot:
    def __init__(self):
        # Environment variables
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Telegram bot
        self.telegram_app = Application.builder().token(self.telegram_token).build()
        
        # Initialize Binance (no API key needed for public endpoints)
        self.exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Trading pairs to monitor
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
            'LTC/USDT', 'AVAX/USDT', 'LINK/USDT', 'ADA/USDT', 'DOGE/USDT'
        ]
        
        # Technical analysis parameters
        self.timeframes = ['30m', '1h', '4h']
        self.rsi_period = 14
        self.ema_periods = [20, 50, 200]
        
        # Signal thresholds
        self.min_confidence = 80
        self.max_signals_per_hour = 3
        self.signal_count = 0
        self.last_signal_time = datetime.now()

    async def fetch_market_data(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            indicators = {}
            
            # Price data
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)[-1]
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macdsignal[-1]
            indicators['macd_histogram'] = macdhist[-1]
            
            # EMAs
            for period in self.ema_periods:
                indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            
            # Volume analysis
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
            indicators['current_volume'] = volume[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
            
            # Price levels
            indicators['current_price'] = close[-1]
            indicators['price_change_24h'] = ((close[-1] - close[-48]) / close[-48]) * 100 if len(close) >= 48 else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect candlestick and chart patterns"""
        if df.empty or len(df) < 20:
            return []
        
        patterns = []
        
        try:
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            # Candlestick patterns
            if talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1] != 0:
                patterns.append("Hammer")
            
            if talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] > 0:
                patterns.append("Bullish Engulfing")
            elif talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] < 0:
                patterns.append("Bearish Engulfing")
            
            if talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)[-1] != 0:
                patterns.append("Doji")
            
            if talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0:
                patterns.append("Morning Star")
            
            if talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0:
                patterns.append("Evening Star")
            
            # Simple trend analysis
            recent_highs = high_prices[-10:]
            recent_lows = low_prices[-10:]
            
            if len(recent_highs) >= 5:
                if all(recent_highs[i] >= recent_highs[i-1] for i in range(1, 5)):
                    patterns.append("Higher Highs")
                elif all(recent_highs[i] <= recent_highs[i-1] for i in range(1, 5)):
                    patterns.append("Lower Highs")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return []

    async def analyze_with_openai(self, symbol: str, indicators: Dict, patterns: List[str]) -> Dict[str, Any]:
        """Use OpenAI to analyze the data and generate signals"""
        try:
            prompt = f"""
            Analyze this crypto trading data for {symbol}:

            Technical Indicators:
            - RSI: {indicators.get('rsi', 'N/A')}
            - MACD: {indicators.get('macd', 'N/A')}
            - MACD Signal: {indicators.get('macd_signal', 'N/A')}
            - Current Price: ${indicators.get('current_price', 'N/A')}
            - 24h Change: {indicators.get('price_change_24h', 'N/A')}%
            - Volume Ratio: {indicators.get('volume_ratio', 'N/A')}x
            - EMA 20: {indicators.get('ema_20', 'N/A')}
            - EMA 50: {indicators.get('ema_50', 'N/A')}

            Detected Patterns: {', '.join(patterns) if patterns else 'None'}

            Based on this data, provide:
            1. Signal: BUY, SELL, or HOLD
            2. Confidence: 0-100%
            3. Reason: Brief explanation
            4. Target: Potential price target if BUY/SELL
            5. Stop Loss: Risk management level

            Respond in JSON format only:
            {{
                "signal": "BUY/SELL/HOLD",
                "confidence": 85,
                "reason": "Strong bullish momentum with volume confirmation",
                "target": 45000,
                "stop_loss": 42000
            }}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return {"signal": "HOLD", "confidence": 0, "reason": "Analysis error"}

    async def send_telegram_alert(self, symbol: str, analysis: Dict, indicators: Dict):
        """Send signal to Telegram"""
        try:
            signal = analysis.get('signal', 'HOLD')
            confidence = analysis.get('confidence', 0)
            
            if signal == 'HOLD' or confidence < self.min_confidence:
                return
            
            # Rate limiting
            current_time = datetime.now()
            if (current_time - self.last_signal_time).total_seconds() < 1200:  # 20 minutes
                if self.signal_count >= self.max_signals_per_hour:
                    return
            else:
                self.signal_count = 0
                self.last_signal_time = current_time
            
            emoji = "🚀" if signal == "BUY" else "🔻"
            
            message = f"""
{emoji} <b>STRONG {signal} SIGNAL</b>

<b>Coin:</b> {symbol}
<b>Price:</b> ${indicators.get('current_price', 0):.4f}
<b>24h Change:</b> {indicators.get('price_change_24h', 0):.2f}%

<b>Analysis:</b>
• {analysis.get('reason', 'No reason provided')}
• Confidence: {confidence}%

<b>Targets:</b>
• Target: ${analysis.get('target', 0):.4f}
• Stop Loss: ${analysis.get('stop_loss', 0):.4f}

<b>Volume:</b> {indicators.get('volume_ratio', 0):.2f}x average
<b>RSI:</b> {indicators.get('rsi', 0):.1f}

<i>Time: {datetime.now().strftime('%H:%M UTC')}</i>
            """
            
            bot = telegram.Bot(token=self.telegram_token)
            await bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            self.signal_count += 1
            logger.info(f"Signal sent for {symbol}: {signal} at {confidence}% confidence")
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    async def scan_all_symbols(self):
        """Scan all symbols for trading opportunities"""
        logger.info("Starting market scan...")
        
        for symbol in self.symbols:
            try:
                # Fetch market data
                df = await self.fetch_market_data(symbol)
                if df.empty:
                    continue
                
                # Calculate indicators
                indicators = self.calculate_technical_indicators(df)
                if not indicators:
                    continue
                
                # Detect patterns
                patterns = self.detect_patterns(df)
                
                # Get AI analysis
                analysis = await self.analyze_with_openai(symbol, indicators, patterns)
                
                # Send alert if strong signal
                await self.send_telegram_alert(symbol, analysis, indicators)
                
                # Small delay to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        logger.info("Market scan completed")

    async def start_telegram_commands(self):
        """Set up Telegram bot commands"""
        async def status_command(update, context):
            message = f"""
🤖 <b>Crypto Bot Status</b>

<b>Monitored Coins:</b> {len(self.symbols)}
<b>Active:</b> ✅ Running
<b>Last Scan:</b> {datetime.now().strftime('%H:%M UTC')}
<b>Signals Sent Today:</b> {self.signal_count}

<b>Coins:</b>
{', '.join([s.replace('/USDT', '') for s in self.symbols])}
            """
            await update.message.reply_text(message, parse_mode='HTML')
        
        self.telegram_app.add_handler(CommandHandler("status", status_command))
        
        # Start the bot
        await self.telegram_app.initialize()
        await self.telegram_app.start()

    def schedule_scans(self):
        """Schedule the scanning every 30 minutes"""
        schedule.every(30).minutes.do(lambda: asyncio.create_task(self.scan_all_symbols()))
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

async def main():
    """Main function"""
    try:
        bot = CryptoTradingBot()
        
        # Start Telegram bot
        await bot.start_telegram_commands()
        
        # Initial scan
        await bot.scan_all_symbols()
        
        # Start scheduled scans
        logger.info("Bot started successfully! Scanning every 30 minutes...")
        bot.schedule_scans()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Railway.app compatible
    port = int(os.environ.get("PORT", 8080))
    
    # Run the bot
    asyncio.run(main())