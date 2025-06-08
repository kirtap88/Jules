import yfinance
from decimal import Decimal
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)
# You might want to configure logging further in your app's main setup
# For example: logging.basicConfig(level=logging.INFO)

def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> list | None:
    """
    Fetches daily historical stock data for the given symbol using yfinance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "VOLV-B.ST").
        period (str): The period for which to fetch data (e.g., "1d", "5d", "1mo", "1y", "max").
        interval (str): The interval of data points (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").

    Returns:
        list | None: A list of [timestamp_ms, close_price] data points, or None if an error occurs.
                     Timestamps are epoch milliseconds (UTC).
    """
    try:
        ticker = yfinance.Ticker(symbol)
        # Note: yfinance history index is typically timezone-aware (e.g., America/New_York for US stocks)
        # For consistency with CoinGecko (which uses UTC midnight timestamps for daily data),
        # we should aim for UTC timestamps. yfinance usually provides this for daily data if index is tz-aware.
        # If index is naive, it's often assumed to be UTC for daily data but needs care.
        # For daily data, yfinance often sets the timestamp to market open time in local timezone.
        # To get a consistent daily timestamp (e.g., midnight UTC), we might need to normalize.
        hist = ticker.history(period=period, interval=interval)

        if hist.empty:
            logger.warning(f"No historical data found for symbol {symbol} with period {period} and interval {interval}.")
            return None

        # Ensure the index is a DatetimeIndex and convert to UTC if it's timezone-aware
        if isinstance(hist.index, pd.DatetimeIndex):
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert('UTC')
            else:
                # If timezone naive, yfinance usually means it's based on market's local time.
                # For daily data, this often represents the start of the trading day.
                # To make it consistent like CoinGecko (midnight UTC), we could normalize:
                # hist.index = hist.index.normalize().tz_localize('UTC')
                # However, for simplicity and common use, using the provided timestamp as is (after potential UTC conversion if tz-aware)
                # and converting to milliseconds epoch is often sufficient.
                # Let's assume for now the direct conversion is acceptable.
                # If precise midnight UTC alignment is needed, normalization before conversion is key.
                pass # Keep as is, will convert to int epoch ms directly

        # Convert data to the required format: list of [timestamp_ms, close_price]
        processed_history = []
        for timestamp, row in hist.iterrows():
            # Convert pandas Timestamp to milliseconds epoch
            # Timestamp.value gives nanoseconds, so divide by 1,000,000
            timestamp_ms = timestamp.value // 1_000_000
            close_price = float(row['Close']) # Ensure it's a float
            processed_history.append([timestamp_ms, close_price])

        return processed_history

    except Exception as e:
        logger.error(f"Error fetching stock history for {symbol}: {e}", exc_info=True)
        return None


def get_current_stock_price(symbol: str) -> Decimal | None:
    """
    Fetches the current (or most recent) price for the given stock symbol using yfinance.

    Args:
        symbol (str): The stock symbol.

    Returns:
        Decimal | None: The current price as a Decimal, or None if an error occurs.
    """
    try:
        ticker = yfinance.Ticker(symbol)
        # 'currentPrice' is available for many, but 'regularMarketPrice' or others might be more reliable
        # or fallback to last close from short history.
        info = ticker.info
        price = None
        if 'currentPrice' in info:
            price = info['currentPrice']
        elif 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
        elif 'previousClose' in info: # Fallback if others aren't there
             price = info['previousClose']

        if price is not None:
            return Decimal(str(price))
        else:
            # If no direct info field, try to get the last close price from a short history
            hist = ticker.history(period="5d", interval="1d") # Get a few days to find the last close
            if not hist.empty:
                return Decimal(str(hist['Close'].iloc[-1]))
            else:
                logger.warning(f"Could not determine current price for {symbol} from info or short history.")
                return None

    except Exception as e:
        logger.error(f"Error fetching current stock price for {symbol}: {e}", exc_info=True)
        return None

def search_stocks(query: str) -> list | None:
    """
    (Placeholder) Searches for stocks. yfinance does not have a direct robust search API.
    Returns a hardcoded list of sample international stocks.

    Args:
        query (str): The search query (currently ignored by this placeholder).

    Returns:
        list | None: A list of sample stocks, each a dict with 'symbol', 'name', and 'exchange'.
    """
    logger.info(f"Stock search called with query (ignored by placeholder): {query}")
    # yfinance Ticker object can take international symbols like 'VOLV-B.ST' for Stockholm
    # or 'AIR.PA' for Paris, 'SIE.DE' for XETRA (Siemens)
    sample_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ'},
        {'symbol': 'GOOG', 'name': 'Alphabet Inc. (C Shares)', 'exchange': 'NASDAQ'},
        {'symbol': 'TSLA', 'name': 'Tesla, Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ'},
        {'symbol': 'VOLV-B.ST', 'name': 'Volvo Car AB Series B', 'exchange': 'STO'}, # Stockholm
        {'symbol': 'ERIC-B.ST', 'name': 'Ericsson B', 'exchange': 'STO'},      # Stockholm
        {'symbol': 'AIR.PA', 'name': 'Airbus SE', 'exchange': 'PARIS'},       # Paris
        {'symbol': 'SIE.DE', 'name': 'Siemens AG', 'exchange': 'XETRA'},      # Germany
        {'symbol': 'BHP.AX', 'name': 'BHP Group Limited', 'exchange': 'ASX'}   # Australia
    ]

    # Basic filtering for the placeholder
    if query:
        query_lower = query.lower()
        return [
            s for s in sample_stocks
            if query_lower in s['symbol'].lower() or query_lower in s['name'].lower()
        ]
    return sample_stocks

if __name__ == '__main__':
    # Basic test calls (will make actual internet requests)
    logging.basicConfig(level=logging.INFO)

    print("--- Testing get_stock_history ---")
    history_aapl = get_stock_history("AAPL", period="7d", interval="1d")
    if history_aapl:
        print(f"AAPL history (first 2): {history_aapl[:2]}")
        print(f"AAPL history (last 2): {history_aapl[-2:]}")
    else:
        print("AAPL history not found or error.")

    history_volvb_st = get_stock_history("VOLV-B.ST", period="7d", interval="1d")
    if history_volvb_st:
        print(f"VOLV-B.ST history (first 2): {history_volvb_st[:2]}")
    else:
        print("VOLV-B.ST history not found or error.")

    print("\n--- Testing get_current_stock_price ---")
    price_aapl = get_current_stock_price("AAPL")
    print(f"AAPL current price: {price_aapl}")
    price_volvb_st = get_current_stock_price("VOLV-B.ST")
    print(f"VOLV-B.ST current price: {price_volvb_st}")
    price_invalid = get_current_stock_price("INVALIDSYMBOLXYZ123")
    print(f"INVALIDSYMBOLXYZ123 current price: {price_invalid}")

    print("\n--- Testing search_stocks ---")
    search_all = search_stocks("")
    print(f"Search all (first 3): {search_all[:3]}")
    search_tesla = search_stocks("Tesla")
    print(f"Search 'Tesla': {search_tesla}")
    search_volvo = search_stocks("VOLV")
    print(f"Search 'VOLV': {search_volvo}")
