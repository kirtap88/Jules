import numpy as np
import pandas as pd

def moving_average_crossover_signal(historical_data, short_window=10, long_window=30):
    """
    Generates a trading signal based on a moving average crossover strategy.

    Args:
        historical_data (list): A list of price points, where each point is [timestamp, price].
                                Prices should be sorted by timestamp in ascending order.
        short_window (int): The period for the short-term moving average. Must be > 0.
        long_window (int): The period for the long-term moving average. Must be > short_window.

    Returns:
        str or None: 'buy' if the short-term SMA crosses above the long-term SMA,
                     'sell' if the short-term SMA crosses below the long-term SMA,
                     'hold' if no crossover is detected or conditions are neutral,
                     None if data is insufficient or parameters are invalid.
    """
    if not historical_data:
        return None # No data provided

    if not (isinstance(short_window, int) and isinstance(long_window, int) and \
            short_window > 0 and long_window > 0):
        #print("Error: Windows must be positive integers.")
        return None

    if short_window >= long_window:
        #print("Error: Long window must be greater than short window.")
        return None # Long window must be greater than short window for meaningful crossover

    # Need at least long_window + 1 data points to have two comparable points for long_sma
    # (long_sma itself will have 2 points, derived from long_window+1 prices)
    if len(historical_data) < long_window + 1:
        #print(f"Error: Insufficient data. Need at least {long_window + 1} points for long_window={long_window}.")
        return None

    prices = np.array([item[1] for item in historical_data], dtype=float)

    # Calculate short-term SMA
    # 'valid' mode means the result is (len(prices) - short_window + 1) long
    short_sma = np.convolve(prices, np.ones(short_window), 'valid') / short_window

    # Calculate long-term SMA
    # 'valid' mode means the result is (len(prices) - long_window + 1) long
    long_sma = np.convolve(prices, np.ones(long_window), 'valid') / long_window

    # For crossover detection, we need at least two points from the long_sma series
    # and correspondingly from the short_sma series.
    if len(long_sma) < 2:
        # This check is theoretically covered by len(historical_data) < long_window + 1,
        # as convolve in 'valid' mode produces len(prices) - N + 1 results.
        # So, for len(long_sma) to be < 2, len(prices) - long_window + 1 < 2
        # => len(prices) < long_window + 1.
        #print("Error: Not enough long_sma points for comparison (should be caught earlier).")
        return None

    # Align SMA arrays for comparison. We are interested in the most recent two points.
    # The long_sma series is shorter. We take the tail of short_sma that aligns with long_sma.
    # Example: prices len 10, short_w=2, long_w=5
    # short_sma len 9: s1,s2,s3,s4,s5,s6,s7,s8,s9
    # long_sma len 6:  l1,l2,l3,l4,l5,l6
    # We need to compare the end of these.
    # short_sma_segment_for_comparison = short_sma[-(len(long_sma)):]
    # short_sma_segment_for_comparison will be: s4,s5,s6,s7,s8,s9 (aligned with l1..l6)

    sma_long_current = long_sma[-1]
    sma_long_previous = long_sma[-2]

    # The corresponding short SMA values.
    # short_sma is longer than long_sma by (long_window - short_window) elements.
    # short_sma[-1] is the most recent short-term average.
    # short_sma[-2] is the previous short-term average.
    # These directly correspond to the time periods of long_sma[-1] and long_sma[-2]
    # because both convolutions are 'valid' and end at the last price point.
    # The *indices* from the original `prices` array that these SMAs correspond to are:
    #   last point of long_sma is average of prices[long_window-1:] ... prices[-1]
    #   last point of short_sma is average of prices[short_window-1:] ... prices[-1]
    # No, this is not quite right. Convolution 'valid' means:
    # short_sma[0] averages prices[0:short_window]
    # short_sma[-1] averages prices[len(prices)-short_window : len(prices)]
    # long_sma[0] averages prices[0:long_window]
    # long_sma[-1] averages prices[len(prices)-long_window : len(prices)]
    # So, short_sma[-1] and long_sma[-1] are SMAs for data ending at the SAME price point (the last one).
    # And short_sma[-2] and long_sma[-2] are SMAs for data ending at the SAME price point (the second to last one).

    sma_short_current = short_sma[-1]
    sma_short_previous = short_sma[-2]


    # Check for crossover
    # Buy signal: Short SMA was below Long SMA and now is above
    if sma_short_previous < sma_long_previous and sma_short_current > sma_long_current:
        return 'buy'
    # Sell signal: Short SMA was above Long SMA and now is below
    elif sma_short_previous > sma_long_previous and sma_short_current < sma_long_current:
        return 'sell'

    return 'hold' # No crossover, or lines are touching/equal, or moving in parallel

def rsi_signal(historical_data, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """
    Generates a trading signal based on the Relative Strength Index (RSI).

    Args:
        historical_data (list): List of [timestamp, price] data points, sorted by timestamp.
        rsi_period (int): The period for RSI calculation.
        rsi_overbought (int): The RSI level considered overbought.
        rsi_oversold (int): The RSI level considered oversold.

    Returns:
        str or None: 'buy' if RSI crosses below oversold,
                     'sell' if RSI crosses above overbought,
                     'hold' otherwise,
                     None if data is insufficient.
    """
    if not historical_data or not isinstance(rsi_period, int) or rsi_period <= 0:
        return None

    prices = np.array([item[1] for item in historical_data], dtype=float)

    # RSI calculation requires at least rsi_period + 1 data points to get the first RSI value.
    # (1 period for initial avg gain/loss, and then comparison for subsequent changes)
    # To detect a crossover, we need at least two RSI values (current and previous).
    # This means we need at least rsi_period + 2 prices.
    if len(prices) < rsi_period + 2:
        return None # Not enough data

    # Calculate price differences
    deltas = np.diff(prices) # Differences between consecutive prices

    # Separate gains and losses
    gains = deltas * (deltas > 0)
    losses = -deltas * (deltas < 0) # Losses are positive values

    # Calculate initial average gain and loss using SMA for the first rsi_period deltas
    # Note: deltas array has len(prices) - 1 elements.
    # We need rsi_period deltas for the first average.
    if len(deltas) < rsi_period: # Should be caught by len(prices) < rsi_period + 1
        return None

    avg_gain = np.mean(gains[:rsi_period])
    avg_loss = np.mean(losses[:rsi_period])

    # Calculate subsequent RSIs using Wilder's smoothing method (EMA-like)
    rsi_values = []

    # First RSI value (after the initial rsi_period deltas)
    if avg_loss == 0: # Avoid division by zero; if all losses are zero, RSI is 100
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

    # Calculate for the rest of the data points
    for i in range(rsi_period, len(deltas)):
        current_gain = gains[i]
        current_loss = losses[i]

        avg_gain = ((avg_gain * (rsi_period - 1)) + current_gain) / rsi_period
        avg_loss = ((avg_loss * (rsi_period - 1)) + current_loss) / rsi_period

        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    if len(rsi_values) < 2: # Need at least two RSI values to detect a crossover
        return 'hold' # Or None, but hold is safer if we have one RSI but no crossover yet

    current_rsi = rsi_values[-1]
    previous_rsi = rsi_values[-2]

    # Buy signal: RSI crosses below oversold threshold
    if previous_rsi > rsi_oversold and current_rsi < rsi_oversold:
        return 'buy'
    # Sell signal: RSI crosses above overbought threshold
    elif previous_rsi < rsi_overbought and current_rsi > rsi_overbought:
        return 'sell'

    return 'hold'

def bollinger_bands_signal(historical_data, period=20, std_devs=2):
    """
    Generates a trading signal based on Bollinger Bands.

    Args:
        historical_data (list): List of [timestamp, price] data points.
        period (int): The period for calculating SMA and Standard Deviation.
        std_devs (int/float): The number of standard deviations for the bands.

    Returns:
        str or None: 'buy' if price crosses below lower band,
                     'sell' if price crosses above upper band,
                     'hold' otherwise,
                     None if data is insufficient.
    """
    if not historical_data or not isinstance(period, int) or period <= 0 or std_devs <= 0:
        return None # Invalid parameters

    prices = np.array([item[1] for item in historical_data], dtype=float)

    # Need at least 'period' prices to calculate SMA and StdDev for the current point.
    # To check for a crossover, we need the current price and the previous price,
    # and their respective positions relative to the bands calculated up to those points.
    # This means we need at least period + 1 prices to compare current price with bands
    # based on data up to previous price, and previous price with bands based on data up to price before previous.
    # Or, simpler: calculate bands for points ending at t-1 and t. Then compare price at t-1 and t.
    # For a signal at current price `prices[-1]`, we need bands calculated from `prices[-(period+1):-1]`.
    # And for previous price `prices[-2]`, we need bands from `prices[-(period+2):-2]`.
    # So, we need at least period + 1 data points to make a decision for the current price.
    # (e.g. if period=20, need 21 prices. prices[0] to prices[20].
    #  current price is prices[20]. previous price is prices[19].
    #  current bands based on prices[1:21] (len 20, indices 1 to 20).
    #  previous bands based on prices[0:20] (len 20, indices 0 to 19).

    if len(prices) < period + 1: # Need period prices for current bands, +1 for previous price
        return None

    # Current values (using the latest 'period' data points for bands)
    # The current price is prices[-1]. The bands should be calculated for the period ending *before* current price.
    # So, for price[-1], bands are from prices[-(period+1):-1]
    # For price[-2], bands are from prices[-(period+2):-2]

    # Current price and its relevant window for band calculation
    current_price = prices[-1]
    window_current = prices[-(period+1):-1] # Data to calculate bands for the period *before* current_price
    if len(window_current) < period: # Not enough data for previous period's bands
        return 'hold' # Or None, but hold if we can't determine prev state

    sma_current = np.mean(window_current)
    std_dev_current = np.std(window_current)
    upper_band_current = sma_current + (std_dev_current * std_devs)
    lower_band_current = sma_current - (std_dev_current * std_devs)

    # Previous price and its relevant window for band calculation
    previous_price = prices[-2]
    window_previous = prices[-(period+2):-2] # Data to calculate bands for period *before* previous_price
    if len(window_previous) < period: # Not enough data for bands of two periods ago
         return 'hold' # Cannot determine crossover without two band states

    sma_previous = np.mean(window_previous)
    std_dev_previous = np.std(window_previous)
    upper_band_previous = sma_previous + (std_dev_previous * std_devs)
    lower_band_previous = sma_previous - (std_dev_previous * std_devs)


    # Buy signal: Current price crosses below lower band from above it
    # Price was above lower band, now it's below
    if previous_price > lower_band_previous and current_price < lower_band_current:
        return 'buy'

    # Sell signal: Current price crosses above upper band from below it
    # Price was below upper band, now it's above
    elif previous_price < upper_band_previous and current_price > upper_band_current:
        return 'sell'

    return 'hold'


def stochastic_oscillator_signal(historical_data, k_period=14, d_period=3, oversold_level=20, overbought_level=80):
    """
    Generates a trading signal based on the Stochastic Oscillator.
    Uses closing prices if high/low are not available.

    Args:
        historical_data (list): List of [timestamp, price] or [timestamp, high, low, close] data points.
                                If HLC are not present, uses close price for all calculations.
        k_period (int): The period for %K calculation (lookback for high/low).
        d_period (int): The period for %D (SMA of %K).
        oversold_level (int): The RSI level considered oversold.
        overbought_level (int): The RSI level considered overbought.

    Returns:
        str or None: 'buy', 'sell', 'hold', or None if data is insufficient.
    """
    if not historical_data:
        return None

    if not (isinstance(k_period, int) and k_period > 0 and
            isinstance(d_period, int) and d_period > 0 and
            isinstance(oversold_level, (int, float)) and 0 <= oversold_level <= 100 and
            isinstance(overbought_level, (int, float)) and 0 <= overbought_level <= 100):
        # logger.warning("Stochastic: Invalid parameters.")
        return None

    if oversold_level >= overbought_level:
        # logger.warning("Stochastic: Oversold level must be less than overbought level.")
        return None

    # Adapt data: use close for H/L if only close is available
    # Assuming historical_data is list of [ts, close] or [ts, open, high, low, close]
    # For simplicity with current data provider, we primarily expect [ts, close]
    # If data point is a list: item[1] is close for [ts, close]
    # If data point is a dict: item['close'], item['high'], item['low']

    closes = []
    highs = []
    lows = []

    if isinstance(historical_data[0], dict):
        closes = pd.Series([item['close'] for item in historical_data], dtype=float)
        highs = pd.Series([item.get('high', item['close']) for item in historical_data], dtype=float) # Fallback to close
        lows = pd.Series([item.get('low', item['close']) for item in historical_data], dtype=float)   # Fallback to close
    elif isinstance(historical_data[0], list) and len(historical_data[0]) == 2: # [ts, close]
        closes = pd.Series([item[1] for item in historical_data], dtype=float)
        highs = closes # Use close for high
        lows = closes  # Use close for low
    elif isinstance(historical_data[0], list) and len(historical_data[0]) >= 4: # [ts, o, h, l, c] or similar
        # Assuming a common structure like [ts, open, high, low, close]
        # This needs to be robust if data format varies. For now, let's assume index 2 is high, 3 is low, 4 is close
        # This part is fragile if data format isn't guaranteed.
        # For this implementation, we'll stick to the "close-only" simplification based on current data sources.
        closes = pd.Series([item[1] for item in historical_data], dtype=float) # Assuming item[1] is close for now
        highs = closes
        lows = closes
    else:
        # logger.warning("Stochastic: Unknown historical_data format.")
        return None


    # Minimum data points needed: k_period for first %K, then d_period for first %D.
    # To check crossover, need two %K and two %D. So, k_period + d_period.
    min_data_len = k_period + d_period
    if len(closes) < min_data_len:
        # logger.info(f"Stochastic: Insufficient data. Have {len(closes)}, need {min_data_len}")
        return None

    # Calculate Lowest Low and Highest High over the k_period
    lowest_low_k = lows.rolling(window=k_period, min_periods=k_period).min()
    highest_high_k = highs.rolling(window=k_period, min_periods=k_period).max()

    # Calculate %K
    # %K = 100 * (Current Close - Lowest Low) / (Highest High - Lowest Low)
    # Handle division by zero if Highest High == Lowest Low
    percent_k = ((closes - lowest_low_k) / (highest_high_k - lowest_low_k).replace(0, np.nan)) * 100
    percent_k = percent_k.fillna(50) # Or some other neutral value if denominator was zero

    # Calculate %D (Signal Line - SMA of %K)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    # Drop NaNs from the start of the series
    percent_k_valid = percent_k.dropna()
    percent_d_valid = percent_d.dropna()

    if len(percent_k_valid) < 2 or len(percent_d_valid) < 2:
        return 'hold'

    # Align series by taking common valid indices
    valid_indices = percent_k_valid.index.intersection(percent_d_valid.index)
    if len(valid_indices) < 2:
        return 'hold'

    k_current = percent_k[valid_indices[-1]]
    k_previous = percent_k[valid_indices[-2]]
    d_current = percent_d[valid_indices[-1]]
    d_previous = percent_d[valid_indices[-2]]

    # Buy signal: %K crosses above %D AND %K (or %D) is below oversold_level
    if k_previous < d_previous and k_current > d_current:
        if k_current < oversold_level or d_current < oversold_level: # Moving out of oversold
             return 'buy'
        # Alternative: if k_previous < oversold_level and k_current > oversold_level (crossing up out of oversold)
        # The prompt's version: "%K crosses above %D AND %K (or %D) is below oversold_level" -- means it can still be in oversold, or just crossed into it.
        # A more common interpretation for buy is %K crossing *up* through oversold_level, or %K crossing *up* through %D *while in oversold territory*.
        # Let's use: %K crosses above %D, and the crossover happened in oversold region or %K is now moving out.
        # For simplicity: K crosses D, and K is currently below oversold (meaning it's just exited or still deep)
        # This is a bit ambiguous. Let's use "K crosses D, AND K is below oversold_level (or D is)".
        # The prompt is: "K crosses above D AND K (or D) is below oversold_level".
        # This means the crossover itself can happen anywhere, as long as one of them is in oversold.
        # A more typical buy: k_previous < d_previous and k_current > d_current and (k_current < oversold_level or d_current < oversold_level)
        # Or, K was oversold and crosses D: k_previous < d_previous and k_current > d_current and k_previous < oversold_level

    # Sell signal: %K crosses below %D AND %K (or %D) is above overbought_level
    elif k_previous > d_previous and k_current < d_current:
        if k_current > overbought_level or d_current > overbought_level: # Moving out of overbought
            return 'sell'

    return 'hold'


if __name__ == '__main__':
    # Example Usage for demonstration and basic testing:
    # This block will not be executed when the function is imported elsewhere.

    print("--- Moving Average Crossover Strategy Examples ---")

    # Example 1: Clear Buy Signal
    # Prices: [10, 12, 11, 10, 15, 25, 30]
    # short_window=2, long_window=4
    # prices_np = np.array([10,12,11,10,15,25,30], dtype=float)
    # short_sma(2): [11, 11.5, 10.5, 12.5, 20, 27.5]
    # long_sma(4):  [10.75, 12, 15.25, 20]
    # sma_long_current = 20
    # sma_long_previous = 15.25
    # sma_short_current = 27.5
    # sma_short_previous = 20
    # Buy Condition: short_prev (20) < long_prev (15.25) -- FALSE. This isn't a buy. Short already crossed.
    # Need short_prev < long_prev AND short_curr > long_curr

    # Let's adjust data for a clear buy for (sw=2, lw=4)
    # Need historical_data length >= lw + 1 = 4 + 1 = 5
    # Data: prices = [10, 8, 7, 6, 20]
    # short_sma(2): [9, 7.5, 6.5, 13]
    # long_sma(4):  [7.75, 10.25]
    # sma_long_curr = 10.25, sma_long_prev = 7.75
    # sma_short_curr = 13,   sma_short_prev = 6.5
    # BUY: short_prev (6.5) < long_prev (7.75) -> TRUE
    #      short_curr (13)  > long_curr (10.25) -> TRUE
    # RESULT: 'buy'
    data_buy = [[1,10],[2,8],[3,7],[4,6],[5,20]] # Timestamps are dummy
    print(f"Data: {[p[1] for p in data_buy]}, SW=2, LW=4. Expected: buy. Actual: {moving_average_crossover_signal(data_buy, 2, 4)}")

    # Example 2: Clear Sell Signal
    # Data: prices = [20, 22, 23, 24, 10]
    # short_sma(2): [21, 22.5, 23.5, 17]
    # long_sma(4):  [22.25, 19.75]
    # sma_long_curr = 19.75, sma_long_prev = 22.25
    # sma_short_curr = 17,    sma_short_prev = 23.5
    # SELL: short_prev (23.5) > long_prev (22.25) -> TRUE
    #       short_curr (17)   < long_curr (19.75) -> TRUE
    # RESULT: 'sell'
    data_sell = [[1,20],[2,22],[3,23],[4,24],[5,10]]
    print(f"Data: {[p[1] for p in data_sell]}, SW=2, LW=4. Expected: sell. Actual: {moving_average_crossover_signal(data_sell, 2, 4)}")

    # Example 3: Hold Signal (short stays above long)
    # Data: prices = [10, 12, 14, 16, 18]
    # short_sma(2): [11, 13, 15, 17]
    # long_sma(4):  [13, 15]
    # sma_long_curr = 15, sma_long_prev = 13
    # sma_short_curr = 17, sma_short_prev = 15
    # Buy:  15 < 13 (F)
    # Sell: 15 > 13 (T) but 17 < 15 (F)
    # RESULT: 'hold'
    data_hold_above = [[1,10],[2,12],[3,14],[4,16],[5,18]]
    print(f"Data: {[p[1] for p in data_hold_above]}, SW=2, LW=4. Expected: hold. Actual: {moving_average_crossover_signal(data_hold_above, 2, 4)}")

    # Example 4: Hold Signal (short stays below long)
    # Data: prices = [18, 16, 14, 12, 10]
    # short_sma(2): [17, 15, 13, 11]
    # long_sma(4):  [15, 13]
    # sma_long_curr = 13, sma_long_prev = 15
    # sma_short_curr = 11, sma_short_prev = 13
    # Buy:  13 < 15 (T) but 11 > 13 (F)
    # Sell: 13 > 15 (F)
    # RESULT: 'hold'
    data_hold_below = [[1,18],[2,16],[3,14],[4,12],[5,10]]
    print(f"Data: {[p[1] for p in data_hold_below]}, SW=2, LW=4. Expected: hold. Actual: {moving_average_crossover_signal(data_hold_below, 2, 4)}")

    # Example 5: Insufficient data
    data_insufficient_short = [[1,10],[2,12],[3,14]] # len 3, needs 5 for LW=4
    print(f"Data: Insufficient (short), SW=2, LW=4. Expected: None. Actual: {moving_average_crossover_signal(data_insufficient_short, 2, 4)}")

    data_insufficient_long_sma_points = [[1,10],[2,12],[3,14],[4,16]] # len 4, long_sma will have 1 point, needs 2
    print(f"Data: Insufficient (long SMA points), SW=2, LW=4. Expected: None. Actual: {moving_average_crossover_signal(data_insufficient_long_sma_points, 2, 4)}")


    # Example 6: Invalid windows
    print(f"Data: Invalid SW=-1, SW=2, LW=4. Expected: None. Actual: {moving_average_crossover_signal(data_buy, -1, 4)}")
    print(f"Data: Invalid SW=4,LW=2, SW=2, LW=4. Expected: None. Actual: {moving_average_crossover_signal(data_buy, 4, 2)}")
    print(f"Data: Invalid SW=2,LW=2, SW=2, LW=4. Expected: None. Actual: {moving_average_crossover_signal(data_buy, 2, 2)}")

    # Example 7: Empty data
    print(f"Data: Empty. Expected: None. Actual: {moving_average_crossover_signal([], 2, 4)}")

    print("--- End of Examples ---")


def macd_signal(historical_data, short_ema_period=12, long_ema_period=26, signal_period=9):
    """
    Generates a trading signal based on the Moving Average Convergence Divergence (MACD).

    Args:
        historical_data (list): List of [timestamp, price] data points.
        short_ema_period (int): Period for the short-term EMA.
        long_ema_period (int): Period for the long-term EMA.
        signal_period (int): Period for the signal line EMA.

    Returns:
        str or None: 'buy', 'sell', 'hold', or None if data is insufficient.
    """
    if not historical_data:
        return None

    if not (isinstance(short_ema_period, int) and short_ema_period > 0 and
            isinstance(long_ema_period, int) and long_ema_period > 0 and
            isinstance(signal_period, int) and signal_period > 0):
        # logger.warning("MACD: EMA periods must be positive integers.")
        return None

    if short_ema_period >= long_ema_period:
        # logger.warning("MACD: Long EMA period must be greater than short EMA period.")
        return None

    prices = pd.Series([item[1] for item in historical_data], dtype=float)

    # Minimum data points needed:
    # To calculate the longest EMA (long_ema_period)
    # Then, to calculate the signal line (EMA of MACD line, signal_period values of MACD needed)
    # MACD line needs at least long_ema_period prices to have its first value.
    # Signal line needs signal_period MACD values.
    # To get a MACD value, you need `long_ema_period` prices.
    # To get `signal_period` MACD values, you need `long_ema_period + signal_period - 1` prices.
    # To detect a crossover (current and previous signal line & MACD line), we need one more point.
    # So, min_data_len = long_ema_period + signal_period.
    min_data_len = long_ema_period + signal_period
    if len(prices) < min_data_len:
        # logger.info(f"MACD: Insufficient data. Have {len(prices)}, need {min_data_len}")
        return None

    # Calculate Short and Long EMAs
    short_ema = prices.ewm(span=short_ema_period, adjust=False, min_periods=short_ema_period).mean()
    long_ema = prices.ewm(span=long_ema_period, adjust=False, min_periods=long_ema_period).mean()

    # Calculate MACD Line
    macd_line = short_ema - long_ema

    # Calculate Signal Line (EMA of MACD Line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()

    # Drop NaN values that result from EMA calculations (especially at the beginning)
    # We need at least two valid (non-NaN) points for both macd_line and signal_line to check crossover
    macd_line_valid = macd_line.dropna()
    signal_line_valid = signal_line.dropna()

    if len(macd_line_valid) < 2 or len(signal_line_valid) < 2:
        # logger.info("MACD: Not enough valid MACD/Signal points for crossover detection after NaN drop.")
        return 'hold' # Or None, but hold is safer if some values exist but not enough for crossover

    # Get the most recent two values for comparison
    # Ensure we are comparing points that are aligned (i.e., both exist for the same time period)
    # This is tricky because dropna() might change lengths differently if NaNs are not aligned.
    # A safer way is to get the last two values from the original series, if they are not NaN.

    # Consider the intersection of valid indices
    valid_indices = macd_line.index.intersection(signal_line.index)
    if len(valid_indices) < 2:
        return 'hold'

    # Get last two values from the aligned series
    macd_current = macd_line[valid_indices[-1]]
    macd_previous = macd_line[valid_indices[-2]]
    signal_current = signal_line[valid_indices[-1]]
    signal_previous = signal_line[valid_indices[-2]]

    # Check for crossover
    # Buy signal: MACD Line crosses above Signal Line
    if macd_previous < signal_previous and macd_current > signal_current:
        return 'buy'
    # Sell signal: MACD Line crosses below Signal Line
    elif macd_previous > signal_previous and macd_current < signal_current:
        return 'sell'

    return 'hold'
