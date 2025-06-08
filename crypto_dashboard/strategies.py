import numpy as np

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
