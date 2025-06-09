from decimal import Decimal, ROUND_DOWN
import logging

logger = logging.getLogger(__name__)

# Constants for trading styles
SWING_MIN_HOLD_PERIOD = 3  # days
LONG_TERM_IGNORE_SIGNALS_PERIOD = 20 # days


def _get_combined_signal(signals: list, logic: str) -> str:
    """
    Combines multiple trading signals based on a specified logic.

    Args:
        signals (list[str]): A list of individual signals (e.g., ['buy', 'hold', 'sell', None]).
        logic (str): The combination logic: "AND", "OR", "MAJORITY_VOTE".

    Returns:
        str: The combined signal ('buy', 'sell', or 'hold').
    """
    # Process None as 'hold'. Filter out None before count if logic depends on counts vs total.
    processed_signals = [s if s is not None else 'hold' for s in signals]

    if not processed_signals:
        return 'hold'

    buy_count = processed_signals.count('buy')
    sell_count = processed_signals.count('sell')

    if logic == "AND":
        if 'sell' in processed_signals:
            return 'hold' if 'buy' in processed_signals else 'sell' # Sell if any sell, unless conflict (buy also present)
        elif 'buy' in processed_signals: # No 'sell' signals, at least one 'buy'
            return 'buy'
        return 'hold' # All are 'hold'

    elif logic == "OR":
        if 'buy' in processed_signals and 'sell' in processed_signals: # Conflicting signals
            return 'hold'
        if 'buy' in processed_signals:
            return 'buy'
        if 'sell' in processed_signals:
            return 'sell'
        return 'hold'

    elif logic == "MAJORITY_VOTE":
        if buy_count > sell_count:
            return 'buy'
        elif sell_count > buy_count:
            return 'sell'
        return 'hold'

    else:
        logger.warning(f"Unknown or no signal combination logic ('{logic}') provided for multiple signals, defaulting to 'hold'.")
        return 'hold'


def run_backtest(strategy_config, historical_data, initial_capital,
                 trade_quantity=Decimal('1.0'), trade_style: str = "default",
                 stop_loss_percent: Decimal | None = None,
                 take_profit_percent: Decimal | None = None):
    """
    Runs a backtest for a given trading strategy or a combination of strategies,
    incorporating a trading style.

    Args:
        strategy_config (dict): Configuration for the strategy/strategies.
            Example:
            {
                "logic": "AND", # "AND", "OR", "MAJORITY_VOTE", or None for single strategy
                "definitions": [
                    {"id": "ma_cross", "function": ma_crossover_signal, "params": {...}},
                    {"id": "rsi", "function": rsi_signal, "params": {...}}
                ]
            }
        historical_data (list): List of [timestamp, price] data points, sorted by timestamp.
        initial_capital (Decimal): Starting capital for the backtest.
        trade_quantity (Decimal, optional): The fixed quantity of asset to buy. Sells all held asset. Defaults to Decimal('1.0').
        trade_style (str, optional): The trading style to apply ("default", "day_trader", etc.). Defaults to "default".

    Returns:
        dict: A dictionary containing backtesting results.
    """
    if not historical_data:
        return {
            "final_capital": initial_capital, "profit_loss": Decimal('0'),
            "profit_loss_percent": Decimal('0'), "trades": [], "capital_over_time": []
        }

    if not strategy_config or not strategy_config.get("definitions"):
        logger.error("Strategy configuration or definitions missing in run_backtest.")
        return {
            "final_capital": initial_capital, "profit_loss": Decimal('0'),
            "profit_loss_percent": Decimal('0'), "trades": [],
            "capital_over_time": [[historical_data[0][0], initial_capital]] if historical_data else [],
            "error": "Strategy configuration missing."
        }

    capital = Decimal(initial_capital)
    asset_held = Decimal('0')
    trades_log = []
    last_buy_day_index = None # For swing/long-term style holding period tracking
    entry_price_for_sl_tp = None # For current holding's entry price for SL/TP checks

    first_timestamp = historical_data[0][0] if historical_data else 0
    capital_over_time_log = [[first_timestamp, initial_capital]]


    for i in range(len(historical_data)):
        current_timestamp = historical_data[i][0]
        current_price = Decimal(str(historical_data[i][1]))
        data_slice_for_strategy = historical_data[0:i+1]

        individual_signals = []
        for strat_def in strategy_config["definitions"]:
            strategy_func = strat_def["function"]
            strategy_params = strat_def.get("params", {})
            if strategy_params is None: strategy_params = {}

            signal = strategy_func(data_slice_for_strategy, **strategy_params)
            individual_signals.append(signal)

        # Determine base signal from strategy/strategies
        base_final_signal = 'hold'
        if not strategy_config.get("logic") or strategy_config["logic"] is None: # Handles "None" string or Python None
            if len(individual_signals) == 1:
                base_final_signal = individual_signals[0] if individual_signals[0] is not None else 'hold'
            elif len(individual_signals) > 1:
                logger.warning("Multiple strategies defined but no combination logic. Defaulting to 'hold'.")
                base_final_signal = 'hold'
        elif len(individual_signals) == 1 and strategy_config.get("logic"):
            base_final_signal = individual_signals[0] if individual_signals[0] is not None else 'hold'
            logger.info(f"Logic '{strategy_config['logic']}' specified for a single strategy. Using its direct signal: {base_final_signal}.")
        elif len(individual_signals) > 1 :
            base_final_signal = _get_combined_signal(individual_signals, strategy_config["logic"])

        # --- Initialize action signal ---
        current_action_signal = base_final_signal
        sl_tp_triggered_sell = False

        # --- Check Stop-Loss / Take-Profit if a position is open ---
        if entry_price_for_sl_tp is not None and asset_held > Decimal('0'):
            current_potential_profit_percent = ((current_price - entry_price_for_sl_tp) / entry_price_for_sl_tp) * Decimal('100')

            if stop_loss_percent is not None and current_potential_profit_percent <= -abs(stop_loss_percent):
                logger.info(f"Stop-Loss triggered on day {i} at price {current_price}. Profit %: {current_potential_profit_percent:.2f}%")
                current_action_signal = 'sell'
                sl_tp_triggered_sell = True
            elif take_profit_percent is not None and current_potential_profit_percent >= abs(take_profit_percent):
                logger.info(f"Take-Profit triggered on day {i} at price {current_price}. Profit %: {current_potential_profit_percent:.2f}%")
                current_action_signal = 'sell'
                sl_tp_triggered_sell = True

        # --- Apply Trading Style Modifications (if SL/TP didn't trigger a sell) ---
        if not sl_tp_triggered_sell:
            if trade_style == "swing_trader":
                if current_action_signal == 'buy':
                    if last_buy_day_index is not None: # Currently holding (based on style logic)
                        logger.info(f"Swing Trader: Holding asset, ignoring buy signal on day {i}.")
                        current_action_signal = 'hold'
                elif current_action_signal == 'sell':
                    if last_buy_day_index is not None and (i - last_buy_day_index) < SWING_MIN_HOLD_PERIOD:
                        logger.info(f"Swing Trader: In min hold period ({SWING_MIN_HOLD_PERIOD} days), ignoring sell signal on day {i}.")
                        current_action_signal = 'hold'

            elif trade_style == "long_term_investor":
                if last_buy_day_index is not None and (i - last_buy_day_index) < LONG_TERM_IGNORE_SIGNALS_PERIOD:
                    if current_action_signal != 'hold': # Log only if it's overriding something
                        logger.info(f"Long Term Investor: In initial holding period ({LONG_TERM_IGNORE_SIGNALS_PERIOD} days), overriding '{current_action_signal}' signal to 'hold' on day {i}.")
                    current_action_signal = 'hold'

            # "Day Trader" logic is a final override for EOD action if still holding *and* SL/TP didn't already trigger a sell.
            if trade_style == "day_trader" and asset_held > Decimal('0') and i < len(historical_data) - 1:
                if current_action_signal != 'sell':
                    logger.info(f"Day Trader: Overriding signal to 'sell' on day {i} as asset is held EOD.")
                current_action_signal = 'sell'

        # --- Simulate trade based on the final_signal_for_action (now current_action_signal) ---
        if current_action_signal == 'buy' and capital > Decimal('0'):
            cost = trade_quantity * current_price
            if capital >= cost:
                # If already holding, this buy adds to position. SL/TP entry price might need averaging if pyramiding.
                # Current logic: fixed trade_quantity for buy, sell all. So, this buy is effectively a new position if asset_held was 0.
                if asset_held == Decimal('0'): # New position initiated by this buy
                    entry_price_for_sl_tp = current_price
                # If pyramiding was allowed, entry_price_for_sl_tp would need to be an average.
                # For now, if asset_held > 0 and another buy happens (not blocked by swing trader), we keep the original entry_price_for_sl_tp.
                # This means SL/TP is based on the first entry of the current continuous holding.

                asset_held += trade_quantity
                capital -= cost
                trades_log.append({
                    "timestamp": current_timestamp, "type": "buy", "price": current_price,
                    "quantity": trade_quantity, "capital_remaining": capital, "signal_source": "strategy" if not sl_tp_triggered_sell else "risk_mgmt"
                })
                if trade_style in ["swing_trader", "long_term_investor"] and not sl_tp_triggered_sell : # only update if not SL/TP sell
                    if asset_held == trade_quantity: # Assuming this buy started the holding for style tracking
                         last_buy_day_index = i

        elif current_action_signal == 'sell' and asset_held > Decimal('0'):
            proceeds = asset_held * current_price # Sells all currently held quantity
            sold_quantity = asset_held
            capital += proceeds

            source_of_signal = "strategy"
            if sl_tp_triggered_sell:
                source_of_signal = "risk_mgmt"
            elif trade_style == "day_trader" and i < len(historical_data) - 1 : # Check if day trader forced this sell
                 # This check needs to be more precise: was it strategy or day_trader?
                 # If sl_tp_triggered_sell is false, and current_action_signal is 'sell', and day_trader conditions met, then it's day_trader.
                 # The current_action_signal already reflects this.
                 if base_final_signal != 'sell' and (trade_style == "day_trader" and i < len(historical_data) -1): # if base was not sell, but day trader made it sell
                      source_of_signal = "day_trader_style"


            trades_log.append({
                "timestamp": current_timestamp, "type": "sell", "price": current_price,
                "quantity": sold_quantity, "capital_remaining": capital, "signal_source": source_of_signal
            })
            asset_held = Decimal('0')
            entry_price_for_sl_tp = None # Position closed
            if trade_style in ["swing_trader", "long_term_investor"]:
                last_buy_day_index = None # Asset sold for style tracking

        current_portfolio_value = capital + (asset_held * current_price)
        capital_over_time_log.append([current_timestamp, current_portfolio_value])

    final_portfolio_value = capital_over_time_log[-1][1] if capital_over_time_log else initial_capital
    profit_loss = final_portfolio_value - initial_capital
    profit_loss_percent = (profit_loss / initial_capital) * Decimal('100') if initial_capital > Decimal('0') else Decimal('0')

    # --- Advanced Metrics Calculation ---
    completed_round_trips = []
    open_positions_for_metrics = []

    for trade in trades_log: # trades_log contains dicts with 'type', 'price', 'quantity'
        if trade['type'] == 'buy':
            open_positions_for_metrics.append({'price': trade['price'], 'quantity': trade['quantity']})
        elif trade['type'] == 'sell' and open_positions_for_metrics:
            sell_price = trade['price']
            sell_quantity_to_account = trade['quantity'] # Total quantity of this sell trade

            # Match this sell against open buy positions (FIFO)
            while sell_quantity_to_account > Decimal('0') and open_positions_for_metrics:
                oldest_buy = open_positions_for_metrics[0]

                # Determine quantity for this specific round trip leg
                quantity_this_leg = min(sell_quantity_to_account, oldest_buy['quantity'])

                pnl_this_leg = (sell_price - oldest_buy['price']) * quantity_this_leg
                completed_round_trips.append({
                    "entry_price": oldest_buy['price'],
                    "exit_price": sell_price,
                    "quantity": quantity_this_leg,
                    "pnl": pnl_this_leg
                })

                oldest_buy['quantity'] -= quantity_this_leg
                sell_quantity_to_account -= quantity_this_leg

                if oldest_buy['quantity'] <= Decimal('0'):
                    open_positions_for_metrics.pop(0) # This buy position is fully closed

    total_round_trip_trades = len(completed_round_trips)
    winning_trades = sum(1 for rt in completed_round_trips if rt['pnl'] > Decimal('0'))
    losing_trades = sum(1 for rt in completed_round_trips if rt['pnl'] < Decimal('0'))
    breakeven_trades = total_round_trip_trades - winning_trades - losing_trades

    total_profit_from_wins = sum(rt['pnl'] for rt in completed_round_trips if rt['pnl'] > Decimal('0'))
    total_loss_from_losses = sum(abs(rt['pnl']) for rt in completed_round_trips if rt['pnl'] < Decimal('0'))

    win_rate_percent = (Decimal(winning_trades) / Decimal(total_round_trip_trades)) * Decimal('100') if total_round_trip_trades > 0 else Decimal('0')

    average_profit_per_winning_trade = total_profit_from_wins / Decimal(winning_trades) if winning_trades > 0 else Decimal('0')
    average_loss_per_losing_trade = total_loss_from_losses / Decimal(losing_trades) if losing_trades > 0 else Decimal('0')

    profit_factor_val = "N/A"
    if total_loss_from_losses > Decimal('0'):
        profit_factor_val = total_profit_from_wins / total_loss_from_losses
    elif total_profit_from_wins > Decimal('0'): # No losses, but profits exist
        profit_factor_val = Decimal('inf')

    # Max Drawdown Calculation
    max_drawdown_percent = Decimal('0')
    peak_capital_so_far = initial_capital
    if capital_over_time_log: # Ensure log is not empty
        peak_capital_so_far = capital_over_time_log[0][1] # Start with the first logged capital value

    for _, capital_value in capital_over_time_log:
        current_capital_val = Decimal(str(capital_value)) # Ensure it's Decimal
        if current_capital_val > peak_capital_so_far:
            peak_capital_so_far = current_capital_val

        if peak_capital_so_far > Decimal('0'):
            drawdown = (peak_capital_so_far - current_capital_val) / peak_capital_so_far
            if drawdown > max_drawdown_percent:
                max_drawdown_percent = drawdown
    max_drawdown_percent *= Decimal('100')

    results = {
        "final_capital": final_portfolio_value,
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent.quantize(Decimal('0.01')),
        "trades": trades_log,
        "capital_over_time": capital_over_time_log,

        "total_round_trip_trades": total_round_trip_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "win_rate_percent": win_rate_percent.quantize(Decimal('0.01')),
        "average_profit_per_winning_trade": average_profit_per_winning_trade.quantize(Decimal('0.01')),
        "average_loss_per_losing_trade": average_loss_per_losing_trade.quantize(Decimal('0.01')),
        "profit_factor": float(profit_factor_val) if isinstance(profit_factor_val, Decimal) and profit_factor_val != Decimal('inf') else str(profit_factor_val),
        "max_drawdown_percent": max_drawdown_percent.quantize(Decimal('0.01')),
    }
    return results

if __name__ == '__main__':
    # Example Usage (conceptual, requires actual strategy functions to be imported)
    # from strategies import moving_average_crossover_signal, rsi_signal

    # Mock strategy functions for standalone testing
    def mock_ma_crossover(data_slice, short_window=10, long_window=20):
        if len(data_slice) < long_window +1 : return 'hold'
        # import numpy as np # Required if np is used here
        # if data_slice[-1][1] > np.mean([d[1] for d in data_slice[-short_window:]]): return 'buy' # Simplified
        return 'hold'

    def mock_rsi_strat(data_slice, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
        if len(data_slice) < rsi_period+2: return 'hold'
        if len(data_slice) < 4: return 'hold'
        p = [d[1] for d in data_slice[-4:]]
        if p[3] > p[2] > p[1] > p[0] : return 'sell'
        if p[3] < p[2] < p[1] < p[0] : return 'buy'
        return 'hold'

    # import numpy as np # For mock_ma_crossover if it uses np

    sample_data = [[i, 100 + i*0.1 - (i%5)*2 + (i%7)*1.5] for i in range(1, 50)]
    sample_data_decimal = [[ts, Decimal(str(p))] for ts,p in sample_data]

    strategy_config_single_ma = {
        "logic": None,
        "definitions": [
            {"id": "ma_cross", "function": mock_ma_crossover, "params": {"short_window": 5, "long_window": 10}}
        ]
    }
    initial_cap = Decimal("10000.00")

    print("--- Testing Single MA Strategy (Default Style) ---")
    results_single_ma = run_backtest(strategy_config_single_ma, sample_data_decimal, initial_cap)
    print(f"Final Capital: {results_single_ma['final_capital']:.2f}, Trades: {len(results_single_ma['trades'])}")

    print("\n--- Testing Single MA Strategy (Day Trader Style) ---")
    # Example: A buy signal occurs, day trader should sell it by EOD if not last day.
    day_trader_data = [
        [1, Decimal('100')], [2, Decimal('105')], [3, Decimal('110')], # Potential buy on day 2 (price 105)
        [4, Decimal('108')], [5, Decimal('112')]
    ]
    def simple_buy_strat(data_slice, **params):
        if len(data_slice) == 2: return 'buy' # Buy on 2nd day
        return 'hold'

    strategy_config_simple_buy = {
        "logic": None,
        "definitions": [{"id": "simple_buy", "function": simple_buy_strat, "params": {}}]
    }
    results_day_trader = run_backtest(strategy_config_simple_buy, day_trader_data, initial_cap, trade_style="day_trader")
    print(f"Final Capital (Day Trader): {results_day_trader['final_capital']:.2f}, Trades: {len(results_day_trader['trades'])}")
    for t in results_day_trader['trades']: print(t)
    # Expected: Buy on day 2 at 105. Sell on day 2 at 105 (if EOD sell is at same price).
    # Or more realistically, buy on day 2, if day_trader logic makes it sell, it's based on price of day 2.
    # The loop for day i: signal is for action AT price[i]. Day trader logic applies to this signal.
    # If day_trader logic forces a sell, it means the asset bought at price[i] (if signal was buy)
    # would be sold at price[i]. Or if held from previous day, sold at price[i].

    print("\n--- Testing _get_combined_signal ---")
    # ... (rest of __main__ remains the same) ...
