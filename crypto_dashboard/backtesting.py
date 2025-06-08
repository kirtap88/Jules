from decimal import Decimal, ROUND_DOWN
import logging

logger = logging.getLogger(__name__)

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


def run_backtest(strategy_config, historical_data, initial_capital, trade_quantity=Decimal('1.0'), trade_style: str = "default"):
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

        # --- Apply Trade Style Modifications ---
        final_signal_for_action = base_final_signal

        # "day_trader": If holding an asset, force sell at the current period's price if not the last day.
        # This simulates selling at market close of the current day `i`.
        if trade_style == "day_trader" and asset_held > Decimal('0') and i < len(historical_data) - 1:
            if base_final_signal != 'sell':
                logger.info(f"Trade Style 'day_trader': Overriding signal to 'sell' for day {i} (ts {current_timestamp}) as asset is held.")
            final_signal_for_action = 'sell'

        # --- Future trade style logic examples (not fully implemented here) ---
        # if trade_style == "swing_trader":
        #     # Example: Prevent buying more if already holding (anti-pyramiding)
        #     if base_final_signal == 'buy' and asset_held > Decimal('0'):
        #         logger.info(f"Trade Style 'swing_trader': Holding current position, buy signal for {current_timestamp} ignored.")
        #         final_signal_for_action = 'hold'
        #
        # if trade_style == "long_term_investor":
        #     # Example: Ignore sell signals unless it's a very strong one (e.g. from a specific risk strategy not implemented here)
        #     if base_final_signal == 'sell':
        #         logger.info(f"Trade Style 'long_term_investor': Sell signal for {current_timestamp} ignored.")
        #         final_signal_for_action = 'hold'


        # Simulate trade based on the final_signal_for_action
        if final_signal_for_action == 'buy' and capital > Decimal('0'):
            cost = trade_quantity * current_price
            if capital >= cost:
                asset_held += trade_quantity
                capital -= cost
                trades_log.append({
                    "timestamp": current_timestamp, "type": "buy", "price": current_price,
                    "quantity": trade_quantity, "capital_remaining": capital
                })
        elif final_signal_for_action == 'sell' and asset_held > Decimal('0'):
            proceeds = asset_held * current_price
            capital += proceeds
            trades_log.append({
                "timestamp": current_timestamp, "type": "sell", "price": current_price,
                "quantity": asset_held, "capital_remaining": capital
            })
            asset_held = Decimal('0')

        current_portfolio_value = capital + (asset_held * current_price)
        capital_over_time_log.append([current_timestamp, current_portfolio_value])

    final_portfolio_value = capital_over_time_log[-1][1] if capital_over_time_log else initial_capital
    profit_loss = final_portfolio_value - initial_capital
    profit_loss_percent = (profit_loss / initial_capital) * Decimal('100') if initial_capital > Decimal('0') else Decimal('0')

    return {
        "final_capital": final_portfolio_value,
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent.quantize(Decimal('0.01')),
        "trades": trades_log,
        "capital_over_time": capital_over_time_log
    }

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
