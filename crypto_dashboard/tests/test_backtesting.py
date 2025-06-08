import pytest
from decimal import Decimal
from crypto_dashboard.backtesting import run_backtest
from crypto_dashboard.strategies import moving_average_crossover_signal

# Mock historical data: list of [timestamp, price]
MOCK_HISTORICAL_DATA_STEADY_RISE = [
    [1609459200000 + i*86400000, Decimal(str(100 + i*2))] for i in range(35) # 35 days, price from 100 to 168
]

MOCK_HISTORICAL_DATA_BUY_OPPORTUNITY = [ # Data crafted for MA Crossover (sw=2, lw=3) to buy
    [1, Decimal('10')], [2, Decimal('5')], [3, Decimal('10')], [4, Decimal('50')]
    # Strategy: sw=2, lw=3. Signal at point 4 (price 50) should be 'buy'
] * 10 # Repeat to have enough data for longer default windows if needed, though strategy params will override

MOCK_HISTORICAL_DATA_SELL_OPPORTUNITY = [ # Data crafted for MA Crossover (sw=2, lw=3) to sell
    [1, Decimal('50')], [2, Decimal('60')], [3, Decimal('70')], [4, Decimal('10')]
    # Strategy: sw=2, lw=3. Signal at point 4 (price 10) should be 'sell'
] * 10


def mock_strategy_always_buy(data_slice, **params):
    # Strategy needs to handle if data_slice is too short for its own logic
    if len(data_slice) < params.get('min_data_points', 1): # Example param for mock
        return 'hold'
    return 'buy'

def mock_strategy_always_sell(data_slice, **params):
    if len(data_slice) < params.get('min_data_points', 1):
        return 'hold'
    return 'sell'

def mock_strategy_always_hold(data_slice, **params):
    return 'hold'

def test_run_backtest_always_buy_strategy():
    initial_capital = Decimal('1000')
    trade_qty = Decimal('1')
    # Use first 5 points of steady rise data. Min_data_points for mock = 1.
    # Prices: 100, 102, 104, 106, 108
    data = MOCK_HISTORICAL_DATA_STEADY_RISE[0:5]

    strategy_config_always_buy = {
        "logic": None,
        "definitions": [
            {"id": "always_buy", "function": mock_strategy_always_buy, "params": {'min_data_points': 1}}
        ]
    }
    results = run_backtest(strategy_config_always_buy, data, initial_capital,
                           trade_quantity=trade_qty)

    assert len(results['trades']) == 5 # Buys at every step
    assert results['trades'][0]['type'] == 'buy'
    assert results['trades'][0]['price'] == Decimal('100')
    assert results['trades'][0]['quantity'] == trade_qty
    assert results['trades'][0]['capital_remaining'] == initial_capital - (Decimal('100') * trade_qty)

    current_assets = sum(t['quantity'] for t in results['trades'] if t['type'] == 'buy')
    assert current_assets == Decimal('5')

    # Final capital = remaining cash + value of assets at last price
    last_price = data[-1][1]
    expected_final_capital = results['trades'][-1]['capital_remaining'] + (current_assets * last_price)
    assert results['final_capital'] == expected_final_capital
    assert results['profit_loss'] == expected_final_capital - initial_capital
    assert len(results['capital_over_time']) == len(data) + 1 # Initial state + each data point

def test_run_backtest_always_sell_strategy_with_initial_asset_purchase():
    initial_capital = Decimal('1000')
    trade_qty_buy = Decimal('2') # Buy 2 units first

    # Simulate an initial purchase before testing "always_sell"
    # For this test, we'll manually construct the state after a first buy
    # then run the backtest. This is a bit complex for unit test.
    # Alternative: a strategy that buys first, then sells.

    # Strategy: Buy on first step, then always sell.
    def buy_then_sell_strategy(data_slice, **params):
        if len(data_slice) == 1: # First data point
            return 'buy'
        if len(data_slice) > 1 and params.get('asset_held', Decimal('0')) > Decimal('0'):
            return 'sell'
        return 'hold'

    # Prices: 100, 102, 104, 106, 108
    data = MOCK_HISTORICAL_DATA_STEADY_RISE[0:5]

    # Need to pass asset_held to strategy if it relies on it.
    # The backtester currently doesn't pass its internal asset_held to strategy_func.
    # So the mock strategy needs to be simple or use passed data.
    # The current run_backtest simulates based on its own asset_held.

    # Let's use a simpler always_sell and check that it doesn't buy, and only sells if assets were somehow acquired.
    # If it starts with 0 assets, always_sell should do nothing.
    strategy_config_always_sell = {
        "logic": None,
        "definitions": [
            {"id": "always_sell", "function": mock_strategy_always_sell, "params": {'min_data_points': 1}}
        ]
    }
    results_no_initial_asset = run_backtest(strategy_config_always_sell, data, initial_capital, trade_quantity=Decimal('1'))
    assert len(results_no_initial_asset['trades']) == 0
    assert results_no_initial_asset['final_capital'] == initial_capital

    # To test selling, let's use a strategy that buys then sells
    # Day 1: Buy. Day 2: Sell. Day 3: Buy. Day 4: Sell etc.
    def buy_sell_alternating_strategy(data_slice, **params):
        # This strategy needs to know the current holding status from the backtester,
        # or make decisions purely on data_slice length or some external state.
        # For simplicity, let's assume it alternates based on day (timestamp).
        current_timestamp = data_slice[-1][0]
        # Assume data[0][0] is 1, data[1][0] is 2 etc for this mock
        # This is not true for MOCK_HISTORICAL_DATA_STEADY_RISE.
        # The simple_ts_data should be defined outside this local strategy function,
        # using the 'data' variable from the outer scope of the test function.
        # current_timestamp = data_slice[-1][0] # This timestamp is from simple_ts_data

        idx = data_slice[-1][0] # Use the actual timestamp from simple_ts_data
        if idx % 2 == 1: # Odd days (1, 3, 5 based on the simple_ts_data) -> Buy
            return 'buy'
        else: # Even days (2, 4) -> Sell
            return 'sell'

    # Let's make a simpler data for this, defined in the test function's scope:
    simple_ts_data_for_test = [[i+1, data[i][1]] for i in range(len(data))]

    strategy_config_buy_sell_alt = {
        "logic": None,
        "definitions": [
            {"id": "buy_sell_alt", "function": buy_sell_alternating_strategy, "params": {}}
        ]
    }
    results_buy_sell = run_backtest(strategy_config_buy_sell_alt, simple_ts_data_for_test, initial_capital, trade_quantity=Decimal('1'))

    assert len(results_buy_sell['trades']) > 0
    assert results_buy_sell['trades'][0]['type'] == 'buy'
    assert results_buy_sell['trades'][0]['price'] == simple_ts_data_for_test[0][1] # Price 100

    if len(results_buy_sell['trades']) > 1:
        assert results_buy_sell['trades'][1]['type'] == 'sell'
        assert results_buy_sell['trades'][1]['price'] == simple_ts_data_for_test[1][1] # Price 102
        # After first buy: capital = 1000 - 100 = 900. asset = 1.
        # After first sell: capital = 900 + 102 = 1002. asset = 0.
        assert results_buy_sell['trades'][1]['capital_remaining'] == Decimal('1002')

    # Final capital should reflect the sequence of buys and sells.
    # Assets held at end should be 1 if odd number of days, 0 if even.
    # The number of days is len(simple_ts_data_for_test). The last day's timestamp is simple_ts_data_for_test[-1][0].
    final_assets = Decimal('1') if simple_ts_data_for_test[-1][0] % 2 == 1 else Decimal('0')
    last_price = simple_ts_data_for_test[-1][1]
    # Calculate final capital manually based on trade log
    final_cash_from_log = results_buy_sell['trades'][-1]['capital_remaining']
    expected_final_val = final_cash_from_log + (final_assets * last_price)
    assert results_buy_sell['final_capital'] == expected_final_val


def test_run_backtest_ma_crossover_strategy():
    initial_capital = Decimal('1000')
    # Use data designed for MA crossover to buy
    # Strategy: sw=2, lw=3. Signal at point 4 (price 50) should be 'buy'
    # Data: [1,10], [2,5], [3,10], [4,50]
    # MA crossover needs lw+1 data points for first possible signal (lw itself for first long MA, +1 for prev long MA)
    # For sw=2, lw=3, it needs 3+1=4 points. Our data has 4 points.
    # Signal is generated AT point historical_data[i] using data up to i.
    # Iteration i=0 (data=[[1,10]]), no signal (strategy returns None)
    # Iteration i=1 (data=[[1,10],[2,5]]), no signal
    # Iteration i=2 (data=[[1,10],[2,5],[3,10]]), no signal (long MA has 1 point, needs 2)
    # Iteration i=3 (data=[[1,10],[2,5],[3,10],[4,50]]), strategy(data[0:4]), signal='buy', price=50
    data = [[1,10],[2,5],[3,10],[4,50]] # Timestamps are 1,2,3,4

    strategy_config_ma_crossover = {
        "logic": None,
        "definitions": [
            {"id": "ma_crossover",
             "function": moving_average_crossover_signal,
             "params": {'short_window': 2, 'long_window': 3}
            }
        ]
    }
    results = run_backtest(strategy_config_ma_crossover, data, initial_capital,
                           trade_quantity=Decimal('1'))

    assert len(results['trades']) == 1
    trade = results['trades'][0]
    assert trade['type'] == 'buy'
    assert trade['price'] == Decimal('50')
    assert trade['timestamp'] == 4
    assert trade['capital_remaining'] == initial_capital - (Decimal('50') * Decimal('1'))

    last_price = data[-1][1]
    assets_held = Decimal('1')
    expected_final_value = trade['capital_remaining'] + (assets_held * last_price)
    assert results['final_capital'] == expected_final_value

def test_run_backtest_no_signals():
    initial_capital = Decimal('1000')
    data = MOCK_HISTORICAL_DATA_STEADY_RISE[0:10] # Prices 100 to 118
    strategy_config_always_hold = {
        "logic": None,
        "definitions": [
            {"id": "always_hold", "function": mock_strategy_always_hold, "params": {}}
        ]
    }
    results = run_backtest(strategy_config_always_hold, data, initial_capital)

    assert len(results['trades']) == 0
    assert results['final_capital'] == initial_capital # No assets bought, so value is initial capital
    assert results['profit_loss'] == Decimal('0')
    assert results['profit_loss_percent'] == Decimal('0')
    assert len(results['capital_over_time']) == len(data) + 1

def test_run_backtest_calculations_profit_loss():
    initial_capital = Decimal('1000')
    # Strategy that buys low, sells high
    def buy_low_sell_high_mock(data_slice, **params):
        price = data_slice[-1][1] # Current price
        if price < Decimal('105'): return 'buy'
        if price > Decimal('115') and params.get('asset_held_for_strat',Decimal(0)) > 0 : return 'sell' # Needs asset_held
        return 'hold'

    # This mock strategy needs asset_held from backtester, which is not passed.
    # So, let's use a simpler data and fixed signals.
    # Data: [1,100], [2,120], [3,110]
    # Trade 1: Buy 1 unit @ 100. Capital = 900. Asset = 1. Value = 900 + 1*100 = 1000.
    # (Portfolio value logged at current price after trade) Value = 900 + 1*100 = 1000.
    # Trade 2: (Hold) @ 120. Capital = 900. Asset = 1. Value = 900 + 1*120 = 1020.
    # Trade 3: (Sell 1 unit) @ 110. Capital = 900+110 = 1010. Asset = 0. Value = 1010.

    test_data = [[1,Decimal('100')], [2,Decimal('120')], [3,Decimal('110')]]

    def custom_strat(data_slice, **params):
        idx = len(data_slice)
        if idx == 1: return 'buy'  # Buy at 100
        if idx == 3: return 'sell' # Sell at 110
        return 'hold'

    strategy_config_custom = {
        "logic": None,
        "definitions": [
            {"id": "custom_strat", "function": custom_strat, "params": {}}
        ]
    }
    results = run_backtest(strategy_config_custom, test_data, initial_capital, trade_quantity=Decimal('1'))

    assert results['final_capital'] == Decimal('1010')
    assert results['profit_loss'] == Decimal('10') # 1010 - 1000
    assert results['profit_loss_percent'] == Decimal('1.00') # (10 / 1000) * 100

    assert len(results['trades']) == 2
    assert results['capital_over_time'][-1][1] == Decimal('1010') # Last portfolio value

def test_run_backtest_zero_initial_capital():
    strategy_config_always_buy_for_zero_cap = {
        "logic": None,
        "definitions": [
            {"id": "always_buy_zero_cap", "function": mock_strategy_always_buy, "params": {'min_data_points': 1}}
        ]
    }
    results = run_backtest(strategy_config_always_buy_for_zero_cap, MOCK_HISTORICAL_DATA_STEADY_RISE[0:5], Decimal('0'))
    assert len(results['trades']) == 0 # Cannot buy
    assert results['final_capital'] == Decimal('0')
    assert results['profit_loss_percent'] == Decimal('0')

def test_run_backtest_trade_quantity_parameter():
    initial_capital = Decimal('1000')
    custom_trade_qty = Decimal('0.5')
    data = MOCK_HISTORICAL_DATA_BUY_OPPORTUNITY[0:4] # Causes 1 buy signal with MA crossover. Data: [1,10],[2,5],[3,10],[4,50]

    strategy_config_ma_custom_qty = {
        "logic": None,
        "definitions": [
            {"id": "ma_crossover_custom_qty",
             "function": moving_average_crossover_signal,
             "params": {'short_window': 2, 'long_window': 3}
            }
        ]
    }
    results = run_backtest(strategy_config_ma_custom_qty, data, initial_capital,
                           trade_quantity=custom_trade_qty)

    assert len(results['trades']) == 1
    assert results['trades'][0]['quantity'] == custom_trade_qty
    cost = data[-1][1] * custom_trade_qty # 50 * 0.5 = 25
    assert results['trades'][0]['capital_remaining'] == initial_capital - cost

def test_run_backtest_empty_historical_data():
    strategy_config_always_buy_empty_data = {
        "logic": None,
        "definitions": [
            {"id": "always_buy_empty_data", "function": mock_strategy_always_buy, "params": {'min_data_points':1}}
        ]
    }
    results = run_backtest(strategy_config_always_buy_empty_data, [], Decimal('1000'))
    assert results['final_capital'] == Decimal('1000')
    assert len(results['trades']) == 0
    assert len(results['capital_over_time']) == 0

from unittest.mock import MagicMock

def test_run_backtest_strategy_params_passing():
    """Test that strategy_params are correctly passed to the strategy function."""
    initial_capital = Decimal('1000')
    data = MOCK_HISTORICAL_DATA_STEADY_RISE[0:5] # Just need some data

    # Create a MagicMock for the strategy function
    mock_strategy_func = MagicMock(return_value='hold')

    test_params = {
        'param1': 10,
        'param2': 'value',
        'rsi_period': 14, # Example of a real param name
        'short_window': 5
    }

    strategy_config_params_passing = {
        "logic": None,
        "definitions": [
            {"id": "params_pass", "function": mock_strategy_func, "params": test_params}
        ]
    }
    run_backtest(
        strategy_config_params_passing,
        data,
        initial_capital,
        trade_quantity=Decimal('1')
    )

    # Check that the mock_strategy_func was called
    assert mock_strategy_func.called

    # Check that it was called with the correct parameters
    if mock_strategy_func.call_count > 0:
        _, last_call_kwargs = mock_strategy_func.call_args_list[-1]
        for key, value in test_params.items():
            assert key in last_call_kwargs
            assert last_call_kwargs[key] == value
    else:
        pytest.fail("Mock strategy function was not called.")

    # Test with empty strategy_params
    mock_strategy_func_empty_params = MagicMock(return_value='hold')
    strategy_config_empty_params = {
        "logic": None,
        "definitions": [
            {"id": "empty_params", "function": mock_strategy_func_empty_params, "params": {}}
        ]
    }
    run_backtest(
        strategy_config_empty_params,
        data,
        initial_capital,
        trade_quantity=Decimal('1')
    )
    if mock_strategy_func_empty_params.call_count > 0:
        _, last_call_kwargs_empty = mock_strategy_func_empty_params.call_args_list[-1]
        assert len(last_call_kwargs_empty) == 0
    else:
        pytest.fail("Mock strategy function (empty params) was not called.")

    # Test with strategy_params=None (should result in empty kwargs if params key is missing or None)
    # The current run_backtest implementation defaults params to {} if not present or None in definition
    mock_strategy_func_none_params = MagicMock(return_value='hold')
    strategy_config_none_params_in_def = { # params explicitly None in definition
        "logic": None,
        "definitions": [
            {"id": "none_params", "function": mock_strategy_func_none_params, "params": None}
        ]
    }
    run_backtest(
        strategy_config_none_params_in_def,
        data,
        initial_capital,
        trade_quantity=Decimal('1')
    )
    if mock_strategy_func_none_params.call_count > 0:
        _, last_call_kwargs_none = mock_strategy_func_none_params.call_args_list[-1]
        assert len(last_call_kwargs_none) == 0 # Expects empty if params:None
    else:
        pytest.fail("Mock strategy function (None params) was not called.")


# --- Tests for Combined Strategies and _get_combined_signal ---
from crypto_dashboard.backtesting import _get_combined_signal

class TestGetCombinedSignal:
    def test_get_combined_signal_and_logic(self):
        assert _get_combined_signal(['buy', 'buy', 'buy'], "AND") == 'buy'
        assert _get_combined_signal(['buy', 'hold', 'buy'], "AND") == 'buy' # 'hold' doesn't veto 'buy' in current AND
        assert _get_combined_signal(['buy', 'sell', 'buy'], "AND") == 'hold' # sell vetoes buy if both present
        assert _get_combined_signal(['sell', 'sell', 'hold'], "AND") == 'sell'
        assert _get_combined_signal(['hold', 'hold', 'hold'], "AND") == 'hold'
        assert _get_combined_signal([None, 'buy', 'buy'], "AND") == 'buy' # None treated as hold
        assert _get_combined_signal([None, None, None], "AND") == 'hold'
        assert _get_combined_signal(['sell', None, 'buy'], "AND") == 'hold'

    def test_get_combined_signal_or_logic(self):
        assert _get_combined_signal(['buy', 'buy', 'buy'], "OR") == 'buy'
        assert _get_combined_signal(['buy', 'hold', 'buy'], "OR") == 'buy'
        assert _get_combined_signal(['buy', 'sell', 'buy'], "OR") == 'hold' # Conflicting buy/sell is hold
        assert _get_combined_signal(['sell', 'sell', 'hold'], "OR") == 'sell'
        assert _get_combined_signal(['hold', 'hold', 'hold'], "OR") == 'hold'
        assert _get_combined_signal([None, 'buy', 'hold'], "OR") == 'buy'
        assert _get_combined_signal([None, 'sell', 'hold'], "OR") == 'sell'
        assert _get_combined_signal([None, None, None], "OR") == 'hold'
        assert _get_combined_signal(['buy', None, 'sell'], "OR") == 'hold'


    def test_get_combined_signal_majority_vote_logic(self):
        assert _get_combined_signal(['buy', 'buy', 'sell'], "MAJORITY_VOTE") == 'buy'
        assert _get_combined_signal(['sell', 'sell', 'buy'], "MAJORITY_VOTE") == 'sell'
        assert _get_combined_signal(['buy', 'hold', 'sell'], "MAJORITY_VOTE") == 'hold' # Tie between buy/sell, hold wins
        assert _get_combined_signal(['buy', 'buy', 'hold'], "MAJORITY_VOTE") == 'buy'
        assert _get_combined_signal(['sell', 'sell', 'hold'], "MAJORITY_VOTE") == 'sell'
        assert _get_combined_signal(['hold', 'hold', 'buy'], "MAJORITY_VOTE") == 'buy' # Hold doesn't count against if one definite signal
        assert _get_combined_signal(['hold', 'hold', 'sell'], "MAJORITY_VOTE") == 'sell'
        assert _get_combined_signal(['hold', 'hold', 'hold'], "MAJORITY_VOTE") == 'hold'
        assert _get_combined_signal([None, 'buy', 'buy'], "MAJORITY_VOTE") == 'buy'
        assert _get_combined_signal([None, 'sell', 'sell'], "MAJORITY_VOTE") == 'sell'
        assert _get_combined_signal([None, 'buy', 'sell'], "MAJORITY_VOTE") == 'hold'
        assert _get_combined_signal([None, None, 'buy'], "MAJORITY_VOTE") == 'buy'
        assert _get_combined_signal([None, None, None], "MAJORITY_VOTE") == 'hold'


class TestRunBacktestCombinedStrategies:
    initial_capital = Decimal('1000')
    trade_qty = Decimal('1')
    # Data: [ts, price]: [1,10], [2,20], [3,15], [4,25], [5,30]
    test_data = [[i+1, Decimal(p)] for i, p in enumerate([10,20,15,25,30])]

    def mock_strat_A(self, data_slice, **params): # Buy on day 2 (price 20), day 4 (price 25)
        ts = data_slice[-1][0]
        if ts == 2 or ts == 4: return 'buy'
        return 'hold'

    def mock_strat_B(self, data_slice, **params): # Buy on day 2 (price 20), Sell on day 4 (price 25)
        ts = data_slice[-1][0]
        if ts == 2: return 'buy'
        if ts == 4: return 'sell'
        return 'hold'

    def mock_strat_C(self, data_slice, **params): # Always hold
        return 'hold'

    def test_single_strategy_via_config(self):
        strategy_config = {
            "logic": None,
            "definitions": [
                {"id": "strat_A", "function": self.mock_strat_A, "params": {}}
            ]
        }
        # Expected: Buy at 20 (day 2), Buy at 25 (day 4)
        results = run_backtest(strategy_config, self.test_data, self.initial_capital, trade_quantity=self.trade_qty)
        assert len(results['trades']) == 2
        assert results['trades'][0]['type'] == 'buy' and results['trades'][0]['timestamp'] == 2
        assert results['trades'][1]['type'] == 'buy' and results['trades'][1]['timestamp'] == 4

    def test_combined_strategies_and_logic(self):
        strategy_config = {
            "logic": "AND",
            "definitions": [
                {"id": "strat_A", "function": self.mock_strat_A, "params": {}}, # Day2:Buy, Day4:Buy
                {"id": "strat_B", "function": self.mock_strat_B, "params": {}}  # Day2:Buy, Day4:Sell
            ]
        }
        # Day 2: A=Buy, B=Buy  => AND = Buy
        # Day 4: A=Buy, B=Sell => AND = Hold (current AND logic for buy+sell conflict)
        results = run_backtest(strategy_config, self.test_data, self.initial_capital, trade_quantity=self.trade_qty)
        assert len(results['trades']) == 1
        assert results['trades'][0]['type'] == 'buy' and results['trades'][0]['timestamp'] == 2

    def test_combined_strategies_or_logic(self):
        strategy_config = {
            "logic": "OR",
            "definitions": [
                {"id": "strat_A", "function": self.mock_strat_A, "params": {}}, # Day2:Buy, Day4:Buy
                {"id": "strat_B", "function": self.mock_strat_B, "params": {}}  # Day2:Buy, Day4:Sell
            ]
        }
        # Day 2: A=Buy, B=Buy  => OR = Buy
        # Day 4: A=Buy, B=Sell => OR = Hold (conflicting buy/sell)
        results = run_backtest(strategy_config, self.test_data, self.initial_capital, trade_quantity=self.trade_qty)
        assert len(results['trades']) == 1
        assert results['trades'][0]['type'] == 'buy' and results['trades'][0]['timestamp'] == 2


    def test_combined_strategies_majority_vote_logic(self):
        def strat_D_buy_day2(data_slice, **params): return 'buy' if data_slice[-1][0] == 2 else 'hold'
        def strat_E_buy_day2(data_slice, **params): return 'buy' if data_slice[-1][0] == 2 else 'hold'
        def strat_F_sell_day2(data_slice, **params): return 'sell' if data_slice[-1][0] == 2 else 'hold'

        strategy_config = {
            "logic": "MAJORITY_VOTE",
            "definitions": [
                {"id": "strat_D", "function": strat_D_buy_day2, "params": {}},
                {"id": "strat_E", "function": strat_E_buy_day2, "params": {}},
                {"id": "strat_F", "function": strat_F_sell_day2, "params": {}}
            ]
        }
        # Day 2: D=Buy, E=Buy, F=Sell => Majority = Buy
        # Other days: Hold, Hold, Hold => Majority = Hold
        results = run_backtest(strategy_config, self.test_data, self.initial_capital, trade_quantity=self.trade_qty)
        assert len(results['trades']) == 1
        assert results['trades'][0]['type'] == 'buy' and results['trades'][0]['timestamp'] == 2

    def test_day_trader_style_with_combined_strategy(self):
        strategy_config = {
            "logic": "OR",
            "definitions": [
                {"id": "strat_A", "function": self.mock_strat_A, "params": {}}, # Day2:Buy, Day4:Buy
                {"id": "strat_C", "function": self.mock_strat_C, "params": {}}  # Always Hold
            ]
        }
        # Combined OR signal: Day 2 = Buy, Day 4 = Buy
        # Day Trader Style:
        # Day 1 (ts=1): price=10, Strat_A=Hold, Strat_C=Hold. OR -> Hold. Action: Hold. Capital=1000, Asset=0.
        # Day 2 (ts=2): price=20, Strat_A=Buy, Strat_C=Hold. OR -> Buy. Action: Buy 1@20. Capital=980, Asset=1.
        #                DayTrader: Holding asset (1), not last day. Override to Sell. Action: Sell 1@20. Capital=980+20=1000. Asset=0.
        # Day 3 (ts=3): price=15, Strat_A=Hold, Strat_C=Hold. OR -> Hold. Action: Hold. Capital=1000, Asset=0.
        # Day 4 (ts=4): price=25, Strat_A=Buy, Strat_C=Hold. OR -> Buy. Action: Buy 1@25. Capital=975, Asset=1.
        #                DayTrader: Holding asset (1), not last day. Override to Sell. Action: Sell 1@25. Capital=975+25=1000. Asset=0.
        # Day 5 (ts=5): price=30, Strat_A=Hold, Strat_C=Hold. OR -> Hold. Action: Hold. Capital=970, Asset=1. (Last day, no DayTrader override. Asset held.)

        results = run_backtest(strategy_config, self.test_data, self.initial_capital,
                               trade_quantity=self.trade_qty, trade_style="day_trader")

        # Expected trades based on current day_trader logic:
        # 1. Day 2 (ts=2): Base signal 'buy'. asset_held=0. Day trader no effect. Action: Buy. Trades: [B@20]
        # 2. Day 3 (ts=3): Base signal 'hold'. asset_held=1. Day trader forces 'sell'. Action: Sell. Trades: [B@20, S@15]
        # 3. Day 4 (ts=4): Base signal 'buy'. asset_held=0. Day trader no effect. Action: Buy. Trades: [B@20, S@15, B@25]
        # 4. Day 5 (ts=5): Base signal 'hold'. asset_held=1. Day trader no override (last day). Action: Hold.

        assert len(results['trades']) == 3
        assert results['trades'][0]['type'] == 'buy' and results['trades'][0]['timestamp'] == 2 # Buy @ 20
        assert results['trades'][0]['price'] == Decimal('20')

        assert results['trades'][1]['type'] == 'sell' and results['trades'][1]['timestamp'] == 3 # Sell @ 15
        assert results['trades'][1]['price'] == Decimal('15')

        assert results['trades'][2]['type'] == 'buy' and results['trades'][2]['timestamp'] == 4 # Buy @ 25
        assert results['trades'][2]['price'] == Decimal('25')

        # Capital after trades: 1000 - 20 (buy) + 15 (sell) - 25 (buy) = 970
        # Assets held: 1 unit (bought at 25)
        # Last price (ts=5): 30
        # Final portfolio value: 970 + (1 * 30) = 1000
        assert results['final_capital'] == self.initial_capital
        assert results['profit_loss'] == Decimal('0')
