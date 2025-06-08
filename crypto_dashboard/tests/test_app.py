import pytest
import json
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Adjust the import according to your project structure.
# This assumes your Flask app instance is named 'app' in 'crypto_dashboard.app'
# If you use an app factory, you'd import and call that.
from crypto_dashboard.app import app as flask_app
from crypto_dashboard.app import db as _db # if your db instance is named _db

@pytest.fixture(scope='module')
def test_client():
    """Create a test client for the Flask application."""
    flask_app.config['TESTING'] = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:' # Use in-memory DB for tests
    # flask_app.config['SERVER_NAME'] = 'localhost.localdomain' # May be needed for url_for if used implicitly

    with flask_app.test_client() as testing_client:
        with flask_app.app_context():
            # _db.create_all() # Create tables if your tests interact with DB models directly
            pass # For now, API tests mock DB interactions or don't rely on them deeply
        yield testing_client
        # with flask_app.app_context():
            # _db.drop_all() # Clean up DB

# --- Tests for /api/strategies ---
def test_api_list_strategies(test_client):
    """Test the /api/strategies endpoint."""
    response = test_client.get('/api/strategies')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0 # Expecting at least one strategy (moving_average_crossover)

    ma_strategy = next((s for s in data if s['id'] == 'moving_average_crossover'), None)
    assert ma_strategy is not None
    assert ma_strategy['name'] == "Moving Average Crossover"
    assert isinstance(ma_strategy['params'], list)
    assert len(ma_strategy['params']) == 2 # short_window, long_window

# --- Tests for /api/backtest (Updated for Combined Strategies) ---

# Import strategy functions to check if they are correctly mapped from IDs
from crypto_dashboard.strategies import moving_average_crossover_signal, rsi_signal, bollinger_bands_signal

class TestBacktestAPI:

    @patch('crypto_dashboard.app.get_crypto_history')
    @patch('crypto_dashboard.app.get_stock_history')
    @patch('crypto_dashboard.app.run_backtest')
    def test_api_run_backtest_single_strategy_success(self, mock_run_backtest, mock_get_stock_hist, mock_get_crypto_hist, test_client):
        mock_get_crypto_hist.return_value = {"prices": [[1609459200000, 100.0]] * 35}
        mock_run_backtest.return_value = {
            "final_capital": Decimal('10000'), "profit_loss": Decimal('0'),
            "profit_loss_percent": Decimal('0'), "trades": [], "capital_over_time": []
        }
        payload = {
            "asset_id": "bitcoin", "asset_type": "crypto", "initial_capital": "10000", "days": "30",
            "trade_quantity": "1.0", "trade_style": "default",
            "strategies": [{"id": "moving_average_crossover", "params": {"short_window": 10, "long_window": 20}}],
            # "combination_logic": None # Optional for single strategy, or explicit "NONE"
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 200

        args, kwargs = mock_run_backtest.call_args
        assert kwargs['initial_capital'] == Decimal('10000')
        assert kwargs['trade_quantity'] == Decimal('1.0')
        assert kwargs['trade_style'] == "default"

        strategy_config = kwargs['strategy_config']
        assert strategy_config['logic'] is None # For single strategy, logic should be None
        assert len(strategy_config['definitions']) == 1
        defin = strategy_config['definitions'][0]
        assert defin['id'] == "moving_average_crossover"
        assert defin['function'] == moving_average_crossover_signal
        assert defin['params'] == {"short_window": 10, "long_window": 20}

    @patch('crypto_dashboard.app.get_crypto_history')
    @patch('crypto_dashboard.app.run_backtest')
    def test_api_run_backtest_combined_strategies_and_logic_success(self, mock_run_backtest, mock_get_history, test_client):
        mock_get_history.return_value = {"prices": [[1609459200000, 100.0]] * 35}
        mock_run_backtest.return_value = {"final_capital": Decimal('10000')} # Simplified mock

        payload = {
            "asset_id": "bitcoin", "asset_type": "crypto", "initial_capital": "10000", "days": "30",
            "trade_quantity": "0.5", "trade_style": "day_trader",
            "combination_logic": "AND",
            "strategies": [
                {"id": "moving_average_crossover", "params": {"short_window": 5, "long_window": 10}},
                {"id": "rsi", "params": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}}
            ]
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 200

        args, kwargs = mock_run_backtest.call_args
        assert kwargs['initial_capital'] == Decimal('10000')
        assert kwargs['trade_quantity'] == Decimal('0.5')
        assert kwargs['trade_style'] == "day_trader" # Check trade_style passthrough

        strategy_config = kwargs['strategy_config']
        assert strategy_config['logic'] == "AND"
        assert len(strategy_config['definitions']) == 2

        ma_def = next(d for d in strategy_config['definitions'] if d['id'] == "moving_average_crossover")
        rsi_def = next(d for d in strategy_config['definitions'] if d['id'] == "rsi")

        assert ma_def['function'] == moving_average_crossover_signal
        assert ma_def['params'] == {"short_window": 5, "long_window": 10}
        assert rsi_def['function'] == rsi_signal
        assert rsi_def['params'] == {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}

    def test_api_run_backtest_missing_fields(self, test_client):
        payload = {"asset_id": "bitcoin", "strategies": [{"id": "rsi"}]} # Missing capital, days
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Missing required fields" in data['error']
        assert "initial_capital" in data['error'] and "days" in data['error']

    def test_api_run_backtest_invalid_payload_structure(self, test_client):
        # Logic specified, but only 1 strategy
        payload1 = {
            "asset_id": "bitcoin", "initial_capital": "1000", "days": "10",
            "combination_logic": "AND",
            "strategies": [{"id": "rsi", "params": {}}]
        }
        # This is now allowed, logic will be ignored (set to None) by the endpoint if only one strategy.
        # response1 = test_client.post('/api/backtest', json=payload1)
        # assert response1.status_code == 400
        # data1 = json.loads(response1.data)
        # assert "Combination logic 'AND' provided for a single strategy." in data1['error'] # Example error

        # Multiple strategies, missing combination_logic
        payload2 = {
            "asset_id": "bitcoin", "initial_capital": "1000", "days": "10",
            "strategies": [{"id": "rsi", "params": {}}, {"id": "moving_average_crossover", "params": {}}]
        }
        response2 = test_client.post('/api/backtest', json=payload2)
        assert response2.status_code == 400
        data2 = json.loads(response2.data)
        assert "Multiple strategies provided but no combination_logic specified" in data2['error']

        # Invalid combination_logic string
        payload3 = {
            "asset_id": "bitcoin", "initial_capital": "1000", "days": "10",
            "combination_logic": "XOR",
            "strategies": [{"id": "rsi", "params": {}}, {"id": "moving_average_crossover", "params": {}}]
        }
        response3 = test_client.post('/api/backtest', json=payload3)
        assert response3.status_code == 400
        data3 = json.loads(response3.data)
        assert "Invalid combination_logic: XOR" in data3['error']

    @patch('crypto_dashboard.app.get_crypto_history')
    def test_api_run_backtest_unknown_strategy_id_in_combo(self, mock_get_history, test_client):
        mock_get_history.return_value = {"prices": [[1609459200000, 100.0]] * 35}
        payload = {
            "asset_id": "bitcoin", "initial_capital": "1000", "days": "10", "trade_style": "default",
            "combination_logic": "AND",
            "strategies": [
                {"id": "rsi", "params": {}},
                {"id": "unknown_strategy_id_blah", "params": {}}
            ]
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "invalid/missing id: 'unknown_strategy_id_blah'" in data['error']

    @patch('crypto_dashboard.app.get_crypto_history')
    def test_api_run_backtest_invalid_params_in_combo(self, mock_get_history, test_client):
        mock_get_history.return_value = {"prices": [[1609459200000, 100.0]] * 35}
        payload = {
            "asset_id": "bitcoin", "initial_capital": "1000", "days": "10", "trade_style": "default",
            "combination_logic": "AND",
            "strategies": [
                {"id": "rsi", "params": {"rsi_period": 0}}, # Invalid rsi_period
                {"id": "moving_average_crossover", "params": {"short_window":10, "long_window":20}}
            ]
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "RSI period must be positive" in data['error']

    @patch('crypto_dashboard.app.get_crypto_history')
    @patch('crypto_dashboard.app.run_backtest')
    def test_api_run_backtest_execution_error_combined(self, mock_run_backtest, mock_get_history, test_client):
        mock_get_history.return_value = {"prices": [[1609459200000, 100.0]] * 35}
        mock_run_backtest.side_effect = Exception("Simulated backtest error in combined")
        payload = {
            "asset_id": "bitcoin", "initial_capital": "10000", "days": "30", "trade_style": "default",
            "strategies": [{"id": "moving_average_crossover", "params": {"short_window": 10, "long_window": 20}}]
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "An error occurred during backtest execution" in data['error']
        assert "Simulated backtest error in combined" in data['error']

    # Old tests for invalid capital, days, history fetch failure can be reused if payload is adapted
    # For example, test_api_run_backtest_invalid_initial_capital needs a "strategies" field in payload.
    def test_api_run_backtest_invalid_initial_capital_adapted(self, test_client):
        payload = {
            "asset_id": "bitcoin", "initial_capital": "zero", "days": "30",
            "strategies": [{"id": "rsi", "params": {}}]
        }
        response = test_client.post('/api/backtest', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Invalid input for capital, days, or quantity" in data['error'] # Error message changed slightly

# --- Tests for /api/optimize_strategy ---
from crypto_dashboard.genetic_optimizer import run_ga_optimization as mockable_run_ga_optimization
# To allow patching if it's imported as 'from .genetic_optimizer import run_ga_optimization' in app.py

class TestOptimizeStrategyAPI:

    @patch('crypto_dashboard.app.run_ga_optimization') # Patch where it's used in app.py
    @patch('crypto_dashboard.app.get_crypto_history') # Mock history for crypto asset type
    @patch('crypto_dashboard.app.get_stock_history') # Mock history for stock asset type
    def test_api_optimize_strategy_success_crypto(self, mock_get_stock_hist, mock_get_crypto_hist, mock_ga_run, test_client):
        mock_ga_run.return_value = {
            "best_config": {"logic": "AND", "definitions": [{"id":"rsi", "params":{"rsi_period":10}}]},
            "best_fitness_profit_percent": Decimal("15.5"),
            "best_config_backtest_results": {"final_capital": Decimal("11550")}
        }
        # Note: get_crypto_history is passed as a function to run_ga_optimization,
        # so it's not directly called by the API endpoint handler before calling run_ga_optimization.
        # However, run_ga_optimization itself will call it.
        # For this test, it's more about asserting run_ga_optimization was called correctly.

        payload = {
            "asset_id": "ethereum", "asset_type": "crypto", "initial_capital": "10000",
            "days": "90", "trade_quantity": "0.5", "pop_size": 20, "generations": 10,
            "trade_style": "swing_trader"
        }
        response = test_client.post('/api/optimize_strategy', json=payload)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['best_fitness_profit_percent'] == "15.5" # Check Decimal to str
        assert data['best_config_backtest_results']['final_capital'] == "11550"

        mock_ga_run.assert_called_once()
        args, kwargs = mock_ga_run.call_args
        assert kwargs['asset_id'] == "ethereum"
        assert kwargs['asset_type'] == "crypto"
        assert kwargs['initial_capital_str'] == "10000"
        assert kwargs['days_str'] == "90"
        assert kwargs['trade_quantity_str'] == "0.5"
        assert kwargs['pop_size'] == 20
        assert kwargs['generations'] == 10
        assert kwargs['trade_style'] == "swing_trader"
        assert callable(kwargs['get_crypto_history_func']) # Check that the function is passed


    def test_api_optimize_strategy_missing_fields(self, test_client):
        payload = {"asset_id": "litecoin", "initial_capital": "5000"} # Missing days
        response = test_client.post('/api/optimize_strategy', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Missing required fields" in data['error']
        assert "days" in data['error']

    def test_api_optimize_strategy_invalid_types(self, test_client):
        payload = {
            "asset_id": "cardano", "asset_type": "crypto", "initial_capital": "not-a-number",
            "days": "90", "trade_quantity": "1"
        }
        # The actual conversion error will be caught inside run_ga_optimization,
        # which returns an error dict that the API then forwards.
        # The API endpoint itself validates pop_size and generations types.

        # Test invalid pop_size
        payload_invalid_pop = {
            "asset_id": "ada", "asset_type": "crypto", "initial_capital": "1000", "days": "30",
            "pop_size": "small"
        }
        response_pop = test_client.post('/api/optimize_strategy', json=payload_invalid_pop)
        assert response_pop.status_code == 400
        data_pop = json.loads(response_pop.data)
        assert "Invalid input for pop_size or generations" in data_pop['error']

        # Test invalid generations (e.g., negative)
        payload_invalid_gen = {
            "asset_id": "ada", "asset_type": "crypto", "initial_capital": "1000", "days": "30",
            "generations": -5
        }
        response_gen = test_client.post('/api/optimize_strategy', json=payload_invalid_gen)
        assert response_gen.status_code == 400
        data_gen = json.loads(response_gen.data)
        assert "Population size must be greater than 1 and generations must be positive." in data_gen['error']


    @patch('crypto_dashboard.app.run_ga_optimization')
    def test_api_optimize_strategy_ga_returns_error(self, mock_ga_run, test_client):
        mock_ga_run.return_value = {"error": "GA failed to fetch data"}
        payload = {
            "asset_id": "solana", "asset_type": "crypto", "initial_capital": "10000",
            "days": "90", "trade_quantity": "1.0"
        }
        response = test_client.post('/api/optimize_strategy', json=payload)
        assert response.status_code == 400 # Error from GA is passed through
        data = json.loads(response.data)
        assert data['error'] == "GA failed to fetch data"
