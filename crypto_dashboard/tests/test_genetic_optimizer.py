import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from crypto_dashboard.genetic_optimizer import (
    generate_random_individual,
    calculate_fitness,
    BASE_STRATEGIES_DETAILS,
    AVAILABLE_LOGICS,
    STRATEGY_FUNCTION_MAP
)
# Assuming run_backtest is imported by genetic_optimizer, so it might be patched there if necessary
# from crypto_dashboard.backtesting import run_backtest # Not directly used here, but by calculate_fitness

class TestGenerateRandomIndividual:
    def test_structure_and_basic_constraints(self):
        for i in range(50): # Run a few times due to randomness
            individual = generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS, max_strategies_in_combo=2)

            assert "logic" in individual
            assert "definitions" in individual
            assert isinstance(individual["definitions"], list)
            assert 1 <= len(individual["definitions"]) <= 2

            if len(individual["definitions"]) == 1:
                assert individual["logic"] is None
            else:
                assert individual["logic"] in ["AND", "OR", "MAJORITY_VOTE"]

            for definition in individual["definitions"]:
                assert "id" in definition
                assert definition["id"] in STRATEGY_FUNCTION_MAP.keys()
                assert "params" in definition
                assert isinstance(definition["params"], dict)

                # Check strategy-specific constraints
                strategy_detail = next(s for s in BASE_STRATEGIES_DETAILS if s["id"] == definition["id"])
                for param_name, spec in strategy_detail["params_spec"].items():
                    assert param_name in definition["params"]
                    param_val = definition["params"][param_name]
                    assert spec["min"] <= param_val <= spec["max"]

                if definition["id"] == "moving_average_crossover":
                    assert definition["params"]["short_window"] < definition["params"]["long_window"]
                elif definition["id"] == "rsi":
                    assert definition["params"]["rsi_oversold"] < definition["params"]["rsi_overbought"]

    def test_max_strategies_in_combo_respected(self):
        individual_single = generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS, max_strategies_in_combo=1)
        assert len(individual_single["definitions"]) == 1
        assert individual_single["logic"] is None

        individual_double = generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS, max_strategies_in_combo=2)
        assert 1 <= len(individual_double["definitions"]) <= 2


class TestCalculateFitness:
    mock_historical_data = [[i, Decimal(str(100 + i))] for i in range(20)] # Sample data
    initial_capital = Decimal("10000")
    trade_quantity = Decimal("1.0")

    @patch('crypto_dashboard.genetic_optimizer.run_backtest')
    def test_successful_backtest_call(self, mock_run_backtest):
        mock_run_backtest.return_value = {
            "profit_loss_percent": Decimal("12.34"),
            # Other fields that run_backtest would return
        }

        individual_config = {
            "logic": None,
            "definitions": [
                {"id": "rsi", "params": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}}
            ]
        }

        fitness = calculate_fitness(
            individual_config_params_only=individual_config,
            historical_data=self.mock_historical_data,
            initial_capital=self.initial_capital,
            trade_quantity=self.trade_quantity,
            strat_func_map=STRATEGY_FUNCTION_MAP,
            trade_style="default"
        )

        assert fitness == 12.34
        mock_run_backtest.assert_called_once()
        args, kwargs = mock_run_backtest.call_args

        # Check that the strategy_config passed to run_backtest has the function mapped
        called_strategy_config = kwargs['strategy_config']
        assert called_strategy_config["definitions"][0]["id"] == "rsi"
        assert callable(called_strategy_config["definitions"][0]["function"])
        assert called_strategy_config["definitions"][0]["function"] == STRATEGY_FUNCTION_MAP["rsi"]
        assert called_strategy_config["definitions"][0]["params"] == individual_config["definitions"][0]["params"]

        assert kwargs['historical_data'] == self.mock_historical_data
        assert kwargs['initial_capital'] == self.initial_capital
        assert kwargs['trade_quantity'] == self.trade_quantity
        assert kwargs['trade_style'] == "default"


    @patch('crypto_dashboard.genetic_optimizer.run_backtest')
    def test_backtest_returns_invalid_format(self, mock_run_backtest):
        mock_run_backtest.return_value = {"error": "Something went wrong"} # Missing profit_loss_percent
        individual_config = {"logic": None, "definitions": [{"id": "rsi", "params": {}}]}

        fitness = calculate_fitness(individual_config, self.mock_historical_data, self.initial_capital,
                                    self.trade_quantity, STRATEGY_FUNCTION_MAP)
        assert fitness == -999999.0

    @patch('crypto_dashboard.genetic_optimizer.run_backtest')
    def test_backtest_raises_exception(self, mock_run_backtest):
        mock_run_backtest.side_effect = Exception("Unexpected error in backtest")
        individual_config = {"logic": None, "definitions": [{"id": "rsi", "params": {}}]}

        fitness = calculate_fitness(individual_config, self.mock_historical_data, self.initial_capital,
                                    self.trade_quantity, STRATEGY_FUNCTION_MAP)
        assert fitness == -999999.0

    def test_no_historical_data(self):
        individual_config = {"logic": None, "definitions": [{"id": "rsi", "params": {}}]}
        fitness = calculate_fitness(individual_config, [], self.initial_capital,
                                    self.trade_quantity, STRATEGY_FUNCTION_MAP)
        assert fitness == -999999.0

    @patch('crypto_dashboard.genetic_optimizer.run_backtest')
    def test_unknown_strategy_id_in_config(self, mock_run_backtest):
        # This should be caught before run_backtest if STRATEGY_FUNCTION_MAP is used correctly
        # calculate_fitness maps functions; if an ID is not in map, it should fail early.
        individual_config = {
            "logic": None,
            "definitions": [{"id": "non_existent_strategy", "params": {}}]
        }
        fitness = calculate_fitness(individual_config, self.mock_historical_data, self.initial_capital,
                                    self.trade_quantity, STRATEGY_FUNCTION_MAP)
        assert fitness == -999999.0
        mock_run_backtest.assert_not_called() # Because mapping should fail
