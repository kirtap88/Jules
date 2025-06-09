import random
from decimal import Decimal
import copy # For deepcopy

# Attempting to import data fetching functions as specified
try:
    from .app import get_crypto_history
except ImportError:
    get_crypto_history = None

from .stock_data_provider import get_stock_history
from .strategies import moving_average_crossover_signal, rsi_signal, bollinger_bands_signal, macd_signal, stochastic_oscillator_signal
from .backtesting import run_backtest

# --- Constants and Configuration ---

BASE_STRATEGIES_DETAILS = [
    {
        "id": "moving_average_crossover",
        "params_spec": {
            "short_window": {"type": "int", "min": 5, "max": 20},
            "long_window": {"type": "int", "min": 21, "max": 50}, # max_short < min_long
        }
    },
    {
        "id": "rsi",
        "params_spec": {
            "rsi_period": {"type": "int", "min": 7, "max": 21},
            "rsi_oversold": {"type": "int", "min": 20, "max": 40}, # max_oversold < min_overbought
            "rsi_overbought": {"type": "int", "min": 60, "max": 80},
        }
    },
    {
        "id": "bollinger_bands",
        "params_spec": {
            "period": {"type": "int", "min": 10, "max": 30},
            "std_devs": {"type": "int", "min": 1, "max": 3},
        }
    },
    {
        "id": "macd",
        "params_spec": {
            "short_ema_period": {"type": "int", "min": 5, "max": 20},
            "long_ema_period": {"type": "int", "min": 21, "max": 50}, # Ensures short_ema < long_ema
            "signal_period": {"type": "int", "min": 5, "max": 15},
        }
    },
    {
        "id": "stochastic_oscillator",
        "params_spec": {
            "k_period": {"type": "int", "min": 5, "max": 20},
            "d_period": {"type": "int", "min": 3, "max": 7},
            "oversold_level": {"type": "int", "min": 10, "max": 30}, # Ensures oversold < overbought by range def
            "overbought_level": {"type": "int", "min": 70, "max": 90},
        }
    }
]
AVAILABLE_LOGICS = ["AND", "OR", "MAJORITY_VOTE", None]

STRATEGY_FUNCTION_MAP = {
    "moving_average_crossover": moving_average_crossover_signal,
    "rsi": rsi_signal,
    "bollinger_bands": bollinger_bands_signal,
    "macd": macd_signal,
    "stochastic_oscillator": stochastic_oscillator_signal,
}

# --- Helper for Parameter Generation ---
def _generate_random_parameter(param_spec):
    if param_spec["type"] == "int":
        return random.randint(param_spec["min"], param_spec["max"])
    elif param_spec["type"] == "float":
        return random.uniform(param_spec["min"], param_spec["max"])
    return None

# --- Genetic Algorithm Core Functions ---

def generate_random_individual(base_details, logics, max_strategies_in_combo=2):
    num_strategies = random.randint(1, max_strategies_in_combo)
    combination_logic = None
    if num_strategies > 1:
        eligible_logics = [l for l in logics if l is not None]
        if eligible_logics:
            combination_logic = random.choice(eligible_logics)
        else:
            combination_logic = "AND"

    definitions = []
    chosen_strategy_ids = []

    for _ in range(num_strategies):
        available_strategies_to_choose = [s for s in base_details if s["id"] not in chosen_strategy_ids]
        if not available_strategies_to_choose:
            break

        chosen_strategy_detail = random.choice(available_strategies_to_choose)
        chosen_strategy_ids.append(chosen_strategy_detail["id"])

        generated_params = {}
        for param_name, spec in chosen_strategy_detail["params_spec"].items():
            generated_params[param_name] = _generate_random_parameter(spec)

        # Constraint fixing for MA, RSI, MACD, and Stochastic (paired params like short/long, oversold/overbought)
        # is currently NOT NEEDED here because their min/max parameter ranges in BASE_STRATEGIES_DETAILS
        # are defined to inherently satisfy these constraints (e.g., max_short_period < min_long_period).
        # This logic IS necessary in crossover and mutation where params can be altered
        # in ways that might violate these constraints.

        definitions.append({
            "id": chosen_strategy_detail["id"],
            "params": generated_params
        })

    return {"logic": combination_logic, "definitions": definitions}


def calculate_fitness(individual_config_params_only, historical_data, initial_capital, trade_quantity, strat_func_map, trade_style="default"):
    if not historical_data:
        return -999999.0
    config_for_backtest = copy.deepcopy(individual_config_params_only)
    for definition in config_for_backtest["definitions"]:
        if definition["id"] in strat_func_map:
            definition["function"] = strat_func_map[definition["id"]]
        else:
            return -999999.0
    try:
        results = run_backtest(
            strategy_config=config_for_backtest, historical_data=historical_data,
            initial_capital=initial_capital, trade_quantity=trade_quantity,
            trade_style=trade_style
        )
        if isinstance(results, dict) and 'profit_loss_percent' in results:
            return float(results['profit_loss_percent'])
        else:
            return -999999.0
    except Exception as e:
        return -999999.0


def selection(population_with_fitness: list, num_parents: int, tournament_size: int = 3) -> list:
    selected_parents = []
    if not population_with_fitness: return selected_parents
    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if actual_tournament_size <= 0: return selected_parents
    for _ in range(num_parents):
        tournament_contenders = random.sample(population_with_fitness, actual_tournament_size)
        winner = max(tournament_contenders, key=lambda item: item[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parent1: dict, parent2: dict, base_strategies_details: list) -> dict:
    offspring = {"logic": random.choice([parent1.get("logic"), parent2.get("logic")]), "definitions": []}
    p1_definitions = parent1.get("definitions", [])
    p2_definitions_map = {defn["id"]: defn["params"] for defn in parent2.get("definitions", [])}

    for p1_strat_def in p1_definitions:
        offspring_strat_def = copy.deepcopy(p1_strat_def)
        p1_strat_id = p1_strat_def["id"]
        current_strat_spec = next((spec for spec in base_strategies_details if spec["id"] == p1_strat_id), None)
        if not current_strat_spec:
            offspring["definitions"].append(offspring_strat_def)
            continue

        if p1_strat_id in p2_definitions_map:
            p2_strat_params = p2_definitions_map[p1_strat_id]
            new_params = {}
            for param_name, p1_val in p1_strat_def["params"].items():
                p2_val = p2_strat_params.get(param_name)
                param_spec_detail = current_strat_spec["params_spec"].get(param_name)
                if p2_val is not None and param_spec_detail and param_spec_detail["type"] in ["int", "float"]:
                    avg_val = (p1_val + p2_val) / 2
                    if param_spec_detail["type"] == "int": avg_val = round(avg_val)
                    min_val, max_val = param_spec_detail.get("min"), param_spec_detail.get("max")
                    if min_val is not None: avg_val = max(min_val, avg_val)
                    if max_val is not None: avg_val = min(max_val, avg_val)
                    new_params[param_name] = avg_val
                else:
                    new_params[param_name] = p1_val
            offspring_strat_def["params"] = new_params

        validated_params = offspring_strat_def["params"]
        if p1_strat_id == "moving_average_crossover":
            if validated_params.get("short_window") >= validated_params.get("long_window", float('inf')):
                short_spec = current_strat_spec["params_spec"]["short_window"]
                long_spec = current_strat_spec["params_spec"]["long_window"]
                if validated_params["short_window"] == short_spec["min"]:
                    validated_params["long_window"] = min(max(validated_params["long_window"], validated_params["short_window"] + 1), long_spec["max"])
                else:
                    validated_params["short_window"] = min(validated_params["short_window"], validated_params.get("long_window", float('inf')) - 1)
                validated_params["short_window"] = max(short_spec["min"], validated_params["short_window"])
                if validated_params["short_window"] >= validated_params.get("long_window", float('inf')):
                     validated_params["short_window"], validated_params["long_window"] = p1_strat_def["params"]["short_window"], p1_strat_def["params"]["long_window"]
        elif p1_strat_id == "rsi":
            if validated_params.get("rsi_oversold") >= validated_params.get("rsi_overbought", float('inf')):
                validated_params["rsi_oversold"], validated_params["rsi_overbought"] = p1_strat_def["params"]["rsi_oversold"], p1_strat_def["params"]["rsi_overbought"]
        elif p1_strat_id == "macd":
            if validated_params.get("short_ema_period") >= validated_params.get("long_ema_period", float('inf')):
                validated_params["short_ema_period"], validated_params["long_ema_period"] = p1_strat_def["params"]["short_ema_period"], p1_strat_def["params"]["long_ema_period"]
        elif p1_strat_id == "stochastic_oscillator":
            if validated_params.get("oversold_level") >= validated_params.get("overbought_level", float('inf')):
                validated_params["oversold_level"], validated_params["overbought_level"] = p1_strat_def["params"]["oversold_level"], p1_strat_def["params"]["overbought_level"]

        offspring_strat_def["params"] = validated_params
        offspring["definitions"].append(offspring_strat_def)

    if not offspring["definitions"] and (p1_definitions or p2_definitions_map):
        offspring["definitions"] = copy.deepcopy(p1_definitions) if p1_definitions else [{"id": k, "params": copy.deepcopy(v)} for k,v in p2_definitions_map.items()]
    if len(offspring["definitions"]) <= 1: offspring["logic"] = None
    elif offspring["logic"] is None and len(offspring["definitions"]) > 1:
        eligible_logics = [l for l in AVAILABLE_LOGICS if l is not None]
        if eligible_logics: offspring["logic"] = random.choice(eligible_logics)
        else: offspring["logic"] = "AND"
    return offspring

def mutation(individual: dict, base_strategies_details: list, mutation_rate: float = 0.1, strategy_change_rate: float = 0.05, logic_change_rate: float = 0.05) -> dict:
    mutated_individual = copy.deepcopy(individual)
    if random.random() < logic_change_rate and len(mutated_individual.get("definitions", [])) > 1:
        current_logic = mutated_individual.get("logic")
        possible_new_logics = [l for l in AVAILABLE_LOGICS if l != current_logic and l is not None]
        if possible_new_logics: mutated_individual["logic"] = random.choice(possible_new_logics)

    new_definitions = []
    for strat_def in mutated_individual.get("definitions", []):
        mutated_strat_def = copy.deepcopy(strat_def)
        original_strat_id_for_param_mutation = mutated_strat_def["id"]

        current_strat_spec = next((s_spec for s_spec in base_strategies_details if s_spec["id"] == original_strat_id_for_param_mutation), None)
        if not current_strat_spec:
            new_definitions.append(mutated_strat_def)
            continue

        active_spec_for_constraints = current_strat_spec # Default to current spec

        if random.random() < strategy_change_rate:
            possible_new_ids = [s_spec["id"] for s_spec in base_strategies_details if s_spec["id"] != original_strat_id_for_param_mutation]
            if not possible_new_ids: possible_new_ids = [original_strat_id_for_param_mutation]

            if possible_new_ids:
                new_id = random.choice(possible_new_ids)
                active_spec_for_constraints = next(s_spec for s_spec in base_strategies_details if s_spec["id"] == new_id)
                new_random_params = {p_name: _generate_random_parameter(p_spec) for p_name, p_spec in active_spec_for_constraints["params_spec"].items()}
                mutated_strat_def["id"] = new_id
                mutated_strat_def["params"] = new_random_params

        if mutated_strat_def["id"] == original_strat_id_for_param_mutation: # Only mutate params if ID didn't change
            new_params = {}
            for p_name, p_val in mutated_strat_def["params"].items():
                if random.random() < mutation_rate:
                    p_spec_detail = current_strat_spec["params_spec"].get(p_name)
                    if p_spec_detail:
                        mutated_val = p_val
                        if p_spec_detail["type"] == "int":
                            change = random.choice([-2, -1, 1, 2])
                            mutated_val = p_val + change
                        elif p_spec_detail["type"] == "float":
                            change_percent = random.uniform(-0.1, 0.1)
                            mutated_val = p_val * (1 + change_percent)
                        min_val, max_val = p_spec_detail.get("min"), p_spec_detail.get("max")
                        if min_val is not None: mutated_val = max(min_val, mutated_val)
                        if max_val is not None: mutated_val = min(max_val, mutated_val)
                        if p_spec_detail["type"] == "int": mutated_val = round(mutated_val)
                        new_params[p_name] = mutated_val
                    else: new_params[p_name] = p_val
                else: new_params[p_name] = p_val
            mutated_strat_def["params"] = new_params

        # Re-apply constraints after any parameter mutation or ID change
        params = mutated_strat_def["params"]
        strat_id = mutated_strat_def["id"]

        if strat_id == "moving_average_crossover":
            if params["short_window"] >= params["long_window"]:
                params["short_window"] = min(params["short_window"], params["long_window"] - 1)
                params["short_window"] = max(params["short_window"], active_spec_for_constraints["params_spec"]["short_window"]["min"])
                if params["short_window"] >= params["long_window"]:
                     params["long_window"] = params["short_window"] + 1
                     params["long_window"] = min(params["long_window"], active_spec_for_constraints["params_spec"]["long_window"]["max"])
                     if params["short_window"] >= params["long_window"]:
                         params["short_window"] = active_spec_for_constraints["params_spec"]["short_window"]["min"]
                         params["long_window"] = active_spec_for_constraints["params_spec"]["long_window"]["min"]
        elif strat_id == "rsi":
            if params["rsi_oversold"] >= params["rsi_overbought"]:
                params["rsi_oversold"] = min(params["rsi_oversold"], params["rsi_overbought"] - 1)
                params["rsi_oversold"] = max(params["rsi_oversold"], active_spec_for_constraints["params_spec"]["rsi_oversold"]["min"])
                if params["rsi_oversold"] >= params["rsi_overbought"]:
                     params["rsi_overbought"] = params["rsi_oversold"] + 1
                     params["rsi_overbought"] = min(params["rsi_overbought"], active_spec_for_constraints["params_spec"]["rsi_overbought"]["max"])
                     if params["rsi_oversold"] >= params["rsi_overbought"]:
                         params["rsi_oversold"] = active_spec_for_constraints["params_spec"]["rsi_oversold"]["min"]
                         params["rsi_overbought"] = active_spec_for_constraints["params_spec"]["rsi_overbought"]["min"]
        elif strat_id == "macd":
            if params["short_ema_period"] >= params["long_ema_period"]:
                params["short_ema_period"] = min(params["short_ema_period"], params["long_ema_period"] - 1)
                params["short_ema_period"] = max(params["short_ema_period"], active_spec_for_constraints["params_spec"]["short_ema_period"]["min"])
                if params["short_ema_period"] >= params["long_ema_period"]:
                    params["long_ema_period"] = params["short_ema_period"] + 1
                    params["long_ema_period"] = min(params["long_ema_period"], active_spec_for_constraints["params_spec"]["long_ema_period"]["max"])
                    if params["short_ema_period"] >= params["long_ema_period"]:
                        params["short_ema_period"] = active_spec_for_constraints["params_spec"]["short_ema_period"]["min"]
                        params["long_ema_period"] = active_spec_for_constraints["params_spec"]["long_ema_period"]["min"]
        elif strat_id == "stochastic_oscillator":
            if params["oversold_level"] >= params["overbought_level"]:
                params["oversold_level"] = min(params["oversold_level"], params["overbought_level"] - 1)
                params["oversold_level"] = max(params["oversold_level"], active_spec_for_constraints["params_spec"]["oversold_level"]["min"])
                if params["oversold_level"] >= params["overbought_level"]:
                    params["overbought_level"] = params["oversold_level"] + 1
                    params["overbought_level"] = min(params["overbought_level"], active_spec_for_constraints["params_spec"]["overbought_level"]["max"])
                    if params["oversold_level"] >= params["overbought_level"]:
                        params["oversold_level"] = active_spec_for_constraints["params_spec"]["oversold_level"]["min"]
                        params["overbought_level"] = active_spec_for_constraints["params_spec"]["overbought_level"]["min"]

        new_definitions.append(mutated_strat_def)
    mutated_individual["definitions"] = new_definitions

    if len(mutated_individual["definitions"]) <= 1: mutated_individual["logic"] = None
    elif mutated_individual["logic"] is None and len(mutated_individual["definitions"]) > 1:
            mutated_individual["logic"] = random.choice([l for l in AVAILABLE_LOGICS if l is not None])
    return mutated_individual

# --- Main GA Orchestration ---
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_STRATEGY_CHANGE_RATE = 0.05
DEFAULT_LOGIC_CHANGE_RATE = 0.05
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_NUM_ELITES = 2

def run_ga_optimization(asset_id, asset_type, initial_capital_str, days_str, trade_quantity_str,
                        pop_size=20, generations=10, trade_style="default",
                        get_crypto_history_func=None,
                        mutation_rate=DEFAULT_MUTATION_RATE,
                        strategy_change_rate=DEFAULT_STRATEGY_CHANGE_RATE,
                        logic_change_rate=DEFAULT_LOGIC_CHANGE_RATE,
                        tournament_size=DEFAULT_TOURNAMENT_SIZE,
                        num_elites=DEFAULT_NUM_ELITES
                        ):
    historical_data = None
    if asset_type == 'crypto':
        if get_crypto_history_func:
            raw_hist = get_crypto_history_func(crypto_id=asset_id, days=days_str)
            if raw_hist and raw_hist.get("prices"):
                historical_data = [[ts, Decimal(str(p))] for ts, p in raw_hist["prices"]]
        else:
            return {"error": "Crypto data fetching function was not provided to optimizer."}
    elif asset_type == 'stock':
        raw_hist = get_stock_history(symbol=asset_id, period=f"{days_str}d")
        if raw_hist: historical_data = [[ts, Decimal(str(p))] for ts, p in raw_hist]
    else: return {"error": f"Invalid asset_type: {asset_type}"}
    if not historical_data: return {"error": f"Could not fetch historical data for {asset_id} (type: {asset_type}, days: {days_str})."}

    try:
        initial_capital = Decimal(str(initial_capital_str))
        trade_quantity = Decimal(str(trade_quantity_str))
        if initial_capital <= Decimal('0') or trade_quantity <= Decimal('0'):
            return {"error": "Initial capital and trade quantity must be positive."}
    except Exception as e: return {"error": f"Invalid input for capital or trade quantity: {str(e)}"}

    population = [generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS) for _ in range(pop_size)]
    best_overall_individual_config, best_overall_fitness, best_backtest_results = None, -float('inf'), None
    actual_num_elites = min(num_elites, pop_size)
    if actual_num_elites < 0: actual_num_elites = 0

    for gen in range(generations):
        population_with_fitness = []
        for individual_config in population:
            fitness = calculate_fitness(individual_config, historical_data, initial_capital, trade_quantity, STRATEGY_FUNCTION_MAP, trade_style)
            population_with_fitness.append((individual_config, fitness))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        if population_with_fitness and population_with_fitness[0][1] > best_overall_fitness:
            best_overall_fitness = population_with_fitness[0][1]
            best_overall_individual_config = copy.deepcopy(population_with_fitness[0][0])
            config_for_best_backtest = copy.deepcopy(best_overall_individual_config)
            for definition in config_for_best_backtest["definitions"]:
                if definition["id"] in STRATEGY_FUNCTION_MAP:
                    definition["function"] = STRATEGY_FUNCTION_MAP[definition["id"]]
            best_backtest_results = run_backtest(
                strategy_config=config_for_best_backtest, historical_data=historical_data,
                initial_capital=initial_capital, trade_quantity=trade_quantity, trade_style=trade_style
            )

        next_population = [copy.deepcopy(p[0]) for p in population_with_fitness[:actual_num_elites]]
        num_offspring_needed = pop_size - len(next_population)
        num_parents_to_select = min(max(2, num_offspring_needed), len(population_with_fitness))
        selected_parents_configs = []
        if num_offspring_needed > 0 and len(population_with_fitness) > 0 :
            selected_parents_configs = selection(population_with_fitness, num_parents=num_parents_to_select, tournament_size=tournament_size)
        if not selected_parents_configs and num_offspring_needed > 0:
            selected_parents_configs = [p[0] for p in population_with_fitness[:num_parents_to_select]]

        while len(next_population) < pop_size:
            if not selected_parents_configs:
                next_population.append(generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS))
                continue
            parent1_config = random.choice(selected_parents_configs)
            parent2_config = random.choice(selected_parents_configs) if len(selected_parents_configs) > 1 else parent1_config
            offspring_config = crossover(parent1_config, parent2_config, BASE_STRATEGIES_DETAILS)
            mutated_offspring_config = mutation(
                offspring_config, BASE_STRATEGIES_DETAILS,
                mutation_rate=mutation_rate, strategy_change_rate=strategy_change_rate, logic_change_rate=logic_change_rate
            )
            next_population.append(mutated_offspring_config)
        population = next_population[:pop_size]

    return {
        "best_config": best_overall_individual_config,
        "best_fitness_profit_percent": float(best_overall_fitness) if best_overall_fitness != -float('inf') else None,
        "best_config_backtest_results": best_backtest_results
    }

if __name__ == '__main__':
    print("Running GA optimizer example (requires proper environment/imports)...")
    def mock_get_crypto_history_for_ga_main(crypto_id, days):
        print(f"Mock fetching crypto for GA main: {crypto_id} for {days} days")
        return {"prices": [[(1678886400000 + i*86400000), (100 + i*0.5 - (i%3)*5 + random.uniform(-1,1))] for i in range(int(days))]}

    _original_get_crypto_history = get_crypto_history
    get_crypto_history = mock_get_crypto_history_for_ga_main
    results = run_ga_optimization(
        asset_id="test_coin", asset_type="crypto", initial_capital_str="10000",
        days_str="60", trade_quantity_str="1.0", pop_size=6, generations=3,
        get_crypto_history_func=get_crypto_history # Pass the mocked one
    )
    get_crypto_history = _original_get_crypto_history

    print("\nGA Optimization Results:")
    if results.get("error"): print(f"Error: {results['error']}")
    else:
        print(f"Best Fitness (Profit %): {results.get('best_fitness_profit_percent')}")
        print(f"Best Config: {results.get('best_config')}")
        if results.get('best_config_backtest_results') and isinstance(results.get('best_config_backtest_results'), dict):
            print(f"Backtest P/L for Best Config: {results['best_config_backtest_results'].get('profit_loss')}")
            print(f"Backtest Trades: {len(results['best_config_backtest_results'].get('trades', []))}")
        elif results.get('best_config_backtest_results'):
             print(f"Best config backtest results (unexpected format): {results.get('best_config_backtest_results')}")

```
