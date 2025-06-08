import random
from decimal import Decimal
import copy # For deepcopy

# Attempting to import data fetching functions as specified
# If these cause circular dependencies or issues, they might need to be refactored
# or their sources adjusted.
try:
    from .app import get_crypto_history # This might be problematic if get_crypto_history uses app context
except ImportError:
    # Fallback or alternative if direct import from .app fails
    # This is a common issue. For now, we'll see if the try/except is enough
    # or if we need to define them locally or pass them in.
    # For the scope of this subtask, we'll assume they can be imported or will be mocked.
# print("Warning: Could not import get_crypto_history from .app directly.") # Quieting this print
    get_crypto_history = None # This will be overridden if passed as an argument

from .stock_data_provider import get_stock_history
from .strategies import moving_average_crossover_signal, rsi_signal, bollinger_bands_signal
from .backtesting import run_backtest

# --- Constants and Configuration ---

BASE_STRATEGIES_DETAILS = [
    {
        "id": "moving_average_crossover",
        "params_spec": {
            "short_window": {"type": "int", "min": 5, "max": 20},
            "long_window": {"type": "int", "min": 21, "max": 50},
        }
    },
    {
        "id": "rsi",
        "params_spec": {
            "rsi_period": {"type": "int", "min": 7, "max": 21},
            "rsi_oversold": {"type": "int", "min": 20, "max": 40},
            "rsi_overbought": {"type": "int", "min": 60, "max": 80},
        }
    },
    {
        "id": "bollinger_bands",
        "params_spec": {
            "period": {"type": "int", "min": 10, "max": 30},
            "std_devs": {"type": "int", "min": 1, "max": 3}, # Typically float, but int for simplicity
        }
    }
]
AVAILABLE_LOGICS = ["AND", "OR", "MAJORITY_VOTE", None] # None for single strategy

STRATEGY_FUNCTION_MAP = {
    "moving_average_crossover": moving_average_crossover_signal,
    "rsi": rsi_signal,
    "bollinger_bands": bollinger_bands_signal,
}

# --- Helper for Parameter Generation ---
def _generate_random_parameter(param_spec):
    if param_spec["type"] == "int":
        return random.randint(param_spec["min"], param_spec["max"])
    elif param_spec["type"] == "float": # Though current spec uses int for std_devs
        return random.uniform(param_spec["min"], param_spec["max"])
    # Add other types if needed
    return None

# --- Genetic Algorithm Core Functions ---

def generate_random_individual(base_details, logics, max_strategies_in_combo=2):
    num_strategies = random.randint(1, max_strategies_in_combo)

    combination_logic = None
    if num_strategies > 1:
        # Filter out None from logics if we have more than one strategy
        eligible_logics = [l for l in logics if l is not None]
        if eligible_logics:
            combination_logic = random.choice(eligible_logics)
        else: # Should not happen if AVAILABLE_LOGICS is defined correctly
            combination_logic = "AND" # Default fallback

    definitions = []
    chosen_strategy_ids = [] # To avoid choosing the same strategy type multiple times in a combo

    for _ in range(num_strategies):
        available_strategies_to_choose = [s for s in base_details if s["id"] not in chosen_strategy_ids]
        if not available_strategies_to_choose:
            break # Cannot choose more unique strategies

        chosen_strategy_detail = random.choice(available_strategies_to_choose)
        chosen_strategy_ids.append(chosen_strategy_detail["id"])

        generated_params = {}
        for param_name, spec in chosen_strategy_detail["params_spec"].items():
            generated_params[param_name] = _generate_random_parameter(spec)

        # Ensure constraints (MA short < long, RSI oversold < overbought)
        if chosen_strategy_detail["id"] == "moving_average_crossover":
            if generated_params["short_window"] >= generated_params["long_window"]:
                # Swap or regenerate: simple swap for now
                generated_params["short_window"], generated_params["long_window"] = \
                    min(generated_params["short_window"], generated_params["long_window"] -1 ), \
                    max(generated_params["short_window"]+1, generated_params["long_window"])
                # Ensure min/max are still respected after potential swap, adjust if necessary
                generated_params["short_window"] = max(chosen_strategy_detail["params_spec"]["short_window"]["min"], generated_params["short_window"])
                generated_params["long_window"] = min(chosen_strategy_detail["params_spec"]["long_window"]["max"], generated_params["long_window"])
                if generated_params["short_window"] >= generated_params["long_window"]: # If still bad, set to reasonable defaults
                    generated_params["short_window"] = chosen_strategy_detail["params_spec"]["short_window"]["min"]
                    generated_params["long_window"] = chosen_strategy_detail["params_spec"]["long_window"]["min"]


        elif chosen_strategy_detail["id"] == "rsi":
            if generated_params["rsi_oversold"] >= generated_params["rsi_overbought"]:
                generated_params["rsi_oversold"], generated_params["rsi_overbought"] = \
                    min(generated_params["rsi_oversold"], generated_params["rsi_overbought"]-1), \
                    max(generated_params["rsi_oversold"]+1, generated_params["rsi_overbought"])
                generated_params["rsi_oversold"] = max(chosen_strategy_detail["params_spec"]["rsi_oversold"]["min"], generated_params["rsi_oversold"])
                generated_params["rsi_overbought"] = min(chosen_strategy_detail["params_spec"]["rsi_overbought"]["max"], generated_params["rsi_overbought"])
                if generated_params["rsi_oversold"] >= generated_params["rsi_overbought"]:
                    generated_params["rsi_oversold"] = chosen_strategy_detail["params_spec"]["rsi_oversold"]["min"]
                    generated_params["rsi_overbought"] = chosen_strategy_detail["params_spec"]["rsi_overbought"]["min"]


        definitions.append({
            "id": chosen_strategy_detail["id"],
            "params": generated_params
        })

    return {"logic": combination_logic, "definitions": definitions}


def calculate_fitness(individual_config_params_only, historical_data, initial_capital, trade_quantity, strat_func_map, trade_style="default"):
    if not historical_data:
        return -999999.0

    # Deepcopy to prevent modification of the original individual config
    config_for_backtest = copy.deepcopy(individual_config_params_only)

    # Map strategy IDs to functions
    for definition in config_for_backtest["definitions"]:
        if definition["id"] in strat_func_map:
            definition["function"] = strat_func_map[definition["id"]]
        else:
            # print(f"Error: Strategy function for ID '{definition['id']}' not found.") # TODO: Logging
            return -999999.0 # Invalid config

    try:
        results = run_backtest(
            strategy_config=config_for_backtest,
            historical_data=historical_data,
            initial_capital=initial_capital,
            trade_quantity=trade_quantity,
            trade_style=trade_style # GA could also optimize this later
        )
        if isinstance(results, dict) and 'profit_loss_percent' in results:
            return float(results['profit_loss_percent']) # Ensure it's float
        else:
            # print(f"Warning: Backtest did not return expected dict or key. Result: {results}") # TODO: Logging
            return -999999.0
    except Exception as e:
        # print(f"Error during backtest in fitness calculation: {e}") # TODO: Logging
        return -999999.0


def selection(population_with_fitness, num_parents):
    """Placeholder: Selects top N individuals."""
    population_with_fitness.sort(key=lambda x: x[1], reverse=True) # Sort by fitness desc

    parents = []
    for i in range(min(num_parents, len(population_with_fitness))):
        parents.append(population_with_fitness[i][0]) # Add the individual config
    return parents

def crossover(parent1, parent2):
    """Placeholder: Returns a copy of parent1."""
    return copy.deepcopy(parent1)

def mutation(individual, base_details, logics, mutation_rate=0.1):
    """Placeholder: Returns a copy of the individual or a new random one."""
    if random.random() < mutation_rate:
        # For simplicity, placeholder mutation generates a new random individual
        # A more sophisticated mutation would alter parts of the existing individual
        return generate_random_individual(base_details, logics)
    return copy.deepcopy(individual)

# --- Main GA Orchestration ---

def run_ga_optimization(asset_id, asset_type, initial_capital_str, days_str, trade_quantity_str,
                        pop_size=10, generations=5, trade_style="default",
                        # Add get_crypto_history_func as a parameter
                        get_crypto_history_func=None):

    # --- 1. Fetch Historical Data ---
    historical_data = None

    # Use the passed-in function for fetching crypto history
    if asset_type == 'crypto':
        if get_crypto_history_func:
            raw_hist = get_crypto_history_func(crypto_id=asset_id, days=days_str)
            if raw_hist and raw_hist.get("prices"):
                historical_data = [[ts, Decimal(str(p))] for ts, p in raw_hist["prices"]]
        else:
            # This case means the function wasn't passed in, which is an issue for crypto.
            return {"error": "Crypto data fetching function was not provided to optimizer."}
    elif asset_type == 'stock':
        raw_hist = get_stock_history(symbol=asset_id, period=f"{days_str}d") # yfinance uses 'd' for days
        if raw_hist:
             historical_data = [[ts, Decimal(str(p))] for ts, p in raw_hist]
    else:
        return {"error": f"Invalid asset_type: {asset_type}"}

    if not historical_data:
        return {"error": f"Could not fetch historical data for {asset_id} (type: {asset_type}, days: {days_str})."}

    # --- 2. Validate and Convert Inputs ---
    try:
        initial_capital = Decimal(str(initial_capital_str))
        # days = int(days_str) # days_str is already used for fetching, not needed as int here
        trade_quantity = Decimal(str(trade_quantity_str))
        if initial_capital <= Decimal('0') or trade_quantity <= Decimal('0'):
            return {"error": "Initial capital and trade quantity must be positive."}
    except Exception as e:
        return {"error": f"Invalid input for capital or trade quantity: {str(e)}"}

    # --- 3. Initialize Population ---
    population = [generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS) for _ in range(pop_size)]

    best_overall_individual_config = None
    best_overall_fitness = -float('inf')
    best_backtest_results = None # To store full results of the best

    # --- 4. GA Loop ---
    for gen in range(generations):
        # print(f"Generation {gen + 1}/{generations}") # TODO: Logging

        population_with_fitness = []
        for individual_config in population:
            fitness = calculate_fitness(
                individual_config_params_only=individual_config,
                historical_data=historical_data,
                initial_capital=initial_capital,
                trade_quantity=trade_quantity,
                strat_func_map=STRATEGY_FUNCTION_MAP,
                trade_style=trade_style
            )
            population_with_fitness.append((individual_config, fitness))

        # Find best in current generation and update overall best
        current_gen_best_individual, current_gen_best_fitness = max(population_with_fitness, key=lambda x: x[1], default=(None, -float('inf')))

        if current_gen_best_individual and current_gen_best_fitness > best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_individual_config = copy.deepcopy(current_gen_best_individual)

            # Re-run backtest for the best one to get full results
            config_for_best_backtest = copy.deepcopy(best_overall_individual_config)
            for definition in config_for_best_backtest["definitions"]:
                if definition["id"] in STRATEGY_FUNCTION_MAP: # Ensure 'function' key is added
                    definition["function"] = STRATEGY_FUNCTION_MAP[definition["id"]]

            best_backtest_results = run_backtest(
                strategy_config=config_for_best_backtest,
                historical_data=historical_data,
                initial_capital=initial_capital,
                trade_quantity=trade_quantity,
                trade_style=trade_style
            )

        # Selection
        num_parents = max(2, pop_size // 2) if pop_size >=2 else pop_size
        parents_configs = selection(population_with_fitness, num_parents=num_parents)

        if not parents_configs:
            population = [generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS) for _ in range(pop_size)]
            continue

        # Crossover and Mutation
        next_population = []
        if best_overall_individual_config and best_overall_fitness > -float('inf'):
             next_population.append(copy.deepcopy(best_overall_individual_config)) # Elitism

        while len(next_population) < pop_size:
            parent1 = random.choice(parents_configs)
            parent2 = random.choice(parents_configs) if len(parents_configs) > 1 else parent1
            child = crossover(parent1, parent2)
            child = mutation(child, BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS, mutation_rate=0.1)
            next_population.append(child)

        population = next_population[:pop_size]

    # --- 5. Return Results ---
    return {
        "best_config": best_overall_individual_config,
        "best_fitness_profit_percent": float(best_overall_fitness) if best_overall_fitness != -float('inf') else None,
        "best_config_backtest_results": best_backtest_results # This can be large
    }

if __name__ == '__main__':
    print("Running GA optimizer example (requires proper environment/imports)...")

    def mock_get_crypto_history_for_ga_main(crypto_id, days):
        print(f"Mock fetching crypto for GA main: {crypto_id} for {days} days")
        return {"prices": [[(1678886400000 + i*86400000), (100 + i*0.5 - (i%3)*5 + random.uniform(-1,1))] for i in range(int(days))]}

    # Override get_crypto_history for the __main__ block if the import was problematic
    # This ensures the __main__ example can run even if the .app import is tricky.
    original_get_crypto_history = get_crypto_history
    get_crypto_history = mock_get_crypto_history_for_ga_main

    results = run_ga_optimization(
        asset_id="test_coin",
        asset_type="crypto",
        initial_capital_str="10000",
        days_str="60", # shorter for quicker test
        trade_quantity_str="1.0",
        pop_size=6,
        generations=3
    )

    get_crypto_history = original_get_crypto_history # Restore if it was changed

    print("\nGA Optimization Results:")
    if results.get("error"):
        print(f"Error: {results['error']}")
    else:
        print(f"Best Fitness (Profit %): {results.get('best_fitness_profit_percent')}")
        print(f"Best Config: {results.get('best_config')}")
        if results.get('best_config_backtest_results') and isinstance(results.get('best_config_backtest_results'), dict):
            print(f"Backtest P/L for Best Config: {results['best_config_backtest_results'].get('profit_loss')}")
            print(f"Backtest Trades for Best Config: {len(results['best_config_backtest_results'].get('trades', []))}")
        elif results.get('best_config_backtest_results'):
             print(f"Best config backtest results (unexpected format): {results.get('best_config_backtest_results')}")
