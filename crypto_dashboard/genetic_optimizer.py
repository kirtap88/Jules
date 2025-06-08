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


def selection(population_with_fitness: list, num_parents: int, tournament_size: int = 3) -> list:
    selected_parents = []
    if not population_with_fitness:
        return selected_parents

    # Ensure tournament_size is not larger than the population
    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if actual_tournament_size <= 0: # Should not happen if population_with_fitness is not empty
        return selected_parents

    for _ in range(num_parents):
        tournament_contenders = random.sample(population_with_fitness, actual_tournament_size)
        # Sort contenders by fitness (descending, as fitness is profit_loss_percent)
        # population_with_fitness is a list of tuples: (individual_config, fitness_score)
        winner = max(tournament_contenders, key=lambda item: item[1])
        selected_parents.append(winner[0]) # Append the individual_config part
    return selected_parents

def crossover(parent1: dict, parent2: dict, base_strategies_details: list) -> dict: # base_strategies_details kept for potential future use or validation
    offspring = {
        "logic": None,
        "definitions": []
    }

    # 1. Crossover for combination_logic
    offspring["logic"] = random.choice([parent1.get("logic"), parent2.get("logic")])

    # 2. Crossover for strategy definitions and their parameters
    # For simplicity, offspring will inherit structure from parent1, but parameters will be averaged if parent2 has the same strategy.
    # More advanced: could try to mix strategy lists, handle different lengths, etc.

    p1_definitions = parent1.get("definitions", [])
    p2_definitions_map = {defn["id"]: defn["params"] for defn in parent2.get("definitions", [])}

    for p1_strat_def in p1_definitions:
        offspring_strat_def = copy.deepcopy(p1_strat_def) # Start with parent1's structure and params

        p1_strat_id = p1_strat_def["id"]

        if p1_strat_id in p2_definitions_map:
            p2_strat_params = p2_definitions_map[p1_strat_id]

            # Find param_spec for validation and type info
            current_strat_spec = next((spec for spec in base_strategies_details if spec["id"] == p1_strat_id), None)
            if not current_strat_spec:
                # Should not happen if parents are valid, but as a safeguard:
                offspring["definitions"].append(offspring_strat_def)
                continue

            new_params = {}
            for param_name, p1_val in p1_strat_def["params"].items():
                p2_val = p2_strat_params.get(param_name)
                param_spec_detail = current_strat_spec["params_spec"].get(param_name)

                if p2_val is not None and param_spec_detail and param_spec_detail["type"] in ["int", "float"]:
                    # Average numeric parameters
                    avg_val = (p1_val + p2_val) / 2
                    if param_spec_detail["type"] == "int":
                        avg_val = round(avg_val)

                    # Clamp within min/max
                    min_val = param_spec_detail.get("min")
                    max_val = param_spec_detail.get("max")
                    if min_val is not None:
                        avg_val = max(min_val, avg_val)
                    if max_val is not None:
                        avg_val = min(max_val, avg_val)
                    new_params[param_name] = avg_val
                else:
                    # For non-numeric or if p2 doesn't have the param, keep p1's value
                    new_params[param_name] = p1_val
            offspring_strat_def["params"] = new_params

        # Constraint validation (e.g., MA short < long)
        # Need to use new_params here, not p1_strat_def["params"]
        validated_params = offspring_strat_def["params"] # initially new_params or p1_strat_def's if no commonality

        if p1_strat_id == "moving_average_crossover":
            # Ensure current_strat_spec is available, otherwise this block is problematic
            if current_strat_spec and validated_params.get("short_window") >= validated_params.get("long_window", float('inf')):
                # If invalid, one simple fix: make short slightly less than long
                current_short_min = current_strat_spec["params_spec"]["short_window"]["min"]
                current_long_max = current_strat_spec["params_spec"]["long_window"]["max"] # Not used here, but good to have

                # Try to adjust long_window first if short_window is at its min
                if validated_params["short_window"] == current_short_min:
                    validated_params["long_window"] = max(validated_params["long_window"], validated_params["short_window"] + 1)
                    validated_params["long_window"] = min(validated_params["long_window"], current_strat_spec["params_spec"]["long_window"]["max"])
                else: # Adjust short_window
                    validated_params["short_window"] = min(validated_params["short_window"], validated_params.get("long_window", float('inf')) -1)

                # Ensure short_window respects its minimum bound after adjustment
                validated_params["short_window"] = max(current_short_min, validated_params["short_window"])

                # If still invalid (e.g., long_window was also at its max and short ended up too high), revert
                if validated_params["short_window"] >= validated_params.get("long_window", float('inf')):
                     validated_params["short_window"] = p1_strat_def["params"]["short_window"]
                     validated_params["long_window"] = p1_strat_def["params"]["long_window"]

        elif p1_strat_id == "rsi":
            if current_strat_spec and validated_params.get("rsi_oversold") >= validated_params.get("rsi_overbought", float('inf')):
                # Simple fix: revert to parent1's params for this strat if validation fails
                # More complex adjustments could be made similar to MA crossover if desired
                validated_params["rsi_oversold"] = p1_strat_def["params"]["rsi_oversold"]
                validated_params["rsi_overbought"] = p1_strat_def["params"]["rsi_overbought"]

        offspring_strat_def["params"] = validated_params # Assign potentially corrected params
        offspring["definitions"].append(offspring_strat_def)

    # Ensure offspring has at least one strategy definition if parents did
    if not offspring["definitions"] and (p1_definitions or p2_definitions_map):
            # Fallback: if somehow offspring ended up with no definitions, choose one parent's definitions
            if p1_definitions: # Prioritize parent1's structure
                offspring["definitions"] = copy.deepcopy(p1_definitions)
            else: # Should only happen if p1 had no definitions but p2 did
                 offspring["definitions"] = [{"id": k, "params": copy.deepcopy(v)} for k,v in p2_definitions_map.items()]


    # If only one strategy in offspring, logic should be None
    if len(offspring["definitions"]) <= 1:
        offspring["logic"] = None
    # If more than one strategy but logic became None (e.g. if parents had None), pick one.
    elif offspring["logic"] is None and len(offspring["definitions"]) > 1:
        eligible_logics = [l for l in AVAILABLE_LOGICS if l is not None]
        if eligible_logics: # Should always be true if AVAILABLE_LOGICS is properly defined
            offspring["logic"] = random.choice(eligible_logics)
        else: # Fallback, though should not be reached
            offspring["logic"] = "AND"

    return offspring

def mutation(individual: dict, base_strategies_details: list, mutation_rate: float = 0.1, strategy_change_rate: float = 0.05, logic_change_rate: float = 0.05) -> dict:
    mutated_individual = copy.deepcopy(individual)

    # 1. Mutate combination_logic
    if random.random() < logic_change_rate and len(mutated_individual.get("definitions", [])) > 1:
        current_logic = mutated_individual.get("logic")
        possible_new_logics = [l for l in AVAILABLE_LOGICS if l != current_logic and l is not None] # Ensure it's a valid logic for multiple strategies
        if possible_new_logics:
            mutated_individual["logic"] = random.choice(possible_new_logics)

    # 2. Mutate strategy definitions (parameters or change strategy ID)
    new_definitions = []
    for strat_def in mutated_individual.get("definitions", []):
        mutated_strat_def = copy.deepcopy(strat_def)

        # 2a. Chance to change the strategy ID itself
        if random.random() < strategy_change_rate:
            current_id = mutated_strat_def["id"]
            # Exclude current_id to ensure it actually changes, if possible
            possible_new_ids = [s_spec["id"] for s_spec in base_strategies_details if s_spec["id"] != current_id]
            if not possible_new_ids: # Only one type of strategy defined in base_strategies_details
                possible_new_ids = [current_id] # No change if only one option

            if possible_new_ids:
                new_id = random.choice(possible_new_ids)
                new_strat_spec = next(s_spec for s_spec in base_strategies_details if s_spec["id"] == new_id)

                # Generate new random parameters for the new strategy type
                new_random_params = {}
                for p_name, p_spec in new_strat_spec["params_spec"].items():
                    if p_spec["type"] == "int":
                        val = random.randint(p_spec["min"], p_spec["max"])
                    elif p_spec["type"] == "float": # Assuming float for std_devs if it were float
                        val = random.uniform(p_spec["min"], p_spec["max"])
                    else: # Default for other types (e.g. if we had categorical)
                        val = p_spec.get("default", p_spec["min"]) # Fallback
                    new_random_params[p_name] = val

                mutated_strat_def["id"] = new_id
                mutated_strat_def["params"] = new_random_params
                # Apply constraints for the new strategy's params
                if new_id == "moving_average_crossover":
                    if mutated_strat_def["params"]["short_window"] >= mutated_strat_def["params"]["long_window"]:
                        # Adjust short_window to be less than long_window, respecting min bound
                        mutated_strat_def["params"]["short_window"] = min(mutated_strat_def["params"]["short_window"], mutated_strat_def["params"]["long_window"] -1)
                        mutated_strat_def["params"]["short_window"] = max(mutated_strat_def["params"]["short_window"], new_strat_spec["params_spec"]["short_window"]["min"])
                        # If still invalid (e.g. long_window was at its min), long_window might need to be adjusted upwards if possible or strategy re-randomized.
                        # For simplicity, we ensure short is at least min and less than long.
                        if mutated_strat_def["params"]["short_window"] >= mutated_strat_def["params"]["long_window"]:
                             mutated_strat_def["params"]["long_window"] = mutated_strat_def["params"]["short_window"] + 1
                             mutated_strat_def["params"]["long_window"] = min(mutated_strat_def["params"]["long_window"], new_strat_spec["params_spec"]["long_window"]["max"])
                             # Re-check short if long got capped
                             if mutated_strat_def["params"]["short_window"] >= mutated_strat_def["params"]["long_window"]:
                                 mutated_strat_def["params"]["short_window"] = new_strat_spec["params_spec"]["short_window"]["min"] # fallback to min
                                 mutated_strat_def["params"]["long_window"] = new_strat_spec["params_spec"]["long_window"]["min"]


                elif new_id == "rsi":
                    if mutated_strat_def["params"]["rsi_oversold"] >= mutated_strat_def["params"]["rsi_overbought"]:
                        mutated_strat_def["params"]["rsi_oversold"] = min(mutated_strat_def["params"]["rsi_oversold"], mutated_strat_def["params"]["rsi_overbought"]-1)
                        mutated_strat_def["params"]["rsi_oversold"] = max(mutated_strat_def["params"]["rsi_oversold"], new_strat_spec["params_spec"]["rsi_oversold"]["min"])
                        if mutated_strat_def["params"]["rsi_oversold"] >= mutated_strat_def["params"]["rsi_overbought"]:
                             mutated_strat_def["params"]["rsi_overbought"] = mutated_strat_def["params"]["rsi_oversold"] + 1
                             mutated_strat_def["params"]["rsi_overbought"] = min(mutated_strat_def["params"]["rsi_overbought"], new_strat_spec["params_spec"]["rsi_overbought"]["max"])
                             if mutated_strat_def["params"]["rsi_oversold"] >= mutated_strat_def["params"]["rsi_overbought"]:
                                 mutated_strat_def["params"]["rsi_oversold"] = new_strat_spec["params_spec"]["rsi_oversold"]["min"]
                                 mutated_strat_def["params"]["rsi_overbought"] = new_strat_spec["params_spec"]["rsi_overbought"]["min"]

        # 2b. Mutate parameters of the (possibly new) strategy
        else: # Only mutate params if strategy ID itself wasn't changed
            current_strat_spec = next((spec for spec in base_strategies_details if spec["id"] == mutated_strat_def["id"]), None)
            if current_strat_spec:
                new_params = {}
                for p_name, p_val in mutated_strat_def["params"].items():
                    if random.random() < mutation_rate:
                        p_spec_detail = current_strat_spec["params_spec"].get(p_name)
                        if p_spec_detail:
                            mutated_val = p_val
                            if p_spec_detail["type"] == "int":
                                # Add/subtract 1 or a small percentage, then clamp
                                change = random.choice([-2, -1, 1, 2]) # Example small integer change
                                mutated_val = p_val + change
                            elif p_spec_detail["type"] == "float":
                                change_percent = random.uniform(-0.1, 0.1) # +/- 10%
                                mutated_val = p_val * (1 + change_percent)

                            min_val = p_spec_detail.get("min")
                            max_val = p_spec_detail.get("max")
                            if min_val is not None:
                                mutated_val = max(min_val, mutated_val)
                            if max_val is not None:
                                mutated_val = min(max_val, mutated_val)

                            if p_spec_detail["type"] == "int":
                                mutated_val = round(mutated_val)
                            new_params[p_name] = mutated_val
                        else:
                            new_params[p_name] = p_val # Keep if no spec detail (should not happen)
                    else:
                        new_params[p_name] = p_val # No mutation for this param
                mutated_strat_def["params"] = new_params
                # Re-apply constraints after parameter mutation
                if mutated_strat_def["id"] == "moving_average_crossover":
                        if mutated_strat_def["params"]["short_window"] >= mutated_strat_def["params"]["long_window"]:
                            mutated_strat_def["params"]["short_window"] = min(mutated_strat_def["params"]["short_window"], mutated_strat_def["params"]["long_window"] -1)
                            if current_strat_spec : # Check if spec is available
                                mutated_strat_def["params"]["short_window"] = max(mutated_strat_def["params"]["short_window"], current_strat_spec["params_spec"]["short_window"]["min"])
                                if mutated_strat_def["params"]["short_window"] >= mutated_strat_def["params"]["long_window"]: # Final fallback
                                     mutated_strat_def["params"]["short_window"] = current_strat_spec["params_spec"]["short_window"]["min"]
                                     mutated_strat_def["params"]["long_window"] = current_strat_spec["params_spec"]["long_window"]["min"]


                elif mutated_strat_def["id"] == "rsi":
                    if mutated_strat_def["params"]["rsi_oversold"] >= mutated_strat_def["params"]["rsi_overbought"]:
                        mutated_strat_def["params"]["rsi_oversold"] = min(mutated_strat_def["params"]["rsi_oversold"], mutated_strat_def["params"]["rsi_overbought"]-1)
                        if current_strat_spec: # Check if spec is available
                            mutated_strat_def["params"]["rsi_oversold"] = max(mutated_strat_def["params"]["rsi_oversold"], current_strat_spec["params_spec"]["rsi_oversold"]["min"])
                            if mutated_strat_def["params"]["rsi_oversold"] >= mutated_strat_def["params"]["rsi_overbought"]: # Final fallback
                                 mutated_strat_def["params"]["rsi_oversold"] = current_strat_spec["params_spec"]["rsi_oversold"]["min"]
                                 mutated_strat_def["params"]["rsi_overbought"] = current_strat_spec["params_spec"]["rsi_overbought"]["min"]

        new_definitions.append(mutated_strat_def)
    mutated_individual["definitions"] = new_definitions

    # Ensure logic is None if only one strategy after mutation
    if len(mutated_individual["definitions"]) <= 1:
        mutated_individual["logic"] = None
    # Ensure logic is not None if multiple strategies (unless it was intentionally set by logic_change_rate to None which is not allowed for >1)
    elif mutated_individual["logic"] is None and len(mutated_individual["definitions"]) > 1:
            mutated_individual["logic"] = random.choice([l for l in AVAILABLE_LOGICS if l is not None])

    return mutated_individual

# --- Main GA Orchestration ---

# Define GA default parameters at a more accessible scope or pass them around
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_STRATEGY_CHANGE_RATE = 0.05
DEFAULT_LOGIC_CHANGE_RATE = 0.05
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_NUM_ELITES = 2

def run_ga_optimization(asset_id, asset_type, initial_capital_str, days_str, trade_quantity_str,
                        pop_size=20, generations=10, trade_style="default",
                        get_crypto_history_func=None,
                        # Allow GA-specific params to be overridden if needed, otherwise use defaults
                        mutation_rate=DEFAULT_MUTATION_RATE,
                        strategy_change_rate=DEFAULT_STRATEGY_CHANGE_RATE,
                        logic_change_rate=DEFAULT_LOGIC_CHANGE_RATE,
                        tournament_size=DEFAULT_TOURNAMENT_SIZE,
                        num_elites=DEFAULT_NUM_ELITES
                        ):

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
    best_backtest_results = None

    # Ensure num_elites is not greater than population size
    actual_num_elites = min(num_elites, pop_size)
    if actual_num_elites < 0: actual_num_elites = 0


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

        # Sort population by fitness (descending) to easily find best and elites
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Update overall best if a new best is found in the current sorted population
        if population_with_fitness and population_with_fitness[0][1] > best_overall_fitness:
            best_overall_fitness = population_with_fitness[0][1]
            best_overall_individual_config = copy.deepcopy(population_with_fitness[0][0])

            # Re-run backtest for this new overall best to get full results
            config_for_best_backtest = copy.deepcopy(best_overall_individual_config)
            for definition in config_for_best_backtest["definitions"]:
                if definition["id"] in STRATEGY_FUNCTION_MAP:
                    definition["function"] = STRATEGY_FUNCTION_MAP[definition["id"]]

            best_backtest_results = run_backtest(
                strategy_config=config_for_best_backtest,
                historical_data=historical_data,
                initial_capital=initial_capital,
                trade_quantity=trade_quantity,
                trade_style=trade_style
            )

        next_population = []

        # Elitism: Carry over top individuals
        for i in range(min(actual_num_elites, len(population_with_fitness))):
            next_population.append(copy.deepcopy(population_with_fitness[i][0]))

        # Parent Selection for the rest of the population
        num_offspring_needed = pop_size - len(next_population)
        # Ensure num_parents is reasonable, e.g., at least 2 if offspring are needed, and not more than population
        num_parents_to_select = min(max(2, num_offspring_needed), len(population_with_fitness))

        selected_parents_configs = []
        if num_offspring_needed > 0 and len(population_with_fitness) > 0 :
            selected_parents_configs = selection(
                population_with_fitness,
                num_parents=num_parents_to_select,
                tournament_size=tournament_size
            )

        if not selected_parents_configs and num_offspring_needed > 0:
            # Fallback: if selection returns no parents (e.g., pop too small for tournament),
            # fill with random new individuals or re-populate from current bests.
            # For now, if selection fails to provide parents, we might end up with a smaller next_pop or rely on elites.
            # Or, simply take top N from population_with_fitness as parents configs
            selected_parents_configs = [p[0] for p in population_with_fitness[:num_parents_to_select]]


        # Offspring Generation (Crossover & Mutation)
        while len(next_population) < pop_size:
            if not selected_parents_configs: # Not enough parents to breed
                # Fill remaining spots with new random individuals
                next_population.append(generate_random_individual(BASE_STRATEGIES_DETAILS, AVAILABLE_LOGICS))
                continue

            parent1_config = random.choice(selected_parents_configs)
            parent2_config = random.choice(selected_parents_configs) if len(selected_parents_configs) > 1 else parent1_config

            offspring_config = crossover(parent1_config, parent2_config, BASE_STRATEGIES_DETAILS)
            mutated_offspring_config = mutation(
                offspring_config,
                BASE_STRATEGIES_DETAILS,
                mutation_rate=mutation_rate,
                strategy_change_rate=strategy_change_rate,
                logic_change_rate=logic_change_rate
            )
            next_population.append(mutated_offspring_config)

        population = next_population[:pop_size] # Ensure correct population size

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
