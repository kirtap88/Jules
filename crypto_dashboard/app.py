from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from decimal import Decimal, ROUND_DOWN

# Imports for strategies and backtesting
from .strategies import moving_average_crossover_signal, rsi_signal, bollinger_bands_signal
from .backtesting import run_backtest # run_backtest will now take trade_style
from .stock_data_provider import get_stock_history, get_current_stock_price as get_current_stock_price_from_provider, search_stocks
from .genetic_optimizer import run_ga_optimization # Import for GA

load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crypto_dashboard.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    virtual_balance = db.Column(db.Float, nullable=False, default=100000.00)
    portfolios = db.relationship('Portfolio', backref='owner', lazy=True)
    trades = db.relationship('Trade', backref='trader', lazy=True)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)
    def __repr__(self): return f'<User {self.username}>'

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_symbol = db.Column(db.String(50), nullable=False)
    asset_type = db.Column(db.String(20), nullable=False, default='crypto')
    exchange = db.Column(db.String(50), nullable=True)
    quantity_str = db.Column(db.String(50), nullable=False, default='0.0')
    @property
    def quantity(self): return Decimal(self.quantity_str)
    @quantity.setter
    def quantity(self, value): self.quantity_str = str(Decimal(value).quantize(Decimal('0.00000001')))
    def __repr__(self): return f'<Portfolio user_id={self.user_id} asset_symbol={self.asset_symbol} asset_type={self.asset_type} qty={self.quantity_str}>'

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_symbol = db.Column(db.String(50), nullable=False)
    asset_type = db.Column(db.String(20), nullable=False, default='crypto')
    exchange = db.Column(db.String(50), nullable=True)
    trade_type = db.Column(db.String(10), nullable=False)
    quantity_str = db.Column(db.String(50), nullable=False)
    price_at_trade_str = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    strategy = db.Column(db.String(100), nullable=True)
    @property
    def quantity(self): return Decimal(self.quantity_str)
    @quantity.setter
    def quantity(self, value): self.quantity_str = str(Decimal(value).quantize(Decimal('0.00000001')))
    @property
    def price_at_trade(self): return Decimal(self.price_at_trade_str)
    @price_at_trade.setter
    def price_at_trade(self, value): self.price_at_trade_str = str(Decimal(value).quantize(Decimal('0.00000001')))
    def __repr__(self): return f'<Trade user_id={self.user_id} type={self.trade_type} asset_symbol={self.asset_symbol} asset_type={self.asset_type} qty={self.quantity_str}>'

# --- CoinGecko & Stock Data Provider Integration ---
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
def get_top_cryptos(limit=50):
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1, "sparkline": "false"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top cryptos: {e}")
        return None

def get_crypto_history(crypto_id, days="30", interval="daily"):
    url = f"{COINGECKO_API_URL}/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": interval}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return {"prices": data.get("prices", [])}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching crypto history for {crypto_id}: {e}")
        return None

def get_current_price(crypto_id): # For crypto
    url = f"{COINGECKO_API_URL}/simple/price"
    params = {"ids": crypto_id, "vs_currencies": "usd"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if crypto_id in data and "usd" in data[crypto_id]:
            return Decimal(str(data[crypto_id]["usd"]))
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current crypto price for {crypto_id}: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    return render_template("index.html")

# --- API Routes ---
@app.route('/api/assets')
def api_assets():
    combined_assets = []
    cryptos = get_top_cryptos(limit=50)
    if cryptos:
        for crypto in cryptos:
            combined_assets.append({
                "id": crypto.get("id"), "symbol": crypto.get("symbol", "").upper(),
                "name": crypto.get("name"), "current_price": crypto.get("current_price"),
                "asset_type": "crypto", "exchange": "Various Crypto Exchanges"
            })
    sample_stocks = search_stocks(query="")
    if sample_stocks:
        for stock_info in sample_stocks:
            price = get_current_stock_price_from_provider(stock_info['symbol'])
            combined_assets.append({
                "id": stock_info.get("symbol"), "symbol": stock_info.get("symbol"),
                "name": stock_info.get("name"), "current_price": str(price) if price is not None else None,
                "asset_type": "stock", "exchange": stock_info.get("exchange", "N/A")
            })
    if combined_assets: return jsonify(combined_assets)
    return jsonify({"error": "Could not fetch any asset data"}), 500

@app.route('/api/history/<string:asset_id>')
def api_asset_history(asset_id):
    days = request.args.get('days', '30')
    asset_type = request.args.get('asset_type', 'crypto')
    history_data = None
    if asset_type == 'stock':
        period_str = f"{days}d"
        raw_history = get_stock_history(symbol=asset_id, period=period_str)
        if raw_history: history_data = {"prices": raw_history}
    elif asset_type == 'crypto':
        history_data = get_crypto_history(crypto_id=asset_id, days=days)
    else:
        return jsonify({"error": f"Invalid asset_type: {asset_type}"}), 400
    if history_data and history_data.get("prices"): return jsonify(history_data)
    return jsonify({"error": f"Could not fetch history for {asset_id} (type: {asset_type})"}), 404

@app.route('/api/strategies', methods=['GET'])
def api_list_strategies():
    strategies = [
        {
            "id": "moving_average_crossover", "name": "Moving Average Crossover",
            "params": [
                {"name": "short_window", "default": 10, "type": "int", "label": "Short Window"},
                {"name": "long_window", "default": 30, "type": "int", "label": "Long Window"}
            ]
        },
        {
            "id": "rsi", "name": "Relative Strength Index (RSI)",
            "params": [
                {"name": "rsi_period", "default": 14, "type": "int", "label": "RSI Period"},
                {"name": "rsi_overbought", "default": 70, "type": "int", "label": "RSI Overbought"},
                {"name": "rsi_oversold", "default": 30, "type": "int", "label": "RSI Oversold"}
            ]
        },
        {
            "id": "bollinger_bands", "name": "Bollinger Bands",
            "params": [
                {"name": "period", "default": 20, "type": "int", "label": "Period"},
                {"name": "std_devs", "default": 2, "type": "float", "label": "Standard Deviations"}
            ]
        }
    ]
    return jsonify(strategies)

# --- Helper for parsing strategy parameters ---
def _get_typed_param(params_dict, param_name, expected_type_name, type_converter, condition=None, error_msg=None):
    if param_name not in params_dict or params_dict[param_name] is None:
        return None
    val_str = str(params_dict[param_name])
    parsed_val = None
    try:
        parsed_val = type_converter(val_str)
    except (ValueError, TypeError): # Error during type conversion
        raise ValueError(f"Parameter '{param_name}' ('{params_dict[param_name]}') must be a valid {expected_type_name}.")

    # If type conversion succeeded, check the condition
    if condition and not condition(parsed_val):
        # Condition failed, raise ValueError with the specific error message if provided
        raise ValueError(error_msg or f"Parameter '{param_name}' ('{parsed_val}') failed condition.")
    return parsed_val

def _parse_and_validate_strategy_params(strategy_id: str, raw_params: dict) -> dict:
    validated_params = {}
    if strategy_id == "moving_average_crossover":
        short_window = _get_typed_param(raw_params, "short_window", "integer", int, lambda x: x > 0, "short_window must be positive")
        long_window = _get_typed_param(raw_params, "long_window", "integer", int, lambda x: x > 0, "long_window must be positive")
        if short_window is not None: validated_params["short_window"] = short_window
        if long_window is not None: validated_params["long_window"] = long_window
        if validated_params.get("short_window") is not None and validated_params.get("long_window") is not None and \
           validated_params["short_window"] >= validated_params["long_window"]:
            raise ValueError("Moving Average Crossover: short_window must be less than long_window.")
    elif strategy_id == "rsi":
        rsi_period = _get_typed_param(raw_params, "rsi_period", "integer", int, lambda x: x > 0, "RSI period must be positive.")
        rsi_overbought = _get_typed_param(raw_params, "rsi_overbought", "integer", int, lambda x: 0 < x < 100, "RSI overbought must be between 0 and 100.")
        rsi_oversold = _get_typed_param(raw_params, "rsi_oversold", "integer", int, lambda x: 0 < x < 100, "RSI oversold must be between 0 and 100.")
        if rsi_period is not None: validated_params["rsi_period"] = rsi_period
        if rsi_overbought is not None: validated_params["rsi_overbought"] = rsi_overbought
        if rsi_oversold is not None: validated_params["rsi_oversold"] = rsi_oversold
        if validated_params.get("rsi_overbought") is not None and validated_params.get("rsi_oversold") is not None and \
           validated_params["rsi_oversold"] >= validated_params["rsi_overbought"]:
            raise ValueError("RSI: rsi_oversold must be less than rsi_overbought.")
    elif strategy_id == "bollinger_bands":
        period = _get_typed_param(raw_params, "period", "integer", int, lambda x: x > 0, "Bollinger Bands period must be positive.")
        std_devs = _get_typed_param(raw_params, "std_devs", "number", float, lambda x: x > 0, "Bollinger Bands std_devs must be positive.")
        if period is not None: validated_params["period"] = period
        if std_devs is not None: validated_params["std_devs"] = std_devs
    else:
        raise ValueError(f"Parameter parsing not implemented for unknown strategy: {strategy_id}")
    return validated_params

@app.route('/api/backtest', methods=['POST'])
def api_run_backtest():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid JSON payload"}), 400

    asset_id = data.get('asset_id')
    asset_type = data.get('asset_type', 'crypto')
    initial_capital_str = data.get('initial_capital')
    days_str = data.get('days')
    trade_quantity_str = data.get('trade_quantity', "1.0")
    trade_style = data.get('trade_style', "default") # New field

    combination_logic_req = data.get("combination_logic")
    strategy_payloads = data.get("strategies")

    # --- Basic Input Validation ---
    required_fields = {"asset_id": asset_id, "initial_capital": initial_capital_str, "days": days_str, "strategies": strategy_payloads}
    missing = [key for key, value in required_fields.items() if value is None]
    if missing: return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    if not isinstance(strategy_payloads, list) or not strategy_payloads:
        return jsonify({"error": "'strategies' must be a non-empty list."}), 400

    actual_combination_logic = combination_logic_req
    if len(strategy_payloads) == 1:
        if combination_logic_req and combination_logic_req.upper() != "NONE":
             pass
        actual_combination_logic = None
    elif len(strategy_payloads) > 1:
        if not combination_logic_req:
            return jsonify({"error": "Multiple strategies provided but no combination_logic specified."}), 400
        if combination_logic_req.upper() not in ["AND", "OR", "MAJORITY_VOTE"]:
            return jsonify({"error": f"Invalid combination_logic: {combination_logic_req}. Must be 'AND', 'OR', or 'MAJORITY_VOTE'."}), 400
        actual_combination_logic = combination_logic_req.upper()

    # Validate trade_style (allow case-insensitivity for robustness)
    valid_trade_styles = ["default", "day_trader", "swing_trader", "long_term_investor"]
    trade_style_to_pass = trade_style.lower() if trade_style else "default"
    if trade_style_to_pass not in valid_trade_styles:
        return jsonify({"error": f"Invalid trade_style: {trade_style}. Must be one of {valid_trade_styles}."}), 400

    try:
        initial_capital = Decimal(str(initial_capital_str))
        if initial_capital <= Decimal('0'): raise ValueError("Initial capital must be positive.")
        days = int(days_str)
        if days <= 0: raise ValueError("Days must be a positive integer.")
        trade_quantity = Decimal(str(trade_quantity_str))
        if trade_quantity <= Decimal('0'): raise ValueError("Trade quantity must be positive.")
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e: return jsonify({"error": f"Invalid input for capital, days, or quantity: {e}"}), 400

    historical_data_points = None
    if asset_type == 'stock':
        raw_hist = get_stock_history(symbol=asset_id, period=f"{days}d")
        if raw_hist: historical_data_points = raw_hist
    elif asset_type == 'crypto':
        crypto_hist = get_crypto_history(crypto_id=asset_id, days=str(days))
        if crypto_hist and crypto_hist.get("prices"): historical_data_points = crypto_hist["prices"]
    else:
        return jsonify({"error": f"Invalid asset_type for backtest: {asset_type}"}), 400
    if not historical_data_points:
        return jsonify({"error": f"Could not fetch historical data for {asset_id} (type: {asset_type})."}), 404

    parsed_strategy_definitions = []
    strategy_function_map = {
        "moving_average_crossover": moving_average_crossover_signal,
        "rsi": rsi_signal,
        "bollinger_bands": bollinger_bands_signal
    }
    try:
        for i, strat_payload in enumerate(strategy_payloads):
            strategy_id_from_payload = strat_payload.get("id")
            raw_params = strat_payload.get("params", {})
            if not strategy_id_from_payload or strategy_id_from_payload not in strategy_function_map:
                return jsonify({"error": f"Strategy at index {i} has invalid/missing id: '{strategy_id_from_payload}'"}), 400

            func_ref = strategy_function_map[strategy_id_from_payload]
            valid_params = _parse_and_validate_strategy_params(strategy_id_from_payload, raw_params)
            parsed_strategy_definitions.append({"id": strategy_id_from_payload, "function": func_ref, "params": valid_params})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    strategy_config = {"logic": actual_combination_logic, "definitions": parsed_strategy_definitions}

    try:
        results = run_backtest(
            strategy_config=strategy_config,
            historical_data=historical_data_points,
            initial_capital=initial_capital,
            trade_quantity=trade_quantity,
            trade_style=trade_style_to_pass
        )
    except Exception as e:
        app.logger.error(f"Error during run_backtest: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during backtest execution: {str(e)}"}), 500

    def serialize_decimal_fields(data_item):
        if isinstance(data_item, Decimal): return str(data_item)
        if isinstance(data_item, list): return [serialize_decimal_fields(item) for item in data_item]
        if isinstance(data_item, dict): return {k: serialize_decimal_fields(v) for k, v in data_item.items()}
        return data_item
    return jsonify(serialize_decimal_fields(results)), 200


@app.route('/api/optimize_strategy', methods=['POST'])
def api_optimize_strategy():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    asset_id = data.get('asset_id')
    asset_type = data.get('asset_type', 'crypto') # Default to crypto
    initial_capital_str = data.get('initial_capital')
    days_str = data.get('days')
    trade_quantity_str = data.get('trade_quantity', "1.0") # Default trade quantity
    trade_style = data.get('trade_style', "default") # Default trade style

    # GA specific parameters
    pop_size = data.get('pop_size', 10) # Default population size
    generations = data.get('generations', 5) # Default generations

    # --- Basic Input Validation ---
    required_fields = {
        "asset_id": asset_id,
        "initial_capital": initial_capital_str,
        "days": days_str
    }
    missing = [key for key, value in required_fields.items() if value is None]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    if asset_type not in ['crypto', 'stock']:
        return jsonify({"error": f"Invalid asset_type: {asset_type}. Must be 'crypto' or 'stock'."}), 400

    valid_trade_styles = ["default", "day_trader", "swing_trader", "long_term_investor"]
    if trade_style.lower() not in valid_trade_styles:
        return jsonify({"error": f"Invalid trade_style: {trade_style}. Must be one of {valid_trade_styles}."}), 400


    # --- Type Conversion and Further Validation for GA Params ---
    try:
        # initial_capital, days, trade_quantity will be converted inside run_ga_optimization
        # Validate pop_size and generations here as they are direct inputs to the optimizer route
        pop_size = int(pop_size)
        generations = int(generations)
        if pop_size <= 1 or generations <= 0:
            raise ValueError("Population size must be greater than 1 and generations must be positive.")
        if pop_size > 50 or generations > 20: # Safety limits for now
             return jsonify({"error": "Population size or generations exceed maximum limits (50/20 respectively)."}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input for pop_size or generations: {str(e)}"}), 400

    # Call the GA optimization function
    # Crucially, pass the app's get_crypto_history function to the optimizer
    optimization_results = run_ga_optimization(
        asset_id=asset_id,
        asset_type=asset_type,
        initial_capital_str=initial_capital_str,
        days_str=days_str,
        trade_quantity_str=trade_quantity_str,
        pop_size=pop_size,
        generations=generations,
        trade_style=trade_style.lower(),
        get_crypto_history_func=get_crypto_history # Pass the function from app.py
    )

    if optimization_results.get("error"):
        return jsonify(optimization_results), 400 # Or 500 if it's an internal server error

    # Serialize Decimal fields in the results before returning
    # The recursive serializer function is already defined above for /api/backtest
    # Let's define it locally or make it accessible if it's not in the same scope
    # For now, assuming serialize_decimal_fields is accessible or redefined if needed.
    # Re-defining for clarity and safety:
    def serialize_ga_results(data_item):
        if isinstance(data_item, Decimal): return str(data_item)
        if isinstance(data_item, list): return [serialize_ga_results(item) for item in data_item]
        if isinstance(data_item, dict): return {k: serialize_ga_results(v) for k, v in data_item.items()}
        return data_item

    return jsonify(serialize_ga_results(optimization_results)), 200


# --- Auth API Routes ---
# ... (Auth routes remain unchanged) ...
@app.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    # ... (existing code, ensure user.virtual_balance is handled as float/Decimal appropriately) ...
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({"error": "Username and password are required"}), 400
    username = data['username']
    password = data['password']
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
    new_user = User(username=username, virtual_balance=100000.00) # Ensure float
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "user": {"id": new_user.id, "username": new_user.username, "virtual_balance": float(new_user.virtual_balance) }}), 201

@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    # ... (existing code, ensure user.virtual_balance is handled as float/Decimal appropriately) ...
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({"error": "Username and password are required"}), 400
    username = data['username']
    password = data['password']
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return jsonify({
            "message": "Login successful",
            "user": {"id": user.id, "username": user.username, "virtual_balance": float(user.virtual_balance) }
        }), 200
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/api/auth/logout', methods=['POST'])
def api_auth_logout():
    # ... (existing code) ...
    return jsonify({"message": "Logout successful (simulated)"}), 200

# --- Trading API Routes ---
# ... (Trading routes remain unchanged from previous state, using asset_id, asset_type etc.) ...
@app.route('/api/trade/buy', methods=['POST'])
def api_trade_buy():
    data = request.get_json()
    if not data or 'user_id' not in data or 'asset_id' not in data or 'quantity' not in data or 'asset_type' not in data:
        return jsonify({"error": "Missing user_id, asset_id, asset_type, or quantity"}), 400
    try:
        user_id = int(data['user_id'])
        asset_id = data['asset_id']
        quantity_to_buy = Decimal(str(data['quantity']))
        asset_type = data.get('asset_type').lower()
        exchange = data.get('exchange')
    except ValueError:
        return jsonify({"error": "Invalid input type for user_id or quantity"}), 400
    if quantity_to_buy <= Decimal('0'): return jsonify({"error": "Quantity must be positive"}), 400
    user = User.query.get(user_id)
    if not user: return jsonify({"error": "User not found"}), 404
    current_price = None
    if asset_type == 'stock': current_price = get_current_stock_price_from_provider(asset_id)
    elif asset_type == 'crypto': current_price = get_current_price(asset_id.lower())
    else: return jsonify({"error": f"Unsupported asset_type: {asset_type}"}), 400
    if not current_price: return jsonify({"error": f"Could not fetch current price for {asset_id} (type: {asset_type})"}), 500
    total_cost = quantity_to_buy * current_price
    user_balance_decimal = Decimal(str(user.virtual_balance))
    if user_balance_decimal < total_cost: return jsonify({"error": "Insufficient funds"}), 400
    user.virtual_balance = float(user_balance_decimal - total_cost)
    db_asset_symbol = asset_id if asset_type == 'stock' else asset_id.lower()
    portfolio_item = Portfolio.query.filter_by(user_id=user_id, asset_symbol=db_asset_symbol, asset_type=asset_type).first()
    if portfolio_item: portfolio_item.quantity += quantity_to_buy
    else:
        portfolio_item = Portfolio(user_id=user_id, asset_symbol=db_asset_symbol, asset_type=asset_type, exchange=exchange)
        portfolio_item.quantity = quantity_to_buy
        db.session.add(portfolio_item)
    trade = Trade(user_id=user_id, asset_symbol=db_asset_symbol, asset_type=asset_type, exchange=exchange, trade_type='buy', quantity=quantity_to_buy, price_at_trade=current_price)
    if 'strategy' in data: trade.strategy = data['strategy']
    db.session.add(trade)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    return jsonify({
        "message": "Buy order successful", "asset_id": db_asset_symbol, "asset_type": asset_type,
        "quantity_bought": str(quantity_to_buy), "price_paid_per_unit": str(current_price),
        "total_cost": str(total_cost), "new_virtual_balance": float(user.virtual_balance),
        "new_portfolio_quantity": str(portfolio_item.quantity)
    }), 200

@app.route('/api/trade/sell', methods=['POST'])
def api_trade_sell():
    data = request.get_json()
    if not data or 'user_id' not in data or 'asset_id' not in data or 'quantity' not in data or 'asset_type' not in data:
        return jsonify({"error": "Missing user_id, asset_id, asset_type, or quantity"}), 400
    try:
        user_id = int(data['user_id'])
        asset_id = data['asset_id']
        quantity_to_sell = Decimal(str(data['quantity']))
        asset_type = data.get('asset_type').lower()
    except ValueError: return jsonify({"error": "Invalid input type for user_id or quantity"}), 400
    if quantity_to_sell <= Decimal('0'): return jsonify({"error": "Quantity must be positive"}), 400
    user = User.query.get(user_id)
    if not user: return jsonify({"error": "User not found"}), 404
    db_asset_symbol = asset_id if asset_type == 'stock' else asset_id.lower()
    portfolio_item = Portfolio.query.filter_by(user_id=user_id, asset_symbol=db_asset_symbol, asset_type=asset_type).first()
    if not portfolio_item or portfolio_item.quantity < quantity_to_sell:
        return jsonify({"error": f"Insufficient {db_asset_symbol} ({asset_type}) to sell or not owned"}), 400
    current_price = None
    if asset_type == 'stock': current_price = get_current_stock_price_from_provider(asset_id)
    elif asset_type == 'crypto': current_price = get_current_price(asset_id.lower())
    else: return jsonify({"error": f"Unsupported asset_type: {asset_type}"}), 400
    if not current_price: return jsonify({"error": f"Could not fetch current price for {asset_id} (type: {asset_type})"}), 500
    total_proceeds = quantity_to_sell * current_price
    portfolio_item.quantity -= quantity_to_sell
    user_balance_decimal = Decimal(str(user.virtual_balance))
    user.virtual_balance = float(user_balance_decimal + total_proceeds)
    trade = Trade(user_id=user_id, asset_symbol=db_asset_symbol, asset_type=asset_type, exchange=portfolio_item.exchange, trade_type='sell', quantity=quantity_to_sell, price_at_trade=current_price)
    if 'strategy' in data: trade.strategy = data['strategy']
    db.session.add(trade)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    return jsonify({
        "message": "Sell order successful", "asset_id": db_asset_symbol, "asset_type": asset_type,
        "quantity_sold": str(quantity_to_sell), "price_received_per_unit": str(current_price),
        "total_proceeds": str(total_proceeds), "new_virtual_balance": float(user.virtual_balance),
        "new_portfolio_quantity": str(portfolio_item.quantity)
    }), 200

# --- Helper for DB creation ---
def create_tables():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    # Determine the path to the .env file relative to app.py's location
    app_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(app_dir, '.env')

    # Check if .env exists at the determined path
    if not os.path.exists(dotenv_path):
        print(f".env file not found at {dotenv_path}, creating one...")
        with open(dotenv_path, 'w') as f:
            f.write(f"SECRET_KEY='{os.urandom(24).hex()}'\n")
            f.write("DATABASE_URL='sqlite:///crypto_dashboard.db'\n")

        # Load the newly created .env file specifically
        load_dotenv(dotenv_path=dotenv_path)

        # Update app.config from the newly loaded environment variables
        # This is crucial because the initial app.config setup might have used defaults or system env vars
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', app.config['SECRET_KEY'])
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', app.config['SQLALCHEMY_DATABASE_URI'])

    # Ensure SQLALCHEMY_DATABASE_URI is not None before proceeding
    if app.config['SQLALCHEMY_DATABASE_URI'] is None:
        print("Error: SQLALCHEMY_DATABASE_URI is None after .env handling. Check .env content and loading.")
        # Fallback or exit if critical, for now, let it try and potentially fail if still None
        # This might happen if .env was created empty or DATABASE_URL was removed.
        # For robustness, one might set a hard default here if os.environ.get also returns None.
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crypto_dashboard.db' # Hard fallback

    db_path_part = app.config['SQLALCHEMY_DATABASE_URI'].split('sqlite:///')[-1]
    # db_dir might be empty if SQLALCHEMY_DATABASE_URI is a relative path like 'sqlite:///crypto_dashboard.db'
    # In that case, db_dir will be an empty string, and os.path.exists('') is False.
    # The database file would be created in the current working directory.
    # If SQLALCHEMY_DATABASE_URI is 'sqlite:///instance/crypto_dashboard.db', then db_dir is 'instance'.

    # Correctly determine db_dir relative to app_dir if the path isn't absolute
    if not os.path.isabs(db_path_part):
        # If db_path_part is like "instance/crypto_dashboard.db", make it relative to app_dir
        # If it's just "crypto_dashboard.db", it will also be relative to app_dir
        # This ensures instance folder is inside crypto_dashboard if specified that way.
        # However, Flask's instance folder is typically outside the app package, relative to app root.
        # The default 'sqlite:///crypto_dashboard.db' would place it in cwd.
        # If 'instance/...' is used, it implies an instance folder relative to app root.
        # The current logic for db_dir creation is fine if SQLALCHEMY_DATABASE_URI is well-defined.
        # Let's assume the default 'sqlite:///crypto_dashboard.db' or 'sqlite:///instance/crypto_dashboard.db'
        # If 'instance/...' is used, the instance folder should be at /app/instance.
        # The original db_dir logic:
        db_dir = os.path.dirname(db_path_part) # This is correct for 'instance/file.db' -> 'instance'
                                             # For 'file.db' -> ''
    else: # Absolute path
        db_dir = os.path.dirname(db_path_part)

    if db_dir and not os.path.exists(db_dir): # Only create if db_dir is not empty
        # If db_dir is relative (e.g. "instance"), create it relative to the app root (/app)
        # This matches Flask's typical instance folder behavior.
        if not os.path.isabs(db_dir):
            db_dir_abs_path = os.path.join(os.getcwd(), db_dir) # os.getcwd() is /app when run with -m
            if not os.path.exists(db_dir_abs_path):
                 os.makedirs(db_dir_abs_path, exist_ok=True)
        else: # Absolute path specified in DATABASE_URL
            os.makedirs(db_dir, exist_ok=True)

    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)
