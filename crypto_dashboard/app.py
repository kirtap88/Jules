from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from decimal import Decimal, ROUND_DOWN # For precise financial calculations

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
    virtual_balance = db.Column(db.Float, nullable=False, default=100000.00) # Store as float, handle precision in logic

    portfolios = db.relationship('Portfolio', backref='owner', lazy=True)
    trades = db.relationship('Trade', backref='trader', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    crypto_id = db.Column(db.String(50), nullable=False) # e.g., 'bitcoin', 'ethereum'
    # Using String for quantity to store as Decimal for precision, converting in logic
    quantity_str = db.Column(db.String(50), nullable=False, default='0.0')

    @property
    def quantity(self):
        return Decimal(self.quantity_str)

    @quantity.setter
    def quantity(self, value):
        self.quantity_str = str(Decimal(value).quantize(Decimal('0.00000001'))) # Example precision for crypto

    def __repr__(self):
        return f'<Portfolio user_id={self.user_id} crypto={self.crypto_id} qty={self.quantity_str}>'

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    crypto_id = db.Column(db.String(50), nullable=False)
    trade_type = db.Column(db.String(10), nullable=False) # 'buy' or 'sell'
    # Using String for quantity and price for Decimal precision
    quantity_str = db.Column(db.String(50), nullable=False)
    price_at_trade_str = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    @property
    def quantity(self):
        return Decimal(self.quantity_str)

    @quantity.setter
    def quantity(self, value):
        self.quantity_str = str(Decimal(value).quantize(Decimal('0.00000001')))

    @property
    def price_at_trade(self):
        return Decimal(self.price_at_trade_str)

    @price_at_trade.setter
    def price_at_trade(self, value):
        self.price_at_trade_str = str(Decimal(value).quantize(Decimal('0.00000001')))


    def __repr__(self):
        return f'<Trade user_id={self.user_id} type={self.trade_type} crypto={self.crypto_id} qty={self.quantity_str}>'

# --- CoinGecko API Integration ---
COINGECKO_API_URL = "https.api.coingecko.com/api/v3"

def get_top_cryptos(limit=10):
    # ... (existing code) ...
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
    # ... (existing code) ...
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

def get_current_price(crypto_id):
    # ... (existing code) ...
    url = f"{COINGECKO_API_URL}/simple/price"
    params = {"ids": crypto_id, "vs_currencies": "usd"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if crypto_id in data and "usd" in data[crypto_id]:
            return Decimal(str(data[crypto_id]["usd"])) # Return as Decimal
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current price for {crypto_id}: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    return render_template("index.html")

# --- API Routes ---
@app.route('/api/cryptos')
def api_top_cryptos():
    # ... (existing code) ...
    cryptos = get_top_cryptos()
    if cryptos:
        return jsonify(cryptos)
    return jsonify({"error": "Could not fetch cryptocurrency data"}), 500

@app.route('/api/history/<string:crypto_id>')
def api_crypto_history(crypto_id):
    # ... (existing code) ...
    days = request.args.get('days', '30')
    history = get_crypto_history(crypto_id, days=days)
    if history:
        return jsonify(history)
    return jsonify({"error": f"Could not fetch history for {crypto_id}"}), 500

# --- Auth API Routes ---
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
# NOTE: These endpoints should be protected and use authenticated user_id in a real app.
# For now, user_id is passed in the request for simplicity.

@app.route('/api/trade/buy', methods=['POST'])
def api_trade_buy():
    data = request.get_json()
    if not data or 'user_id' not in data or 'crypto_id' not in data or 'quantity' not in data:
        return jsonify({"error": "Missing user_id, crypto_id, or quantity"}), 400

    try:
        user_id = int(data['user_id'])
        crypto_id = data['crypto_id'].lower() # Ensure consistent casing
        quantity_to_buy = Decimal(str(data['quantity']))
    except ValueError:
        return jsonify({"error": "Invalid input type for user_id or quantity"}), 400

    if quantity_to_buy <= Decimal('0'):
        return jsonify({"error": "Quantity must be positive"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    current_price = get_current_price(crypto_id)
    if not current_price:
        return jsonify({"error": f"Could not fetch price for {crypto_id}"}), 500

    total_cost = quantity_to_buy * current_price
    user_balance_decimal = Decimal(str(user.virtual_balance))

    if user_balance_decimal < total_cost:
        return jsonify({"error": "Insufficient funds"}), 400

    # Update balance
    user.virtual_balance = float(user_balance_decimal - total_cost) # Store back as float

    # Update portfolio
    portfolio_item = Portfolio.query.filter_by(user_id=user_id, crypto_id=crypto_id).first()
    if portfolio_item:
        portfolio_item.quantity += quantity_to_buy
    else:
        portfolio_item = Portfolio(user_id=user_id, crypto_id=crypto_id)
        portfolio_item.quantity = quantity_to_buy # Setter handles Decimal conversion
        db.session.add(portfolio_item)

    # Record trade
    trade = Trade(user_id=user_id, crypto_id=crypto_id, trade_type='buy')
    trade.quantity = quantity_to_buy
    trade.price_at_trade = current_price
    db.session.add(trade)

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({
        "message": "Buy order successful",
        "crypto_id": crypto_id,
        "quantity_bought": str(quantity_to_buy),
        "price_paid_per_unit": str(current_price),
        "total_cost": str(total_cost),
        "new_virtual_balance": float(user.virtual_balance),
        "new_portfolio_quantity": str(portfolio_item.quantity)
    }), 200


@app.route('/api/trade/sell', methods=['POST'])
def api_trade_sell():
    data = request.get_json()
    if not data or 'user_id' not in data or 'crypto_id' not in data or 'quantity' not in data:
        return jsonify({"error": "Missing user_id, crypto_id, or quantity"}), 400

    try:
        user_id = int(data['user_id'])
        crypto_id = data['crypto_id'].lower()
        quantity_to_sell = Decimal(str(data['quantity']))
    except ValueError:
        return jsonify({"error": "Invalid input type for user_id or quantity"}), 400

    if quantity_to_sell <= Decimal('0'):
        return jsonify({"error": "Quantity must be positive"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    portfolio_item = Portfolio.query.filter_by(user_id=user_id, crypto_id=crypto_id).first()
    if not portfolio_item or portfolio_item.quantity < quantity_to_sell:
        return jsonify({"error": f"Insufficient {crypto_id} to sell or not owned"}), 400

    current_price = get_current_price(crypto_id)
    if not current_price:
        return jsonify({"error": f"Could not fetch price for {crypto_id}"}), 500

    total_proceeds = quantity_to_sell * current_price

    # Update portfolio
    portfolio_item.quantity -= quantity_to_sell
    # Optionally, remove portfolio item if quantity is zero, or keep it.
    # if portfolio_item.quantity == Decimal('0'):
    #     db.session.delete(portfolio_item)

    # Update balance
    user_balance_decimal = Decimal(str(user.virtual_balance))
    user.virtual_balance = float(user_balance_decimal + total_proceeds) # Store back as float

    # Record trade
    trade = Trade(user_id=user_id, crypto_id=crypto_id, trade_type='sell')
    trade.quantity = quantity_to_sell
    trade.price_at_trade = current_price
    db.session.add(trade)

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({
        "message": "Sell order successful",
        "crypto_id": crypto_id,
        "quantity_sold": str(quantity_to_sell),
        "price_received_per_unit": str(current_price),
        "total_proceeds": str(total_proceeds),
        "new_virtual_balance": float(user.virtual_balance),
        "new_portfolio_quantity": str(portfolio_item.quantity)
    }), 200


# --- Helper for DB creation ---
def create_tables():
    with app.app_context():
        db.create_all()
    # print("Database tables created (if they didn't exist).") # Less verbose

if __name__ == '__main__':
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(f"SECRET_KEY='{os.urandom(24).hex()}'\n")
            f.write("DATABASE_URL='sqlite:///crypto_dashboard.db'\n")
        # print("Created .env file with a new SECRET_KEY and default DATABASE_URL.")
        load_dotenv()
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')

    db_path_part = app.config['SQLALCHEMY_DATABASE_URI'].split('sqlite:///')[-1]
    db_dir = os.path.dirname(db_path_part)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    with app.app_context():
        db.create_all()
        # print("Database tables ensured.")

    app.run(debug=True, host='0.0.0.0', port=5001)

EOL
