"""
High-Performance Trade Simulator
--------------------------------
A real-time market data simulator that estimates transaction costs and market impact.
This system connects to WebSocket endpoints streaming L2 orderbook data from cryptocurrency exchanges.

Features:
- Real-time L2 orderbook processing
- Almgren-Chriss market impact model
- Linear regression for slippage estimation
- Logistic regression for maker/taker proportion prediction
- Performance optimized data processing

Author: Your Name
Date: May 14, 2025
"""
import sys
import os
import asyncio
import json
import logging
import time
import tkinter as tk
from tkinter import ttk, messagebox
import websocket
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    filename='trade_simulator.log', 
                    format='%(asctime)s %(levelname)s %(message)s')

# OKX Fee Structure (as of May 2025)
OKX_FEE_TIERS = {
    "VIP0": {"maker": 0.0008, "taker": 0.001},
    "VIP1": {"maker": 0.0006, "taker": 0.0008},
    "VIP2": {"maker": 0.0004, "taker": 0.0007},
    "VIP3": {"maker": 0.0002, "taker": 0.0006},
    "VIP4": {"maker": 0.0000, "taker": 0.0005},
    "VIP5": {"maker": -0.0001, "taker": 0.0004}
}

class AlmgrenChrissModel:
    """Implementation of the Almgren-Chriss market impact model."""
    
    def __init__(self, permanent_impact=0.1, temporary_impact=0.1):
        """
        Initialize the Almgren-Chriss model parameters.
        
        Parameters:
        - permanent_impact: Coefficient for permanent price impact
        - temporary_impact: Coefficient for temporary price impact
        """
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        
    def calculate_market_impact(self, quantity, volatility, avg_daily_volume, current_spread):
        """
        Calculate market impact according to Almgren-Chriss model.
        
        Parameters:
        - quantity: Order size in base currency
        - volatility: Market volatility (daily)
        - avg_daily_volume: Average daily trading volume
        - current_spread: Current bid-ask spread
        
        Returns:
        - market_impact: Estimated market impact cost
        """
        # Permanent impact component
        permanent = self.permanent_impact * volatility * np.sqrt(quantity / avg_daily_volume)
        
        # Temporary impact component (affected by order size and spread)
        temporary = self.temporary_impact * current_spread * np.power(quantity / avg_daily_volume, 0.6)
        
        # Total market impact
        market_impact = (permanent + temporary) * quantity
        
        return market_impact

class OrderbookProcessor:
    """Processes and analyzes orderbook data for trading simulations."""
    
    def __init__(self, max_history=100):
        """
        Initialize the orderbook processor.
        
        Parameters:
        - max_history: Maximum number of orderbook states to keep in history
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.latest = None
        
    def update(self, orderbook):
        """
        Update with new orderbook data.
        
        Parameters:
        - orderbook: Dictionary containing bids, asks and timestamp
        """
        self.latest = orderbook
        self.history.append(orderbook)
        
    def get_spread(self):
        """Calculate the current bid-ask spread."""
        if not self.latest or len(self.latest['asks']) == 0 or len(self.latest['bids']) == 0:
            return 0.0
            
        best_ask = float(self.latest['asks'][0][0])
        best_bid = float(self.latest['bids'][0][0])
        return best_ask - best_bid
        
    def get_spread_pct(self):
        """Calculate the current bid-ask spread as a percentage."""
        if not self.latest or len(self.latest['asks']) == 0 or len(self.latest['bids']) == 0:
            return 0.0
            
        best_ask = float(self.latest['asks'][0][0])
        best_bid = float(self.latest['bids'][0][0])
        mid_price = (best_ask + best_bid) / 2
        
        return (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
        
    def calculate_slippage(self, quantity, side="buy"):
        """
        Calculate expected slippage for a given order quantity.
        
        Parameters:
        - quantity: Order size in quote currency (e.g., USD)
        - side: Order side, either "buy" or "sell"
        
        Returns:
        - slippage: Estimated slippage amount
        """
        if not self.latest or len(self.latest['asks']) == 0 or len(self.latest['bids']) == 0:
            return 0.0
            
        if side == "buy":
            # For buy orders, we walk up the ask book
            orderbook_side = np.array(self.latest['asks'], dtype=float)
            base_price = float(orderbook_side[0][0])
        else:
            # For sell orders, we walk down the bid book
            orderbook_side = np.array(self.latest['bids'], dtype=float)
            base_price = float(orderbook_side[0][0])
            
        # Walk the book to calculate average execution price
        remaining = quantity
        total_cost = 0.0
        total_quantity = 0.0
        
        for price, size in orderbook_side:
            price = float(price)
            size = float(size)
            
            # Convert size to quote currency (e.g., USD)
            level_quote_size = price * size
            
            if remaining <= level_quote_size:
                # We can fill the remaining quantity at this level
                total_cost += remaining
                total_quantity += remaining / price
                break
            else:
                # We take all available at this level and continue
                total_cost += level_quote_size
                total_quantity += size
                remaining -= level_quote_size
                
        # If we couldn't fill the order completely
        if remaining > 0:
            # Use the last available price for the remaining quantity
            last_price = float(orderbook_side[-1][0])
            total_cost += remaining
            total_quantity += remaining / last_price
            
        # Calculate average execution price
        avg_price = total_cost / total_quantity if total_quantity > 0 else base_price
        
        # Calculate slippage
        if side == "buy":
            slippage = (avg_price - base_price) * total_quantity
        else:
            slippage = (base_price - avg_price) * total_quantity
            
        return slippage
        
    def estimate_avg_daily_volume(self):
        """
        Estimate average daily volume based on recent order book data.
        In a production system, this would use historical data.
        """
        # Mock value - in production, this would use historical data API
        # Based on ~$10B daily volume for BTC-USDT pair across exchanges
        return 10_000_000  
        
    def estimate_volatility(self, window=20):
        """
        Estimate current volatility based on recent mid-price changes.
        
        Parameters:
        - window: Number of recent orderbook states to use
        
        Returns:
        - volatility: Estimated annualized volatility
        """
        if len(self.history) < 2:
            return 0.02  # Default 2% if not enough history
            
        # Extract mid prices from history
        mid_prices = []
        for ob in list(self.history)[-window:]:
            if len(ob['asks']) > 0 and len(ob['bids']) > 0:
                best_ask = float(ob['asks'][0][0])
                best_bid = float(ob['bids'][0][0])
                mid_price = (best_ask + best_bid) / 2
                mid_prices.append(mid_price)
                
        if len(mid_prices) < 2:
            return 0.02
            
        # Calculate returns
        returns = np.diff(mid_prices) / mid_prices[:-1]
        
        # Calculate standard deviation of returns
        std_dev = np.std(returns)
        
        # Annualize (assuming 252 trading days, 24 hours, data every second)
        # Adjust based on actual data frequency
        samples_per_day = 24 * 60 * 60  # seconds in a day
        annualized_vol = std_dev * np.sqrt(samples_per_day)
        
        return max(0.005, min(annualized_vol, 0.5))  # Cap between 0.5% and 50%

class TradeSimulator:
    """High-performance trade simulator for cryptocurrency markets."""
    
    def __init__(self, root):
        """Initialize the trade simulator application."""
        self.root = root
        self.root.title("GoQuant Trade Simulator")
        self.root.geometry("1200x700")  # Larger window for better UI
        self.root.configure(bg="#f5f5f5")  # Light gray background
        
        # Create menubar
        self.create_menu()
        
        # Initialize data structures
        self.orderbook_processor = OrderbookProcessor(max_history=1000)
        self.impact_model = AlmgrenChrissModel()
        self.latency_history = deque(maxlen=100)  # Track recent latencies
        self.running = True
        
        # Initialize regression models
        self.models = self.initialize_models()
        
        # Initialize UI
        self.setup_ui()
        
        # Start WebSocket connection
        self.start_websocket()
        
        # Update timer for UI refresh (every 100ms)
        self.root.after(100, self.update_timer)

    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Performance Analysis", command=self.show_performance)
        tools_menu.add_command(label="Export Data", command=self.export_data)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def initialize_models(self):
        """Initialize statistical models for trading simulation."""
        # In production, these would be trained on historical data
        slippage_model = LinearRegression()
        maker_taker_model = LogisticRegression()
        
        # Dummy data for regression models (to be replaced with real training)
        X_dummy = np.array([[0.1, 100], [0.2, 200], [0.3, 300]]) # volatility, quantity
        y_slippage = np.array([0.5, 1.0, 1.5])  # slippage values
        y_maker = np.array([0, 1, 0])  # maker/taker (0=taker, 1=maker)
        
        # Train models on dummy data
        slippage_model.fit(X_dummy, y_slippage)
        maker_taker_model.fit(X_dummy, y_maker)
        
        return {
            'slippage': slippage_model,
            'maker_taker': maker_taker_model
        }

    def setup_ui(self):
        """Set up the user interface components."""
        # Create and configure frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with title and status
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(header_frame, text="GoQuant Trade Simulator", 
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = ttk.Label(header_frame, textvariable=self.status_var, 
                                     foreground="red")
        self.status_label.pack(side=tk.RIGHT)
        
        # Create content frame with input and output panels
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Input parameters
        left_frame = ttk.LabelFrame(content_frame, text="Input Parameters")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input parameters form
        form_frame = ttk.Frame(left_frame)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Exchange selection
        ttk.Label(form_frame, text="Exchange:").grid(row=0, column=0, sticky="w", pady=5)
        self.exchange_var = tk.StringVar(value="OKX")
        ttk.Combobox(form_frame, textvariable=self.exchange_var, 
                    values=["OKX"], 
                    state='readonly').grid(row=0, column=1, sticky="we", pady=5)
        
        # Asset selection
        ttk.Label(form_frame, text="Spot Asset:").grid(row=1, column=0, sticky="w", pady=5)
        self.asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        ttk.Combobox(form_frame, textvariable=self.asset_var, 
                    values=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"], 
                    state='readonly').grid(row=1, column=1, sticky="we", pady=5)
        
        # Order type
        ttk.Label(form_frame, text="Order Type:").grid(row=2, column=0, sticky="w", pady=5)
        self.order_type_var = tk.StringVar(value="market")
        ttk.Combobox(form_frame, textvariable=self.order_type_var, 
                    values=["market"], 
                    state='readonly').grid(row=2, column=1, sticky="we", pady=5)
        
        # Quantity
        ttk.Label(form_frame, text="Quantity (USD):").grid(row=3, column=0, sticky="w", pady=5)
        self.quantity_var = tk.StringVar(value="100")
        ttk.Entry(form_frame, textvariable=self.quantity_var).grid(row=3, column=1, sticky="we", pady=5)
        
        # Volatility
        ttk.Label(form_frame, text="Volatility:").grid(row=4, column=0, sticky="w", pady=5)
        self.volatility_var = tk.StringVar(value="0.02")  # 2%
        ttk.Entry(form_frame, textvariable=self.volatility_var).grid(row=4, column=1, sticky="we", pady=5)
        
        # Fee tier
        ttk.Label(form_frame, text="Fee Tier:").grid(row=5, column=0, sticky="w", pady=5)
        self.fee_tier_var = tk.StringVar(value="VIP0")
        ttk.Combobox(form_frame, textvariable=self.fee_tier_var, 
                    values=list(OKX_FEE_TIERS.keys()), 
                    state='readonly').grid(row=5, column=1, sticky="we", pady=5)
        
        # Button to recalculate manually
        ttk.Button(form_frame, text="Recalculate", 
                  command=self.manual_recalculate).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Right panel: Output parameters
        right_frame = ttk.LabelFrame(content_frame, text="Output Parameters")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output parameters display
        output_frame = ttk.Frame(right_frame)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create output labels with better formatting
        self.output_labels = {}
        outputs = [
            "Expected Slippage ($)", 
            "Expected Fees ($)", 
            "Expected Market Impact ($)", 
            "Net Cost ($)", 
            "Maker/Taker Proportion", 
            "Internal Latency (ms)"
        ]
        
        for i, output in enumerate(outputs):
            ttk.Label(output_frame, text=output + ":", 
                     font=("Arial", 10, "bold")).grid(row=i, column=0, sticky="w", pady=5)
            
            self.output_labels[output] = ttk.Label(output_frame, text="0", width=10)
            self.output_labels[output].grid(row=i, column=1, sticky="e", pady=5)
        
        # Add a section for orderbook visualization
        chart_frame = ttk.LabelFrame(main_frame, text="Orderbook Visualization")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 3), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize the plot
        self.initialize_plot()
        
        # Status bar at the bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        self.conn_status = tk.StringVar(value="No connection")
        ttk.Label(status_frame, textvariable=self.conn_status).pack(side=tk.LEFT)
        
        self.last_update = tk.StringVar(value="Last update: Never")
        ttk.Label(status_frame, textvariable=self.last_update).pack(side=tk.RIGHT)

    def initialize_plot(self):
        """Initialize the orderbook visualization plot."""
        self.ax.clear()
        self.ax.set_title("Order Book Depth")
        self.ax.set_xlabel("Price")
        self.ax.set_ylabel("Quantity")
        
        # Create empty plots for bids and asks
        self.bid_plot, = self.ax.plot([], [], 'g-', label='Bids')
        self.ask_plot, = self.ax.plot([], [], 'r-', label='Asks')
        
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

    def update_plot(self):
        """Update the orderbook visualization with current data."""
        if not self.orderbook_processor.latest:
            return
            
        try:
            # Extract orderbook data
            bids = np.array(self.orderbook_processor.latest['bids'], dtype=float)
            asks = np.array(self.orderbook_processor.latest['asks'], dtype=float)
            
            if len(bids) == 0 or len(asks) == 0:
                return
                
            # Sort by price
            bids = bids[bids[:, 0].argsort()]
            asks = asks[asks[:, 0].argsort()]
            
            # Calculate cumulative quantities for depth chart
            bid_prices = bids[:, 0]
            bid_sizes = bids[:, 1]
            bid_cum_sizes = np.cumsum(bid_sizes)
            
            ask_prices = asks[:, 0]
            ask_sizes = asks[:, 1]
            ask_cum_sizes = np.cumsum(ask_sizes)
            
            # Update the plots
            self.bid_plot.set_data(bid_prices, bid_cum_sizes)
            self.ask_plot.set_data(ask_prices, ask_cum_sizes)
            
            # Adjust axes limits
            min_price = np.min(bid_prices) * 0.999
            max_price = np.max(ask_prices) * 1.001
            max_size = max(np.max(bid_cum_sizes), np.max(ask_cum_sizes)) * 1.1
            
            self.ax.set_xlim(min_price, max_price)
            self.ax.set_ylim(0, max_size)
            
            # Draw the updated plot
            self.canvas.draw_idle()
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")

    def start_websocket(self):
        """Initialize and start the WebSocket connection."""
        self.ws = None
        self.connect_websocket()

    def connect_websocket(self):
        """Establish WebSocket connection to the orderbook stream."""
        try:
            # Update connection status
            self.conn_status.set("Connecting...")
            self.status_var.set("Connecting...")
            self.status_label.config(foreground="orange")
            
            # Initialize WebSocket
            self.ws = websocket.WebSocketApp(
                f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{self.asset_var.get()}",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run WebSocket in a separate thread
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
            
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self.conn_status.set(f"Connection failed: {str(e)}")
            self.status_var.set("Disconnected")
            self.status_label.config(foreground="red")
            self.schedule_reconnect()

    def on_open(self, ws):
        """Handle WebSocket connection open event."""
        logging.info("WebSocket connected")
        self.root.after(0, self.update_connection_status, "Connected", "green")

    def on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        
        Parameters:
        - ws: WebSocket connection
        - message: Incoming message data (JSON string)
        """
        start_time = time.perf_counter()
        
        try:
            # Parse message
            data = json.loads(message)
            
            # Process orderbook data
            orderbook = {
                'bids': data.get('bids', []),
                'asks': data.get('asks', []),
                'timestamp': data.get('timestamp', '')
            }
            
            # Update orderbook processor
            self.orderbook_processor.update(orderbook)
            
            # Process and update UI
            self.process_orderbook(start_time)
            
            # Update connection status
            self.root.after(0, self.update_last_received, data.get('timestamp', ''))
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}, message: {message[:100]}...")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket error events."""
        logging.error(f"WebSocket error: {error}")
        self.root.after(0, self.update_connection_status, f"Error: {str(error)}", "red")
        self.schedule_reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close events."""
        logging.info(f"WebSocket closed: {close_status_code}, {close_msg}")
        self.root.after(0, self.update_connection_status, "Disconnected", "red")
        
        if self.running:
            self.schedule_reconnect()

    def update_connection_status(self, status, color):
        """Update the connection status display in the UI."""
        self.conn_status.set(f"WebSocket: {status}")
        self.status_var.set(status)
        self.status_label.config(foreground=color)

    def update_last_received(self, timestamp):
        """Update the last received timestamp display in the UI."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%H:%M:%S.%f")[:-3]  # Show milliseconds
            self.last_update.set(f"Last update: {formatted_time}")
        except Exception:
            self.last_update.set(f"Last update: {timestamp}")

    def schedule_reconnect(self):
        """Schedule a reconnection attempt after delay."""
        self.root.after(5000, self.connect_websocket)

    def process_orderbook(self, start_time):
        """
        Process orderbook data to calculate trading metrics.
        
        Parameters:
        - start_time: Time when processing started (for latency calculation)
        """
        try:
            # Get input parameters
            quantity = float(self.quantity_var.get())
            volatility = float(self.volatility_var.get())
            fee_tier = self.fee_tier_var.get()
            
            # Calculate spread
            spread = self.orderbook_processor.get_spread()
            spread_pct = self.orderbook_processor.get_spread_pct()
            
            # Calculate slippage using orderbook depth
            slippage = self.orderbook_processor.calculate_slippage(quantity, "buy")
            
            # Get fee rates based on selected tier
            fee_rates = OKX_FEE_TIERS.get(fee_tier, OKX_FEE_TIERS["VIP0"])
            
            # Predict maker/taker proportion
            X_predict = np.array([[volatility, quantity]])
            maker_prob = self.models['maker_taker'].predict_proba(X_predict)[0][1] if hasattr(self.models['maker_taker'], 'predict_proba') else 0.3
            
            # Calculate fees based on maker/taker proportion
            maker_fee = fee_rates["maker"] * quantity * maker_prob
            taker_fee = fee_rates["taker"] * quantity * (1 - maker_prob)
            fees = maker_fee + taker_fee
            
            # Get daily volume estimate
            avg_daily_volume = self.orderbook_processor.estimate_avg_daily_volume()
            
            # Calculate market impact using Almgren-Chriss model
            market_impact = self.impact_model.calculate_market_impact(
                quantity, volatility, avg_daily_volume, spread)
                
            # Calculate net cost
            net_cost = slippage + fees + market_impact
            
            # Calculate processing latency in milliseconds
            latency = (time.perf_counter() - start_time) * 1000
            self.latency_history.append(latency)
            
            # Update UI on main thread
            self.root.after(0, self.update_ui, slippage, fees, market_impact, 
                           net_cost, maker_prob, latency)
                           
            # Update visualization
            self.root.after(0, self.update_plot)
            
        except Exception as e:
            logging.error(f"Error processing orderbook: {e}")

    def update_ui(self, slippage, fees, market_impact, net_cost, maker_proportion, latency):
        """
        Update the UI with calculated values.
        
        Parameters:
        - slippage: Calculated slippage value
        - fees: Trading fees
        - market_impact: Calculated market impact
        - net_cost: Total transaction cost
        - maker_proportion: Proportion of maker vs taker
        - latency: Processing latency in milliseconds
        """
        # Update output labels
        self.output_labels["Expected Slippage ($)"].config(text=f"{slippage:.2f}")
        self.output_labels["Expected Fees ($)"].config(text=f"{fees:.2f}")
        self.output_labels["Expected Market Impact ($)"].config(text=f"{market_impact:.2f}")
        self.output_labels["Net Cost ($)"].config(text=f"{net_cost:.2f}")
        self.output_labels["Maker/Taker Proportion"].config(text=f"{maker_proportion:.2f}")
        
        # Color the latency based on performance
        latency_label = self.output_labels["Internal Latency (ms)"]
        latency_label.config(text=f"{latency:.2f}")
        
        if latency < 5:  # Fast (green)
            latency_label.config(foreground="green")
        elif latency < 20:  # Acceptable (black)
            latency_label.config(foreground="black")
        else:  # Slow (red)
            latency_label.config(foreground="red")

    def update_timer(self):
        """Regular timer for UI updates and maintenance tasks."""
        if self.running:
            # Update the UI periodically even without new data
            self.update_plot()
            
            # Schedule the next update
            self.root.after(100, self.update_timer)

    def manual_recalculate(self):
        """Manually trigger recalculation of trading metrics."""
        if self.orderbook_processor.latest:
            self.process_orderbook(time.perf_counter())
            messagebox.showinfo("Recalculation", "Trading metrics have been recalculated.")
        else:
            messagebox.showwarning("No Data", "No orderbook data available. Please wait for data.")

    def save_config(self):
        """Save current configuration to file."""
        config = {
            "exchange": self.exchange_var.get(),
            "asset": self.asset_var.get(),
            "order_type": self.order_type_var.get(),
            "quantity": self.quantity_var.get(),
            "volatility": self.volatility_var.get(),
            "fee_tier": self.fee_tier_var.get()
        }
        
        try:
            with open("trade_simulator_config.json", "w") as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", "Configuration saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def load_config(self):
        """Load configuration from file."""
        try:
            with open("trade_simulator_config.json", "r") as f:
                config = json.load(f)
                
            # Update UI variables
            self.exchange_var.set(config.get("exchange", "OKX"))
            self.asset_var.set(config.get("asset", "BTC-USDT-SWAP"))
            self.order_type_var.set(config.get("order_type", "market"))
            self.quantity_var.set(config.get("quantity", "100"))
            self.volatility_var.set(config.get("volatility", "0.02"))
            self.fee_tier_var.set(config.get("fee_tier", "VIP0"))
            
            messagebox.showinfo("Success", "Configuration loaded successfully.")
        except FileNotFoundError:
            messagebox.showwarning("Not Found", "No saved configuration found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def show_performance(self):
        """Display performance analysis window."""
        performance_window = tk.Toplevel(self.root)
        performance_window.title("Performance Analysis")
        performance_window.geometry("600x400")
        
        # Create notebook with tabs
        notebook = ttk.Notebook(performance_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Latency analysis tab
        latency_frame = ttk.Frame(notebook)
        notebook.add(latency_frame, text="Latency Analysis")
        
        # Create latency histogram
        fig, ax = plt.subplots(figsize=(5, 3))
        if self.latency_history:
            ax.hist(list(self.latency_history), bins=20, alpha=0.7, color='blue')
            ax.axvline(np.mean(list(self.latency_history)), color='red', linestyle='dashed', linewidth=1)
            
        ax.set_title("Processing Latency Distribution")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=latency_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Latency statistics
        stats_frame = ttk.Frame(latency_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        if self.latency_history:
            latencies = list(self.latency_history)
            ttk.Label(stats_frame, text=f"Mean Latency: {np.mean(latencies):.2f} ms").pack(side=tk.LEFT, padx=5)
            ttk.Label(stats_frame, text=f"Min Latency: {np.min(latencies):.2f} ms").pack(side=tk.LEFT, padx=5)
            ttk.Label(stats_frame, text=f"Max Latency: {np.max(latencies):.2f} ms").pack(side=tk.LEFT, padx=5)
            ttk.Label(stats_frame, text=f"P95 Latency: {np.percentile(latencies, 95):.2f} ms").pack(side=tk.LEFT, padx=5)
        
        # System info tab
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System Info")
        
        system_info = [
            ("System", "Trade Simulator v1.0"),
            ("Python Version", f"{sys.version.split()[0]}"),
            ("Operating System", f"{os.name}"),
            ("WebSocket Library", f"websocket-client {websocket.__version__}"),
            ("NumPy Version", f"{np.__version__}"),
            ("Scikit-learn Version", f"{sklearn.__version__}"),
            ("Active Threads", f"{threading.active_count()}")
        ]
        
        for i, (label, value) in enumerate(system_info):
            ttk.Label(system_frame, text=f"{label}:", font=("Arial", 10, "bold")).grid(row=i, column=0, sticky="w", padx=10, pady=5)
            ttk.Label(system_frame, text=value).grid(row=i, column=1, sticky="w", padx=10, pady=5)

    def export_data(self):
        """Export current data to CSV file."""
        try:
            if not self.orderbook_processor.latest:
                messagebox.showwarning("No Data", "No orderbook data available to export.")
                return
                
            # Create dataframe for orderbook data
            bids = pd.DataFrame(self.orderbook_processor.latest['bids'], columns=['price', 'size'])
            bids['side'] = 'bid'
            
            asks = pd.DataFrame(self.orderbook_processor.latest['asks'], columns=['price', 'size'])
            asks['side'] = 'ask'
            
            # Combine data
            orderbook_df = pd.concat([bids, asks])
            
            # Add timestamp
            orderbook_df['timestamp'] = self.orderbook_processor.latest['timestamp']
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orderbook_export_{timestamp}.csv"
            orderbook_df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Successful", f"Data exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def show_documentation(self):
        """Show documentation window."""
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("700x500")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(doc_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Documentation content
        doc_content = """# Trade Simulator Documentation

## Overview
This application simulates trading on cryptocurrency exchanges by estimating transaction costs and market impact based on real-time orderbook data.

## Features
- Real-time L2 orderbook data processing
- Market impact calculation using Almgren-Chriss model
- Slippage estimation based on orderbook depth
- Fee calculation based on exchange fee tiers
- Maker/Taker proportion prediction
- Performance monitoring and analysis

## Input Parameters
- **Exchange**: Trading venue (currently OKX)
- **Spot Asset**: Trading pair (e.g., BTC-USDT-SWAP)
- **Order Type**: Order execution type (currently Market)
- **Quantity**: Order size in USD
- **Volatility**: Market volatility parameter
- **Fee Tier**: Exchange fee tier based on trading volume

## Output Parameters
- **Expected Slippage**: Price slippage based on orderbook depth
- **Expected Fees**: Trading fees based on exchange fee structure
- **Expected Market Impact**: Price impact calculated using Almgren-Chriss model
- **Net Cost**: Total transaction cost (slippage + fees + market impact)
- **Maker/Taker Proportion**: Estimated proportion of maker vs. taker fills
- **Internal Latency**: System processing time per tick

## Models
### Almgren-Chriss Market Impact Model
The Almgren-Chriss model estimates price impact with two components:
1. Permanent impact: Price changes that persist after trading
2. Temporary impact: Price changes that dissipate after trading

### Slippage Calculation
Slippage is calculated by walking the orderbook to simulate order execution.

### Maker/Taker Proportion
Logistic regression model to predict the likelihood of maker vs. taker fills.

## Performance Optimization
- Efficient data structures for orderbook processing
- Thread management for WebSocket communication
- UI updates on main thread to prevent blocking
- Periodic reconnection on connection loss

## References
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
- OKX API Documentation: https://www.okx.com/docs-v5/
"""
        
        text_widget.insert(tk.END, doc_content)
        text_widget.config(state=tk.DISABLED)  # Make read-only

    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Trade Simulator",
            "GoQuant Trade Simulator v1.0\n\n"
            "A high-performance trade simulator leveraging real-time market data "
            "to estimate transaction costs and market impact.\n\n"
            "Created for GoQuant Assessment"
        )

    def on_closing(self):
        """Clean up resources and close the application."""
        self.running = False
        if self.ws:
            self.ws.close()
        self.root.destroy()


def main():
    """Main entry point for the application."""
    # Enable exception logging
    def handle_exception(exc_type, exc_value, exc_traceback):
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Set up global exception handler
    sys.excepthook = handle_exception
    
    try:
        # Check required libraries
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import websocket
        
        # Create and run application
        root = tk.Tk()
        app = TradeSimulator(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install required packages with: pip install numpy pandas scikit-learn matplotlib websocket-client")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        logging.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()