import os
import asyncio
import json
import logging
import time
import tkinter as tk
from tkinter import ttk
import websocket
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import threading

# Configure logging to save errors and info
logging.basicConfig(level=logging.INFO, filename='trade_simulator.log', 
                    format='%(asctime)s %(levelname)s %(message)s')

class TradeSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Trade Simulator")
        self.orderbook = {'bids': [], 'asks': [], 'timestamp': ''}
        self.models = self.initialize_models()
        self.setup_ui()
        self.start_websocket()
        self.running = True

    def initialize_models(self):
        # Mock regression models (in production, train with real data)
        slippage_model = LinearRegression()  # For slippage
        maker_taker_model = LogisticRegression()  # For maker/taker
        return {'slippage': slippage_model, 'maker_taker': maker_taker_model}

    def setup_ui(self):
        # Left panel: Input parameters
        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        ttk.Label(left_frame, text="Exchange").grid(row=0, column=0, sticky="w")
        self.exchange_var = tk.StringVar(value="OKX")
        ttk.Entry(left_frame, textvariable=self.exchange_var, state='readonly').grid(row=0, column=1)

        ttk.Label(left_frame, text="Spot Asset").grid(row=1, column=0, sticky="w")
        self.asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        ttk.Combobox(left_frame, textvariable=self.asset_var, values=["BTC-USDT-SWAP"], state='readonly').grid(row=1, column=1)

        ttk.Label(left_frame, text="Order Type").grid(row=2, column=0, sticky="w")
        self.order_type_var = tk.StringVar(value="market")
        ttk.Entry(left_frame, textvariable=self.order_type_var, state='readonly').grid(row=2, column=1)

        ttk.Label(left_frame, text="Quantity (USD)").grid(row=3, column=0, sticky="w")
        self.quantity_var = tk.StringVar(value="100")
        ttk.Entry(left_frame, textvariable=self.quantity_var).grid(row=3, column=1)

        ttk.Label(left_frame, text="Volatility").grid(row=4, column=0, sticky="w")
        self.volatility_var = tk.StringVar(value="0.02")  # 2%
        ttk.Entry(left_frame, textvariable=self.volatility_var).grid(row=4, column=1)

        ttk.Label(left_frame, text="Fee Tier").grid(row=5, column=0, sticky="w")
        self.fee_tier_var = tk.StringVar(value="VIP0")
        ttk.Combobox(left_frame, textvariable=self.fee_tier_var, values=["VIP0"], state='readonly').grid(row=5, column=1)

        # Right panel: Output parameters
        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.output_labels = {}
        outputs = ["Slippage ($)", "Fees ($)", "Market Impact ($)", "Net Cost ($)", "Maker/Taker", "Latency (ms)"]
        for i, output in enumerate(outputs):
            ttk.Label(right_frame, text=output).grid(row=i, column=0, sticky="w")
            self.output_labels[output] = ttk.Label(right_frame, text="0")
            self.output_labels[output].grid(row=i, column=1)

    def start_websocket(self):
        self.ws = None
        self.connect_websocket()

    def connect_websocket(self):
        try:
            self.ws = websocket.WebSocketApp(
                "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            # Run WebSocket in a separate thread
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self.schedule_reconnect()

    def on_open(self, ws):
        logging.info("WebSocket connected")

    def on_message(self, ws, message):
        start_time = time.perf_counter()
        try:
            data = json.loads(message)
            self.orderbook['bids'] = np.array(data['bids'], dtype=float)
            self.orderbook['asks'] = np.array(data['asks'], dtype=float)
            self.orderbook['timestamp'] = data['timestamp']
            self.process_orderbook(start_time)
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")
        self.schedule_reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        logging.info(f"WebSocket closed: {close_status_code}, {close_msg}")
        if self.running:
            self.schedule_reconnect()

    def schedule_reconnect(self):
        # Attempt to reconnect after 5 seconds
        self.root.after(5000, self.connect_websocket)

    def process_orderbook(self, start_time):
        try:
            quantity = float(self.quantity_var.get())
            volatility = float(self.volatility_var.get())

            # Slippage: Estimate based on orderbook depth
            bids = self.orderbook['bids']
            slippage = 0.0
            if len(bids) > 0:
                depth = np.sum(bids[:, 1])  # Total bid volume
                slippage = (quantity / (depth + 1e-6)) * 0.01  # Simplified linear model
                # In production, use trained LinearRegression with orderbook features

            # Fees: Rule-based (OKX VIP0 taker fee: 0.1%)
            fees = quantity * 0.001  # From OKX fee schedule

            # Market Impact: Almgren-Chriss model
            # Impact = sigma * sqrt(quantity / avg_daily_volume)
            avg_daily_volume = 1000  # Mock value (replace with real data)
            market_impact = volatility * np.sqrt(quantity / avg_daily_volume)

            # Net Cost
            net_cost = slippage + fees + market_impact

            # Maker/Taker: Mock logistic regression (70% maker)
            maker_taker = 0.7  # In production, use trained LogisticRegression

            # Latency: Processing time in milliseconds
            latency = (time.perf_counter() - start_time) * 1000

            # Update UI on main thread
            self.root.after(0, self.update_ui, slippage, fees, market_impact, net_cost, maker_taker, latency)
        except Exception as e:
            logging.error(f"Error processing orderbook: {e}")

    def update_ui(self, slippage, fees, market_impact, net_cost, maker_taker, latency):
        self.output_labels["Slippage ($)"].config(text=f"{slippage:.2f}")
        self.output_labels["Fees ($)"].config(text=f"{fees:.2f}")
        self.output_labels["Market Impact ($)"].config(text=f"{market_impact:.2f}")
        self.output_labels["Net Cost ($)"].config(text=f"{net_cost:.2f}")
        self.output_labels["Maker/Taker"].config(text=f"{maker_taker:.2f}")
        self.output_labels["Latency (ms)"].config(text=f"{latency:.2f}")

    def on_closing(self):
        self.running = False
        if self.ws:
            self.ws.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradeSimulator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
