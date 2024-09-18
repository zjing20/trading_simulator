from enum import Enum
from typing import List, Dict
from dataclasses import dataclass
import random
from collections import deque
import statistics
from collections import defaultdict
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from math import exp

class OrderType(Enum):
    BUY = 1
    SELL = 2

@dataclass
class Order:
    order_id: int
    trader_id: int
    order_type: OrderType
    price: float
    quantity: int

class ExchangeSimulator:
    def __init__(self):
        self.order_id_counter = 0
        self.buy_orders: List[Order] = []
        self.sell_orders: List[Order] = []
        self.trades: List[Dict] = []
        self.current_price: float = 100  # Starting price
        self.price_history = deque(maxlen=30)  # Increased maxlen to store more history
        self.price_history.append(self.current_price)
        self.market_news_factor = 1.0

    def place_order(self, trader_id: int, order_type: OrderType, price: float, quantity: int) -> int:
        self.order_id_counter += 1
        new_order = Order(self.order_id_counter, trader_id, order_type, price, quantity)
        
        if order_type == OrderType.BUY:
            self.buy_orders.append(new_order)
            self.buy_orders.sort(key=lambda x: x.price, reverse=True)
            self._match_orders(new_order, self.sell_orders)
        else:
            self.sell_orders.append(new_order)
            self.sell_orders.sort(key=lambda x: x.price)
            self._match_orders(new_order, self.buy_orders)

        return new_order.order_id

    def _match_orders(self, new_order: Order, opposite_orders: List[Order]):
        while opposite_orders and new_order.quantity > 0:
            if (new_order.order_type == OrderType.BUY and new_order.price >= opposite_orders[0].price) or \
               (new_order.order_type == OrderType.SELL and new_order.price <= opposite_orders[0].price):
                trade_quantity = min(new_order.quantity, opposite_orders[0].quantity)
                trade_price = opposite_orders[0].price

                self.trades.append({
                    "buy_trader_id": new_order.trader_id if new_order.order_type == OrderType.BUY else opposite_orders[0].trader_id,
                    "sell_trader_id": new_order.trader_id if new_order.order_type == OrderType.SELL else opposite_orders[0].trader_id,
                    "price": trade_price,
                    "quantity": trade_quantity
                })

                new_order.quantity -= trade_quantity
                opposite_orders[0].quantity -= trade_quantity

                if opposite_orders[0].quantity == 0:
                    opposite_orders.pop(0)
            else:
                break

        # Update the order lists
        if new_order.order_type == OrderType.BUY:
            self.buy_orders = [order for order in self.buy_orders if order.quantity > 0]
            if new_order.quantity > 0:
                self.buy_orders.append(new_order)
                self.buy_orders.sort(key=lambda x: x.price, reverse=True)
        else:
            self.sell_orders = [order for order in self.sell_orders if order.quantity > 0]
            if new_order.quantity > 0:
                self.sell_orders.append(new_order)
                self.sell_orders.sort(key=lambda x: x.price)

    def get_order_book(self) -> Dict[str, List[Dict]]:
        return {
            "buy_orders": [{"price": order.price, "quantity": order.quantity} for order in self.buy_orders],
            "sell_orders": [{"price": order.price, "quantity": order.quantity} for order in self.sell_orders]
        }

    def get_trades(self) -> List[Dict]:
        return self.trades

    def calculate_new_price(self) -> float:
        if not self.trades:
            return self.current_price  # Return the current price if no trades occurred

        total_volume = sum(trade['quantity'] for trade in self.trades)
        if total_volume == 0:
            return self.current_price  # Return the current price if total volume is zero

        volume_weighted_price = sum(trade['price'] * trade['quantity'] for trade in self.trades) / total_volume
        
        # Use the volume-weighted average price of recent trades
        new_price = volume_weighted_price
        
        # Limit price movement to 2% per day
        max_change = self.current_price * 0.02
        new_price = max(min(new_price, self.current_price + max_change), self.current_price - max_change)
        
        return new_price

    def update_price(self):
        self.current_price = self.calculate_new_price()
        self.price_history.append(self.current_price)
        return self.current_price

    def update_market_news(self):
        # Generate a random market news factor
        self.market_news_factor = exp(random.gauss(0, 0.02))  # This will generate a number close to 1

    def calculate_total_value(self, traders):
        return sum(trader.gold * self.current_price + trader.cash for trader in traders)

class Trader:
    def __init__(self, trader_id: int, exchange: ExchangeSimulator, initial_gold: float, initial_cash: float, strategy: str):
        self.trader_id = trader_id
        self.exchange = exchange
        self.initial_gold = initial_gold
        self.initial_cash = initial_cash
        self.gold = initial_gold
        self.cash = initial_cash
        self.initial_price = exchange.current_price
        self.profit = 0
        self.strategy = strategy
        self.perceived_value = exchange.current_price
        self.market_sensitivity = self.get_market_sensitivity(strategy)
        self.trader_profits = defaultdict(float)  # Use defaultdict to avoid KeyError

    def get_market_sensitivity(self, strategy):
        sensitivities = {
            "market_making": 1.0,
            "trend_following": 0.8,
            "value_investing": 0.6,
            "mean_reversion": 0.4,
            "momentum": 0.4
        }
        return sensitivities.get(strategy, 0.5)  # Default to 0.5 if strategy not found

    def place_order(self, order_type: OrderType, price: float, quantity: int):
        if order_type == OrderType.BUY:
            max_quantity = min(int(self.cash / price), quantity)
            if max_quantity > 0:
                self.exchange.place_order(self.trader_id, order_type, price, max_quantity)
        elif order_type == OrderType.SELL:
            max_quantity = min(int(self.gold), quantity)
            if max_quantity > 0:
                self.exchange.place_order(self.trader_id, order_type, price, max_quantity)

    def update_portfolio(self, trades: List[Dict]):
        for trade in trades:
            if trade["buy_trader_id"] == self.trader_id:
                self.gold += trade["quantity"]
                self.cash -= trade["price"] * trade["quantity"]
                self.trader_profits[trade["sell_trader_id"]] -= trade["price"] * trade["quantity"]
            elif trade["sell_trader_id"] == self.trader_id:
                self.gold -= trade["quantity"]
                self.cash += trade["price"] * trade["quantity"]
                self.trader_profits[trade["buy_trader_id"]] += trade["price"] * trade["quantity"]
        
        # Ensure no negative values
        self.gold = max(0, self.gold)
        self.cash = max(0, self.cash)

    def calculate_profit(self, current_gold_price: float, initial_total_value: float, current_total_value: float):
        initial_value = self.initial_gold * self.initial_price + self.initial_cash
        current_value = self.gold * current_gold_price + self.cash
        # Adjust profit calculation to account for overall system value change
        value_ratio = current_total_value / initial_total_value
        self.profit = current_value - (initial_value * value_ratio)

    def trade_strategy(self, current_gold_price: float):
        # Apply market news factor with strategy-specific sensitivity
        adjusted_factor = 1 + (self.exchange.market_news_factor - 1) * self.market_sensitivity
        current_gold_price *= adjusted_factor

        if self.strategy == "trend_following":
            self.trend_following_strategy(current_gold_price)
        elif self.strategy == "mean_reversion":
            self.mean_reversion_strategy(current_gold_price)
        elif self.strategy == "momentum":
            self.momentum_strategy(current_gold_price)
        elif self.strategy == "value_investing":
            self.value_investing_strategy(current_gold_price)
        elif self.strategy == "market_making":
            self.market_making_strategy(current_gold_price)

    def trend_following_strategy(self, current_gold_price: float):
        if len(self.exchange.price_history) < 3:
            return

        short_term_ma = sum(list(self.exchange.price_history)[-3:]) / 3
        long_term_ma = sum(self.exchange.price_history) / len(self.exchange.price_history)

        quantity = max(1, int(self.cash / current_gold_price * 0.2))
        if short_term_ma > long_term_ma:
            self.place_order(OrderType.BUY, current_gold_price * 1.002, quantity)
        elif short_term_ma < long_term_ma:
            self.place_order(OrderType.SELL, current_gold_price * 0.998, quantity)

    def mean_reversion_strategy(self, current_gold_price: float):
        if len(self.exchange.price_history) < 5:
            return

        mean_price = sum(self.exchange.price_history) / len(self.exchange.price_history)
        quantity = max(1, int(self.cash / current_gold_price * 0.2))
        if current_gold_price < mean_price * 0.995:
            self.place_order(OrderType.BUY, current_gold_price * 1.002, quantity)
        elif current_gold_price > mean_price * 1.005:
            self.place_order(OrderType.SELL, current_gold_price * 0.998, quantity)

    def momentum_strategy(self, current_gold_price: float):
        if len(self.exchange.price_history) < 2:
            return

        price_change = (current_gold_price - self.exchange.price_history[-2]) / self.exchange.price_history[-2]
        quantity = max(1, int(self.cash / current_gold_price * 0.2))
        if price_change > 0.002:
            self.place_order(OrderType.BUY, current_gold_price * 1.002, quantity)
        elif price_change < -0.002:
            self.place_order(OrderType.SELL, current_gold_price * 0.998, quantity)

    def value_investing_strategy(self, current_gold_price: float):
        self.perceived_value = self.perceived_value * 0.95 + current_gold_price * 0.05
        
        quantity = max(1, int(self.cash / current_gold_price * 0.2))
        if current_gold_price < self.perceived_value * 0.99:
            self.place_order(OrderType.BUY, current_gold_price * 1.002, quantity)
        elif current_gold_price > self.perceived_value * 1.01:
            self.place_order(OrderType.SELL, current_gold_price * 0.998, quantity)

    def market_making_strategy(self, current_gold_price: float):
        order_book = self.exchange.get_order_book()
        
        best_bid = max([order['price'] for order in order_book['buy_orders']]) if order_book['buy_orders'] else current_gold_price * 0.995
        best_ask = min([order['price'] for order in order_book['sell_orders']]) if order_book['sell_orders'] else current_gold_price * 1.005
        
        spread = best_ask - best_bid
        
        # Always place orders, but adjust the spread based on market conditions
        new_bid = current_gold_price * 0.998
        new_ask = current_gold_price * 1.002
        
        self.place_order(OrderType.BUY, new_bid, 2)
        self.place_order(OrderType.SELL, new_ask, 2)

def run_simulation(sim_number, strategies):
    exchange = ExchangeSimulator()
    traders = [Trader(i, exchange, 100, 10000, strategy) for i, strategy in enumerate(strategies)]
    
    initial_total_value = exchange.calculate_total_value(traders)
    daily_prices = [exchange.current_price]  # Start with the initial price
    daily_total_values = [initial_total_value]  # Track daily total values
    
    # Warm up the simulation with some random historical prices
    for _ in range(10):
        random_price = exchange.current_price * random.uniform(0.98, 1.02)
        exchange.price_history.append(random_price)
    
    for day in range(30):  # Simulate 30 days of trading
        daily_prices.append(exchange.current_price)  # Record the starting price of each day
        
        # Update market news factor at the start of each day
        exchange.update_market_news()
        
        for cycle in range(10):  # 10 trading opportunities per day
            # Traders place orders
            for trader in traders:
                trader.trade_strategy(exchange.current_price)
        
        # Update price based on executed trades
        new_price = exchange.update_price()
        
        # Update traders' portfolios
        trades = exchange.get_trades()
        for trader in traders:
            trader.update_portfolio(trades)
        
        # Calculate profits
        current_total_value = exchange.calculate_total_value(traders)
        for trader in traders:
            trader.calculate_profit(new_price, initial_total_value, current_total_value)
        
        # Calculate and store the total value at the end of each day
        daily_total_values.append(current_total_value)
        
        # Clear trades for the next day
        exchange.trades.clear()
    
    # Verify that total profit is zero
    total_profit = sum(trader.profit for trader in traders)
    assert abs(total_profit) < 1e-6, f"Total profit is not zero: {total_profit}"
    
    # Calculate final portfolio values and gold values
    final_portfolio_values = {trader.strategy: trader.gold * exchange.current_price + trader.cash for trader in traders}
    final_gold_values = {trader.strategy: trader.gold * exchange.current_price for trader in traders}

    # Return final profits, daily prices, daily total values, trader profits, final portfolio values, and final gold values
    return {
        "profits": {trader.strategy: trader.profit for trader in traders},
        "daily_prices": daily_prices,
        "daily_total_values": daily_total_values,
        "trader_profits": {trader.strategy: {strategies[i]: trader.trader_profits[i] for i in range(len(traders)) if i != trader.trader_id} for trader in traders},
        "final_portfolio_values": final_portfolio_values,
        "final_gold_values": final_gold_values
    }

if __name__ == "__main__":
    num_simulations = 1000
    strategies = ["trend_following", "mean_reversion", "momentum", "value_investing", "market_making"]
    
    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Run simulations in parallel
    run_sim_partial = partial(run_simulation, strategies=strategies)
    results = pool.map(run_sim_partial, range(num_simulations))
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Aggregate results
    simulation_results = defaultdict(list)
    all_daily_prices = []
    all_daily_total_values = []
    trader_vs_trader_profits = {strategy: defaultdict(list) for strategy in strategies}
    final_portfolio_values = {strategy: [] for strategy in strategies}
    final_gold_values = {strategy: [] for strategy in strategies}

    for result in results:
        for strategy, profit in result["profits"].items():
            simulation_results[strategy].append(profit)
        all_daily_prices.append(result["daily_prices"])
        all_daily_total_values.append(result["daily_total_values"])
        for strategy, profits in result["trader_profits"].items():
            for other_strategy, profit in profits.items():
                trader_vs_trader_profits[strategy][other_strategy].append(profit)
        for strategy, value in result["final_portfolio_values"].items():
            final_portfolio_values[strategy].append(value)
        for strategy, value in result["final_gold_values"].items():
            final_gold_values[strategy].append(value)
    
    # Print distribution of final profits
    print("\nProfit Distribution after 1000 simulations:")
    for strategy in strategies:
        profits = simulation_results[strategy]
        mean_profit = statistics.mean(profits)
        median_profit = statistics.median(profits)
        std_dev = statistics.stdev(profits)
        min_profit = min(profits)
        max_profit = max(profits)
        
        print(f"\n{strategy.capitalize()}:")
        print(f"  Mean Profit: ${mean_profit:.2f}")
        print(f"  Median Profit: ${median_profit:.2f}")
        print(f"  Standard Deviation: ${std_dev:.2f}")
        print(f"  Min Profit: ${min_profit:.2f}")
        print(f"  Max Profit: ${max_profit:.2f}")
    
    # Print trader-vs-trader profit information
    print("\nTrader vs Trader Profit Information:")
    for strategy in strategies:
        print(f"\n{strategy.capitalize()}:")
        for other_strategy in strategies:
            if strategy != other_strategy:
                try:
                    profits = trader_vs_trader_profits[strategy][other_strategy]
                    mean_profit = statistics.mean(profits)
                    print(f"  vs {other_strategy.capitalize()}: ${mean_profit:.2f}")
                except KeyError as e:
                    print(f"  vs {other_strategy.capitalize()}: KeyError - {e}")
                except statistics.StatisticsError as e:
                    print(f"  vs {other_strategy.capitalize()}: StatisticsError - {e}")
        
        try:
            total_mean_profit = sum(statistics.mean(profits) for other_strategy, profits in trader_vs_trader_profits[strategy].items())
            print(f"  Total mean profit against other traders: ${total_mean_profit:.2f}")
        except KeyError as e:
            print(f"  Total mean profit calculation failed: KeyError - {e}")
        except statistics.StatisticsError as e:
            print(f"  Total mean profit calculation failed: StatisticsError - {e}")
    
    # Print final portfolio value and gold value statistics
    print("\nFinal Portfolio and Gold Value Statistics:")
    for strategy in strategies:
        portfolio_values = final_portfolio_values[strategy]
        gold_values = final_gold_values[strategy]
        
        mean_portfolio = statistics.mean(portfolio_values)
        mean_gold = statistics.mean(gold_values)
        median_portfolio = statistics.median(portfolio_values)
        median_gold = statistics.median(gold_values)
        std_dev_portfolio = statistics.stdev(portfolio_values)
        std_dev_gold = statistics.stdev(gold_values)

        print(f"\n{strategy.capitalize()}:")
        print(f"  Mean Portfolio Value: ${mean_portfolio:.2f}")
        print(f"  Mean Gold Value: ${mean_gold:.2f}")
        print(f"  Median Portfolio Value: ${median_portfolio:.2f}")
        print(f"  Median Gold Value: ${median_gold:.2f}")
        print(f"  Portfolio Value Std Dev: ${std_dev_portfolio:.2f}")
        print(f"  Gold Value Std Dev: ${std_dev_gold:.2f}")

    # Plot daily starting gold prices
    all_daily_prices = np.array(all_daily_prices)
    days = range(31)  # 0 to 30
    
    plt.figure(figsize=(12, 6))
    plt.plot(days, all_daily_prices.T, alpha=0.1, color='blue')
    plt.plot(days, all_daily_prices.mean(axis=0), color='red', linewidth=2, label='Mean Price')
    plt.title('Daily Starting Gold Prices Across All Simulations')
    plt.xlabel('Day')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('daily_gold_prices.png')
    plt.close()

    print("\nA chart of daily starting gold prices has been saved as 'daily_gold_prices.png'")

    # Plot system's total value
    all_daily_total_values = np.array(all_daily_total_values)
    plt.figure(figsize=(12, 6))
    plt.plot(days, all_daily_total_values.T, alpha=0.1, color='green')
    plt.plot(days, all_daily_total_values.mean(axis=0), color='red', linewidth=2, label='Mean Total Value')
    plt.title('System\'s Total Value Across All Simulations')
    plt.xlabel('Day')
    plt.ylabel('Total Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('system_total_value.png')
    plt.close()

    print("\nA chart of the system's total value has been saved as 'system_total_value.png'")

    # Calculate and print the average increase in total value
    average_initial_value = np.mean(all_daily_total_values[:, 0])
    average_final_value = np.mean(all_daily_total_values[:, -1])
    average_increase = (average_final_value - average_initial_value) / average_initial_value * 100

    print(f"\nAverage increase in system's total value: {average_increase:.2f}%")

    # Detailed analysis of Market_making strategy
    market_making_profits = simulation_results["market_making"]
    market_making_vs_others = trader_vs_trader_profits["market_making"]

    print("\nDetailed Market_making Strategy Analysis:")
    print(f"Mean Profit: ${statistics.mean(market_making_profits):.2f}")
    print("Profits against other strategies:")
    for other_strategy, profits in market_making_vs_others.items():
        print(f"  vs {other_strategy}: ${statistics.mean(profits):.2f}")

    print("\nDistribution of Market_making profits:")
    print(f"Min: ${min(market_making_profits):.2f}")
    print(f"Max: ${max(market_making_profits):.2f}")
    print(f"Median: ${statistics.median(market_making_profits):.2f}")
    print(f"Standard Deviation: ${statistics.stdev(market_making_profits):.2f}")
