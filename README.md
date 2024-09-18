# Gold Trading Exchange Simulator

## Overview

This is built from my interaction with Cursor + Claude-3.5-sonnet.

This project simulates a gold trading exchange with multiple trading strategies. It aims to analyze the performance of different trading strategies in a simplified market environment.

## Features

- Simulates a gold trading exchange over a 30-day period
- Implements five different trading strategies:
  - Trend Following
  - Mean Reversion
  - Momentum
  - Value Investing
  - Market Making
- Runs multiple simulations in parallel for statistical analysis
- Generates performance metrics and visualizations

## Requirements

- Python 3.7+
- Required libraries:
  - numpy
  - matplotlib
  - multiprocessing
  - statistics

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gold-trading-simulator.git
   cd gold-trading-simulator
   ```

2. Install the required libraries:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

Run the simulation by executing the main script:

```bash
python exchange_simulator.py
```

This will run 1000 simulations and output the results, including:
- Profit distribution for each strategy
- Trader vs. trader profit information
- Final portfolio and gold value statistics
- Charts of daily gold prices and system total value

## Output

The simulation generates two charts:
1. `daily_gold_prices.png`: Shows the daily starting gold prices across all simulations
2. `system_total_value.png`: Displays the system's total value across all simulations

Additionally, detailed statistics are printed to the console, including:
- Profit distribution for each strategy
- Trader vs. trader performance
- Final portfolio and gold value statistics
- Detailed analysis of the Market Making strategy

## Customization

You can modify the following parameters in `exchange_simulator.py`:
- `num_simulations`: Number of simulations to run (default: 1000)
- `strategies`: List of trading strategies to include in the simulation
- Simulation duration (default: 30 days)
- Initial conditions for traders (gold and cash)

## Contributing

Contributions to improve the simulation or add new features are welcome. Please feel free to submit a pull request or open an issue for discussion.

## License

This project is open-source and available under the [MIT License](LICENSE).
