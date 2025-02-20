The script constructs the reconstitution baskets (and backtests their performance) of the 4 following quant strategies:
1. Mean Variance Optimization. Optional implementation of Monte Carlo sims on stock returns to control input instability.
2. Risk Parity based in FF 4 factor model
3. MDP based on optimization with diversification ratio as an input
4. Equal Weights

As of now a flat file of stock prices can be used as the input but the script can be easily updated to leverage an API and is scalable for larger universes.
