from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import yfinance as yf

class PortfolioOptimizer:
    def __init__(self, symbols, period='5y', num_portfolios=20000, simulations=3):
        self.symbols = symbols
        self.period = period
        self.num_portfolios = num_portfolios
        self.simulations = simulations
        self.data = self.initialize_data()

    def initialize_data(self):
        """Fetch and prepare data for all symbols."""
        data = {}
        errors = []
        for s in self.symbols:
            try:
                data[s] = self.get_data(ticker=s, period=self.period)
            except Exception as e:
                errors.append(f"Error fetching data for {s}: {str(e)}")
        
        if errors:
            print("\n".join(errors))  # Log errors for debugging
        
        # Create a DataFrame and clean up missing data
        df = pd.DataFrame(data)
        # Drop columns with more than 10% missing data
        df = df.dropna(axis=1, thresh=int(len(df) * 0.9))
        # Fill remaining NaN values with forward/backward filling
        return df.fillna(method='ffill').fillna(method='bfill')

    def get_data(self, ticker, period):
        """Fetch historical data for a given ticker."""
        data = yf.download(ticker, period=period, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data['Close']

    def calculate_symbol_cv(self, returns):
        """Calculate the coefficient of variation (CV) for each symbol."""
        mean_returns = returns.mean()
        std_devs = returns.std()
        return abs(std_devs / mean_returns)

    def minimize_cv(self):
        """Run optimization to minimize the coefficient of variation (CV)."""
        if self.data.empty:
            raise ValueError("No valid data available to optimize.")

        # Calculate returns, mean returns, and covariance matrix
        returns = self.data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        symbol_cvs = self.calculate_symbol_cv(returns)

        # Prepare to store portfolio results
        results = np.zeros((3, self.num_portfolios))
        weights_record = []

        for i in range(self.num_portfolios):
            # Generate random weights and normalize
            weights = np.random.random(len(self.symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)

            # Calculate portfolio return and standard deviation
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std_dev = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            ) * np.sqrt(252)

            # Calculate CV for the portfolio
            cv = abs(portfolio_std_dev / portfolio_return)

            # Store results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = cv

        # Find the portfolio with the minimum CV
        min_cv_idx = np.argmin(results[2])
        min_cv_allocation = weights_record[min_cv_idx]

        # Create allocation DataFrame
        allocation = pd.DataFrame({
            'Ticker': self.symbols,
            'Allocation (%)': min_cv_allocation * 100
        })

        # Sort symbols by their individual CVs
        sorted_symbol_cvs = symbol_cvs.sort_values()

        return allocation, results[0, min_cv_idx], results[1, min_cv_idx], sorted_symbol_cvs

    def average_allocations(self):
        """Run optimization multiple times and return the average allocation percentages."""
        allocations_list = []
        valid_simulations = 0

        for _ in range(self.simulations):
            try:
                allocation, _, _, _ = self.minimize_cv()
                allocations_list.append(allocation['Allocation (%)'].values)
                valid_simulations += 1
            except Exception as e:
                print(f"Simulation error: {str(e)}")
                continue

        # Ensure at least one valid simulation
        if valid_simulations == 0:
            raise ValueError("All simulations failed. Ensure valid symbols and data.")

        # Aggregate results and calculate average allocation
        allocations_df = pd.DataFrame(allocations_list, columns=self.symbols)
        mean_allocation = allocations_df.mean()

        # Construct the DataFrame for averaged allocation
        averaged_allocation = pd.DataFrame({
            'Ticker': self.symbols,
            'Allocation (%)': mean_allocation.tolist()
        })

        return averaged_allocation
