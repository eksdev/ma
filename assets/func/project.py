import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf

def predict_gbm(df, days_in_future, iterations):
    # Convert DataFrame to a TensorFlow tensor for price values
    prices = tf.convert_to_tensor(df.values, dtype=tf.float32)

    # Calculate log returns
    log_returns = tf.math.log(prices[1:] / prices[:-1])

    # Calculate drift and standard deviation
    drift = tf.reduce_mean(log_returns) - 0.5 * tf.math.reduce_variance(log_returns)
    stdev = tf.math.reduce_std(log_returns)

    # Generate random values for simulations
    random_values = tf.random.normal([days_in_future, iterations], dtype=tf.float32)

    # Calculate daily returns using GBM formula
    daily_returns = tf.exp(drift + stdev * random_values)

    # Initialize the price paths array with the last actual price
    initial_prices = tf.fill([iterations], prices[-1])
    price_paths = tf.TensorArray(dtype=tf.float32, size=days_in_future + 1, clear_after_read=False)
    price_paths = price_paths.write(0, initial_prices)

    # Compute price paths using a loop
    for t in range(1, days_in_future + 1):
        last_prices = price_paths.read(t - 1)
        current_prices = last_prices * daily_returns[t - 1]  # t-1 to adjust index since daily_returns start from 0
        price_paths = price_paths.write(t, current_prices)

    # Stack all days together to form the complete paths
    price_paths = price_paths.stack()  # Shape will be [days_in_future + 1, iterations]

    return price_paths.numpy()

def get_projection_values(df, days, iterations):
    """
    Returns projection index (future days) and average predicted values for those days.

    Parameters:
    - df (pd.DataFrame): Historical price data
    - days (int): Number of days to project into the future
    - iterations (int): Number of Monte Carlo iterations

    Returns:
    - future_index (np.ndarray): Array of future day indices
    - predicted_avg_values (np.ndarray): Array of average predicted values for each future day
    """
    price_paths = predict_gbm(df, days, iterations)

    # Calculate the average predicted price for each day
    predicted_avg_values = price_paths.mean(axis=1)

    # Create a future index
    future_index = np.arange(1, days + 1)

    return future_index, predicted_avg_values
