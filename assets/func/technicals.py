import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from scipy.stats import norm
from textblob import TextBlob

# ---------------------- HELPER FUNCTIONS ----------------------

def requests_custom(url):
    """
    A custom requests function that sets headers for a more reliable fetch.
    Returns the HTML content if successful, else None.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

def convert_market_cap(market_cap_str):
    """
    Converts a Finviz-style Market Cap string to a float (e.g. '1.2B' -> 1200000000).
    """
    if not market_cap_str:
        return None
    if market_cap_str.endswith('B'):
        return float(market_cap_str[:-1]) * 1e9
    elif market_cap_str.endswith('M'):
        return float(market_cap_str[:-1]) * 1e6
    elif market_cap_str.endswith('K'):
        return float(market_cap_str[:-1]) * 1e3
    else:
        # Attempt a direct float parse
        try:
            return float(market_cap_str)
        except ValueError:
            return None

# ---------------------- METRIC SCRAPING ----------------------

def get_metrics(ticker):
    """
    Fetches key metrics for a given ticker (P/E, Forward P/E, Market Cap, etc.)
    from finviz.com. Returns a DataFrame with 'Metric' and 'Value' columns.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    metrics_table = soup.find('table', class_='js-snapshot-table snapshot-table2 screener_snapshot-table-body')

    metrics = []
    if metrics_table:
        for row in metrics_table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) % 2 == 0:
                for i in range(0, len(cols), 2):
                    metric_name = cols[i].text.strip()
                    metric_value = cols[i + 1].text.strip()
                    metrics.append({'Metric': metric_name, 'Value': metric_value})

    metrics_df = pd.DataFrame(metrics)
    # Keep only relevant rows
    metrics_df = metrics_df[
        metrics_df['Metric'].isin([
            'Market Cap','Forward P/E','P/E','Insider Own','Short Interest','Income','Sales',
            'ROE','ROA','Beta','Employees','Sales Y/Y TTM'
        ])
    ]
    metrics_df = metrics_df.reset_index(drop=True)

    return metrics_df

def get_similar_stocks(ticker):
    """
    Attempts to fetch "similar" stocks from Yahoo Finance's main quote page.
    Returns a DataFrame with columns including 'name' and various metrics (PE, ROE, etc.).
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    similar_stocks = []

    # Example approach: looking for links with class_='loud-link fin-size-large...'
    stocks = soup.find_all('a', class_='loud-link fin-size-large svelte-wdkn18')
    for stock_link in stocks:
        stock_name = stock_link.get('aria-label', '').strip().upper()
        if stock_name:
            similar_stocks.append({'name': stock_name})

    s_stocks = pd.DataFrame(similar_stocks)
    if s_stocks.empty:
        return s_stocks

    # Prepare columns in advance
    for metric in ['PE', 'ROE', 'ROA', 'Short Interest', 'Income', 'Sales', 'Market Cap',
                   'Forward P/E', 'Beta', 'Employees', 'Sales Y/Y TTM', 'Insider Own']:
        s_stocks[metric] = np.nan

    # For each "similar" stock, fetch metrics
    for i in range(len(s_stocks)):
        symbol = s_stocks.loc[i, 'name']
        metrics_df = get_metrics(symbol)
        if not metrics_df.empty:
            for row_index, row_data in metrics_df.iterrows():
                metric_name = row_data['Metric']
                metric_value = row_data['Value']
                if metric_name in s_stocks.columns:
                    s_stocks.loc[i, metric_name] = metric_value

    # Clean up
    s_stocks = s_stocks.dropna(subset=['name']).reset_index(drop=True)
    s_stocks = s_stocks[~s_stocks['name'].str.contains('\.')]
    s_stocks = s_stocks[s_stocks['name'] != ticker].reset_index(drop=True)

    return s_stocks

def get_balance_sheet_metrics(ticker):
    """
    Fetches partial balance sheet data (e.g. from Yahoo's /balance-sheet page).
    Returns a DataFrame with columns like 'Metric', '1/31/2024', etc.
    """
    url = f'https://finance.yahoo.com/quote/{ticker}/balance-sheet/'
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    rows_list = []
    bsheet = soup.find_all('div', class_='row lv-0 svelte-1xjz32c')
    if not bsheet:
        return pd.DataFrame()

    for row in bsheet:
        metric_title_div = row.find('div', class_='rowTitle svelte-1xjz32c')
        if metric_title_div:
            metric_name = metric_title_div.get('title', '').strip()
            data_cols = row.find_all('div', class_='column svelte-1xjz32c')[1:]
            data_values = [col.text.strip() for col in data_cols]
            # Example: match the number of columns to the years
            # This is a placeholder, adjust to your expected columns:
            row_dict = {'Metric': metric_name}
            for i, val in enumerate(data_values):
                row_dict[f'Val_{i}'] = val
            rows_list.append(row_dict)

    bsheet_metrics = pd.DataFrame(rows_list).dropna()
    # If you specifically only want certain rows:
    # bsheet_metrics = bsheet_metrics[bsheet_metrics['Metric'].isin([...])]
    return bsheet_metrics

# ---------------------- RATES & WACC ----------------------

def risk_free_rate():
    """
    Tries to scrape the 1-Year Treasury Rate from ycharts.com.
    Returns the float rate, or None if not found.
    """
    url = "https://ycharts.com/indicators/1_year_treasury_rate#:~:text=1%20Year%20Treasury%20Rate%20(I%3A1YTCMR)..."
    html = requests_custom(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, 'html.parser')
    key_stat_div = soup.find('div', class_='key-stat-title')
    if key_stat_div:
        key_stat_text = key_stat_div.get_text(strip=True)
        try:
            rate_str, date_str = key_stat_text.split("  for ")
            rate = float(rate_str.replace('%', ''))
            return rate
        except Exception as e:
            print(f"Error extracting rate: {e}")
    else:
        print("Div with class 'key-stat-title' not found")
    return None

def cost_of_debt(ticker):
    """
    Returns the interest expense from Yahoo's /financials to approximate cost of debt.
    If found, returns a float. Otherwise None.
    """
    url = f'https://finance.yahoo.com/quote/{ticker}/financials/'
    html = requests_custom(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, 'html.parser')
    table_container = soup.find('div', class_='tableContainer svelte-1pgoo1f')
    if not table_container:
        return None

    headers = []
    table_header = table_container.find('div', class_='tableHeader svelte-1pgoo1f')
    if table_header:
        header_row = table_header.find('div', class_='row svelte-1ezv2n5')
        headers = [col.get_text(strip=True) for col in header_row.find_all('div', class_='column svelte-1ezv2n5')]

    rows = table_container.find_all('div', class_='row lv-0 svelte-1xjz32c')
    for row in rows:
        metric_name_div = row.find('div', class_='rowTitle svelte-1xjz32c')
        if metric_name_div and 'interest expense' in metric_name_div.get_text(strip=True).lower():
            values = [col.get_text(strip=True).replace(',', '') for col in row.find_all('div', class_='column svelte-1xjz32c')]
            # Typically, values[0] might be the label 'Interest Expense', values[1] the most recent year, etc.
            if len(values) >= 2:
                # The first numeric column after the label is presumably the most recent
                most_recent_value = values[1] if len(values) > 1 else None
                if most_recent_value:
                    most_recent_value = most_recent_value.replace('(', '-').replace(')', '')
                    try:
                        return float(most_recent_value)
                    except ValueError:
                        return None

    print("Operating Interest Expense not found in the data.")
    return None

def wacc_collection(ticker):
    """
    Example function to fetch net debt (from get_balance_sheet_metrics)
    and market cap (from get_metrics) to help compute WACC or cost-of-capital.
    Returns (net_debt_value, market_cap) or (None, None) if missing.
    """
    balance_sheet = get_balance_sheet_metrics(ticker)  # For net debt
    key_metrics = get_metrics(ticker)                  # For market cap

    # Example logic: find 'Net Debt' in your extracted balance sheet
    # This is just a placeholder example because the columns may differ
    net_debt_row = balance_sheet[balance_sheet['Metric'].str.contains('Net Debt', case=False, na=False)]
    if net_debt_row.empty:
        return None, None
    # Suppose the most recent value is in 'Val_0'
    net_debt_str = net_debt_row['Val_0'].iloc[0].replace(",", "")
    try:
        net_debt_value = float(net_debt_str)
    except:
        net_debt_value = None

    # Extract Market Cap from key_metrics
    mc_row = key_metrics[key_metrics['Metric'] == 'Market Cap']
    if mc_row.empty:
        return net_debt_value, None
    market_cap_str = mc_row['Value'].iloc[0]
    market_cap = convert_market_cap(market_cap_str)

    return net_debt_value, market_cap

def getWACC(ticker):
    """
    Uses simplistic logic to compute a WACC, given net debt, market cap, cost_of_debt, risk_free_rate, etc.
    This is purely illustrative and may not reflect real-world WACC calculations.
    """
    corporate_tax = 0.21
    netdebt, mktcap = wacc_collection(ticker)
    if netdebt is None or mktcap is None:
        return None

    debtcost = cost_of_debt(ticker)
    rfr = risk_free_rate()

    # For demonstration, let's say cost_of_equity = rfr
    if debtcost is None or rfr is None:
        return None

    # Simple WACC formula
    # WACC = (E/(D+E)*cost_of_equity) + (D/(D+E)*cost_of_debt*(1 - tax))
    EROA = rfr  # example cost_of_equity
    WACC = (
        (mktcap/(netdebt+mktcap)*EROA) +
        (netdebt/(netdebt+mktcap)*(debtcost / abs(netdebt))* (1 - corporate_tax))
    )
    return WACC

# ---------------------- ANALYST RATINGS & NEWS ----------------------

def get_analyst_ratings(ticker):
    """
    Fetches analyst ratings, insider sales, company description, and selected news from Finviz.
    Returns (list_ratings, list_insider_trades, description, list_news).
    Each is a Python list (except description = string).
    """
    news_sources = ['(Motley Fool)', '(Reuters)', '(InvestorPlace)', '(The Wall Street Journal)']
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return [], [], "", []

    soup = BeautifulSoup(html, 'html.parser')
    ratings = soup.find_all('tr', class_='styled-row is-hoverable is-bordered is-rounded is-border-top is-hover-borders has-label has-color-text')
    insider_sales = soup.find_all('tr', class_='fv-insider-row')
    news = soup.find_all('tr', class_='cursor-pointer has-label')

    list_ratings = []
    list_insider_trades = []
    list_news = []
    description = ""

    # Description
    descript = soup.find('td', class_='fullview-profile')
    if descript:
        description = descript.text.strip()

    # News
    for n in news:
        news_store = {}
        try:
            publisher = n.find('div', class_='news-link-right flex gap-1 items-center').find_next('span')
            if publisher:
                news_store["publisher"] = publisher.text.strip()
        except Exception as e:
            pass

        try:
            article = n.find('a', class_='tab-link-news')
            if article:
                news_store["article"] = article.text.strip()
                news_store["link"] = article.get('href')
        except Exception as e:
            pass

        # Sentiment
        if "article" in news_store:
            blob = TextBlob(news_store["article"])
            sentiment = blob.sentiment.polarity
            sentiment = round(sentiment, 1)
            if sentiment != 0:
                news_store['sentiment'] = sentiment
            else:
                news_store['sentiment'] = 0
        else:
            news_store['sentiment'] = 'N/A'

        if news_store.get("publisher") in news_sources:
            list_news.append(news_store)

    # Ratings
    for r in ratings:
        rating_store = {}
        try:
            date = r.find('td')
            if date:
                rating_store["date"] = date.text.strip()
        except:
            pass

        try:
            analyst = r.find('td', class_='text-left')
            if analyst:
                rating_store["analyst"] = analyst.text.strip()
        except:
            pass

        try:
            rating_type = r.find('td', class_='text-left').find_next()
            if rating_type:
                rating_store["rating_type"] = rating_type.text.strip()
        except:
            pass

        list_ratings.append(rating_store)

    # Insider trades
    for i in insider_sales:
        insider_store = {}
        tds = i.find_all('td')
        if len(tds) >= 6:
            insider_store["insider_type"] = tds[1].text.strip()
            insider_store["date"] = tds[2].text.strip()
            insider_store["trade_type"] = tds[3].text.strip()
            insider_store["avg_price"] = tds[4].text.strip()
            insider_store["quantity"] = tds[5].text.strip()
            try:
                # approximate book value
                share_ct = float(insider_store["quantity"].replace(',', ''))
                avg_prc = float(insider_store["avg_price"].replace(',', ''))
                insider_store["book_value"] = str(round(share_ct * avg_prc))
            except:
                insider_store["book_value"] = 'N/A'

            if insider_store.get("insider_type") == "President and CEO":
                list_insider_trades.append(insider_store)

    return list_ratings, list_insider_trades, description, list_news

# ---------------------- TECHNICAL / ANALYSIS CLASS ----------------------

class StonkGather:
    """
    A class that fetches historical data for a given ticker (default 5y, 1d) from yfinance,
    then performs some technical analysis calculations (like Bollinger Bands).
    We do NOT rely on streamlit for plotting; instead we return strings or DataFrames
    so that you can visualize them in Dash or another framework.
    """
    def __init__(self, ticker, period='5y', interval='1d'):
        self.ticker = ticker.upper()
        self.p = period
        self.i = interval
        try:
            self.data = yf.download(self.ticker, period=self.p, interval=self.i)
            if self.data.empty:
                # Attempt maximum data if initial fetch is empty
                print("Data is empty for specified period/interval, trying max.")
                self.data = yf.download(self.ticker, period="max")
        except Exception as e:
            print(f"Error downloading data for {self.ticker}: {e}")
            print("Attempting maximum data fetch...")
            self.data = yf.download(self.ticker, period="max")

        # If still empty, the user might handle it externally
        self.data = self.data.dropna()

    def technical_analysis_statement(self):
        """
        Calculates Bollinger Band stats (20-day/50-day MAs) and returns a textual statement
        about how often the 20-day MA is above/below the 50-day MA.
        """
        if self.data.empty:
            return "No data available for technical analysis."

        df = self.data.copy()
        df['SD'] = df['Close'].rolling(window=20).std()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['UB'] = df['MA20'] + (2 * df['SD'])
        df['LB'] = df['MA20'] - (2 * df['SD'])
        df = df.dropna()

        total_days = len(df)
        if total_days == 0:
            return "No sufficient data after rolling calculations."

        diff_above = (df['MA20'] > df['MA50']).sum()
        diff_below = (df['MA20'] < df['MA50']).sum()
        if diff_below > 0:
            percentage_diff = ((diff_above - diff_below) / diff_below) * 100
        else:
            percentage_diff = float('inf')

        if percentage_diff > 0:
            statement = (
                f"Over the last {self.p}, the 20-day MA traded {percentage_diff:.2f}% more time "
                f"above the 50-day MA than below, indicating relatively good performance."
            )
        else:
            statement = (
                f"Over the last {self.p}, the 20-day MA traded {abs(percentage_diff):.2f}% more time "
                f"below the 50-day MA than above, indicating relatively weak performance."
            )

        return statement

    def get_rsi(self, window=14):
        """
        Compute RSI for the current data, return a pd.Series or empty if no data.
        """
        if self.data.empty:
            return pd.Series(dtype=float)

        df = self.data.copy()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = "RSI"
        return rsi.dropna()

    def predict_gbm(self, df_close, days_in_future, iterations):
        """
        A naive Monte-Carlo approach with geometric Brownian motion, returning
        a (days_in_future+1 x iterations) array of simulated price paths.
        """
        if df_close.empty:
            return np.array([])

        prices = tf.convert_to_tensor(df_close.values, dtype=tf.float32)
        log_returns = tf.math.log(prices[1:] / prices[:-1])
        drift = tf.reduce_mean(log_returns) - 0.5 * tf.math.reduce_variance(log_returns)
        stdev = tf.math.reduce_std(log_returns)
        random_values = tf.random.normal([days_in_future, iterations], dtype=tf.float32)
        daily_returns = tf.exp(drift + stdev * random_values)
        initial_prices = tf.fill([iterations], prices[-1])
        price_paths = tf.TensorArray(dtype=tf.float32, size=days_in_future + 1, clear_after_read=False)
        price_paths = price_paths.write(0, initial_prices)

        for t in range(1, days_in_future + 1):
            last_prices = price_paths.read(t - 1)
            current_prices = last_prices * daily_returns[t - 1]
            price_paths = price_paths.write(t, current_prices)

        price_paths = price_paths.stack()
        return price_paths.numpy()

    def forecast_prices(self, future_days=365, iterations=30):
        """
        Returns an array of shape (future_days+1, iterations) representing simulated price paths,
        plus a 1D average path. The last row is the simulated prices after 'future_days'.
        """
        if self.data.empty:
            return None, None

        df_close = self.data['Close'].dropna()
        if df_close.empty:
            return None, None

        price_paths = self.predict_gbm(df_close, future_days, iterations)
        if price_paths.size == 0:
            return None, None

        # The average across iterations at each day
        avg_path = price_paths.mean(axis=1)
        return price_paths, avg_path

    # Additional Monte Carlo or forecasting methods could be added, returning
    # raw data that your Dash app can turn into figures or tables.
