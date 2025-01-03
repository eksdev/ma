import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import timedelta
import tensorflow as tf
from scipy.stats import norm
from textblob import TextBlob

# Custom request function
def requests_custom(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
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

def get_metrics(ticker):
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
    metrics_df = metrics_df[metrics_df['Metric'].isin(['Market Cap','Forward P/E','P/E','Insider Own','Short Interest','Income','Sales','ROE','ROA',"Beta","Employees","Sales Y/Y TTM"])]
    metrics_df = metrics_df.reset_index(drop=True)
    
    return metrics_df

def get_similar_stocks(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    similar_stocks = []

    stocks = soup.find_all('a', class_='loud-link fin-size-large svelte-wdkn18')
    for stock_link in stocks:
        stock = {}
        stock_name = stock_link.get('aria-label', '').strip().upper()
        if stock_name:
            stock['name'] = stock_name
            similar_stocks.append(stock)

    s_stocks = pd.DataFrame(similar_stocks)
    if s_stocks.empty:
        return s_stocks

    for metric in ['PE', 'ROE', 'ROA', 'Short Interest', 'Income', 'Sales', 'Market Cap', 'Forward P/E', 'Beta', 'Employees', 'Sales Y/Y TTM', 'Insider Own']:
        s_stocks[metric] = np.nan

    for i in range(len(s_stocks)):
        metrics_df = get_metrics(s_stocks['name'].iloc[i])
        if not metrics_df.empty:
            for metric in metrics_df['Metric']:
                if metric in s_stocks.columns and metric in metrics_df['Metric'].values:
                    s_stocks.loc[i, metric] = metrics_df.loc[metrics_df['Metric'] == metric, 'Value'].values[0]

    s_stocks = s_stocks.dropna(subset=['name'])
    s_stocks = s_stocks.reset_index(drop=True)
    s_stocks = s_stocks[~s_stocks['name'].str.contains('\.')]
    s_stocks = s_stocks.reset_index(drop=True)
    s_stocks = s_stocks[s_stocks['name'] != ticker]
    s_stocks = s_stocks.reset_index(drop=True)

    return s_stocks

def get_balance_sheet_metrics(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/balance-sheet/'
    html = requests_custom(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    rows_list = []
    bsheet = soup.find_all('div', class_='row lv-0 svelte-1xjz32c')
    if bsheet:
        for row in bsheet:
            metric_title_div = row.find('div', class_='rowTitle svelte-1xjz32c')
            if metric_title_div:
                metric_name = metric_title_div.get('title', '').strip()
                data_cols = row.find_all('div', class_='column svelte-1xjz32c')[1:]
                data_values = [col.text.strip() for col in data_col]
                rows_list.append(dict(zip(['Metric', '1/31/2024', '1/31/2023', '1/31/2022', '1/31/2021', '1/31/2020'], [metric_name] + data_values)))

    bsheet_metrics = pd.DataFrame(rows_list)
    bsheet_metrics = bsheet_metrics.dropna()
    bsheet_metrics = bsheet_metrics[bsheet_metrics['Metric'].isin(['Total Assets', 'Working Capital', 'Net Tangible Assets', 'Tangible Book Value', 'Net Debt', 'Shares Issued'])]
    
    
    return bsheet_metrics



def risk_free_rate():
    # Navigate to the target URL
    url = "https://ycharts.com/indicators/1_year_treasury_rate#:~:text=1%20Year%20Treasury%20Rate%20(I%3A1YTCMR)&text=1%20Year%20Treasury%20Rate%20is,a%20maturity%20of%201%20year."

    html = requests_custom(url)
    if html is None:
        return None
    
    # Get the page source and parse it with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the div with class 'key-stat-title' and get the text content
    key_stat_div = soup.find('div', class_='key-stat-title')
    if key_stat_div:
        key_stat_text = key_stat_div.get_text(strip=True)
       
        # Extract the rate and date from the text
        rate_str, date_str = key_stat_text.split("  for ")

        # Convert the date string to a datetime object
        date = datetime.strptime(date_str, "%b %d %Y")
        
        # Convert the rate string to a float, stripping the '%' sign
        rate = float(rate_str.replace('%', ''))
        
        return rate
    else:
        print("Div with class 'key-stat-title' not found")

    return None
    
def cost_of_debt(ticker):
    # will return the total amt spent on debt interest in most recent ann report.
    # Cost of Debt = Interest Spent ($)/ Net Debt
    url = f'https://finance.yahoo.com/quote/{ticker}/financials/'

    html = requests_custom(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Find the financial table container
    table_container = soup.find('div', class_='tableContainer svelte-1pgoo1f')

    if not table_container:
        return None

    # Extract headers (dates)
    headers = []
    table_header = table_container.find('div', class_='tableHeader svelte-1pgoo1f')
    if table_header:
        header_row = table_header.find('div', class_='row svelte-1ezv2n5')
        headers = [col.get_text(strip=True) for col in header_row.find_all('div', class_='column svelte-1ezv2n5')]
    
    # Extract data rows
    rows = table_container.find_all('div', class_='row lv-0 svelte-1xjz32c')
    for row in rows:
        metric_name_div = row.find('div', class_='rowTitle svelte-1xjz32c')
        if metric_name_div and 'interest expense' in metric_name_div.get_text(strip=True).lower():
            values = [col.get_text(strip=True).replace(',', '') for col in row.find_all('div', class_='column svelte-1xjz32c') if col.get_text(strip=True)]
            if len(values) == len(headers):
                most_recent_value = values[0]  # Assuming the first column after 'Metric' is the most recent year
                most_recent_value = most_recent_value.replace(',', '').replace('(', '-').replace(')', '')
                return float(most_recent_value)

    print("Operating Interest Expense not found in the data.")
    return None


def convert_market_cap(market_cap_str):
    # Remove letters and convert market_cap to float
    if market_cap_str.endswith('B'):
        return float(market_cap_str[:-1]) * 1e9
    elif market_cap_str.endswith('M'):
        return float(market_cap_str[:-1]) * 1e6
    elif market_cap_str.endswith('K'):
        return float(market_cap_str[:-1]) * 1e3
    else:
        return float(market_cap_str)

def wacc_collection(ticker):
    balance_sheet = get_balance_sheet_metrics(ticker) #for net debt
    key_metrics = get_metrics(ticker) #for market cap
    
    # Extract the value from the '1/31/2024' column where 'Metric' is 'Net Debt'
    net_debt_value = balance_sheet.loc[balance_sheet['Metric'] == 'Net Debt', '1/31/2024'].values[0]
    # Remove commas and convert to float
    net_debt_value = net_debt_value.replace(",", "")
    net_debt_value = float(net_debt_value)

    # Extract Market Cap as text from Key Metrics
    market_cap_str = key_metrics.loc[key_metrics['Metric'] == 'Market Cap', 'Value'].values[0]
    # Convert market_cap to float
    market_cap = convert_market_cap(market_cap_str)
    
    return net_debt_value, market_cap
    
    

def getWACC(ticker):
    corporate_tax = 0.21
    # for sake of simplicity, the cost_of_equity will be double the risk free rate.
    # WACC FORMULA = ((mkt cap / net debt + mkt cap) * [cost_of_equity]) + ((net debt / mkt cap + net debt) * [cost of debt])
    # cost of equity is essentially the min % ROI to make the project worthwhile as an investor 
    # cost of debt % is == interest expense ann. / net debt outstanding
    netdebt, mktcap = wacc_collection(ticker)
    
    debtcost = cost_of_debt(ticker)
    
    rfr = risk_free_rate()
    
    EROA = rfr  # expected return on equity is cost of equity
    
    if netdebt is None or mktcap is None or debtcost is None or rfr is None:
        return None
    
    WACC = ((mktcap / (netdebt + mktcap) * EROA)) + (netdebt / (mktcap + netdebt) * (debtcost / mktcap) * (1 - corporate_tax))
    return WACC

#def CAPM(ticker):
    # Capital Asset Pricing Model
    # rfr + 
    

def get_analyst_ratings(ticker):
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
    descript = soup.find('td', class_='fullview-profile')
    if descript:
        description = descript.text.strip()

    for n in news:
        news_store = {}
        try:
            publisher = n.find('div', class_='news-link-right flex gap-1 items-center').find_next('span')
            if publisher:
                news_store["publisher"] = publisher.text.strip()
        except Exception as e:
            print(f"Error retrieving publisher: {e}")

        try:
            article = n.find('a', class_='tab-link-news')
            if article:
                news_store["article"] = article.text.strip()
        except Exception as e:
            print(f"Error retrieving article: {e}")

        try:
            link = n.find('a', class_='tab-link-news')
            if link:
                news_store["link"] = link.get('href')
        except Exception as e:
            print(f"Error retrieving link: {e}")

        if article and article.text.strip():
            blob = TextBlob(article.text.strip())
            sentiment = blob.sentiment.polarity
            sentiment = round(sentiment, 1)
            if sentiment != 0:
                news_store['sentiment'] = sentiment
            else:
                continue
        else:
            news_store['sentiment'] = 'N/A'

        if news_store.get("publisher") in news_sources:
            list_news.append(news_store)

    for r in ratings:
        rating_store = {}
        try:
            date = r.find('td')
            if date:
                rating_store["date"] = date.text.strip()
        except Exception as e:
            print(f"Error retrieving date: {e}")

        try:
            analyst = r.find('td', class_='text-left')
            if analyst:
                rating_store["analyst"] = analyst.text.strip()
        except Exception as e:
            print(f"Error retrieving analyst: {e}")

        try:
            rating_type = r.find('td', class_='text-left').find_next()
            if rating_type:
                rating_store["rating_type"] = rating_type.text.strip()
        except Exception as e:
            print(f"Error retrieving rating type: {e}")

        list_ratings.append(rating_store)

    for i in insider_sales:
        insider_store = {}
        try:
            insider_type = i.find_all('td')[1]
            if insider_type:
                insider_store["insider_type"] = insider_type.text.strip()
        except Exception as e:
            print(f"Error retrieving insider type: {e}")

        try:
            date = i.find_all('td')[2]
            if date:
                insider_store["date"] = date.text.strip()
        except Exception as e:
            print(f"Error retrieving date of trade: {e}")

        try:
            trade_type = i.find_all('td')[3]
            if trade_type:
                insider_store["trade_type"] = trade_type.text.strip()
        except Exception as e:
            print(f"Error retrieving trade type: {e}")

        try:
            avg_price = i.find_all('td')[4]
            if avg_price:
                insider_store["avg_price"] = avg_price.text.strip()
        except Exception as e:
            print(f"Error retrieving average price: {e}")

        try:
            quantity = i.find_all('td')[5]
            if quantity:
                insider_store["quantity"] = quantity.text.strip()
        except Exception as e:
            print(f"Error retrieving quantity: {e}")

        try:
            insider_store["book_value"] = str(round(float(insider_store["quantity"].replace(',', '')) * float(insider_store["avg_price"].replace(',', ''))))
        except Exception as e:
            insider_store["book_value"] = 'N/A'

        if insider_store.get("insider_type") == "President and CEO":
            list_insider_trades.append(insider_store)

    return list_ratings, list_insider_trades, description, list_news

class StonkGather:
    def __init__(self, ticker):
        self.ticker = ticker
        self.p = '5y'  # Example: 5 years
        self.i = '1d'  # Example: 1 day interval
        # if data doesn't download for given period, download max amount of data:
        try:
            self.data = yf.download(self.ticker, period=self.p, interval=self.i)
            if self.data.empty:  # Checks if the downloaded data is empty
                raise ValueError("Data is empty, trying to download maximum available data.")
        except Exception as e:
            print(f"Error downloading data for specified period: {e}")
            print("Attempting to download maximum available data...")
            self.data = yf.download(self.ticker, period="max")

    def technical_analysis(self):
        self.data['SD'] = self.data['Close'].rolling(window=20).std()
        self.data['MA20'] = self.data['Close'].rolling(20).mean()
        self.data['MA50'] = self.data['Close'].rolling(50).mean()
        self.data['UB'] = self.data['MA20'] + (2 * self.data['SD'])
        self.data['LB'] = self.data['MA20'] - (2 * self.data['SD'])
        self.data = self.data.dropna()

        diff = (self.data['MA20'] > self.data['MA50']).astype(int)
        diff2 = (self.data['MA20'] < self.data['MA50']).astype(int)

        total_days = len(self.data)
        days_above = diff.sum()
        days_below = diff2.sum()

        if days_below > 0:
            percentage_diff = ((days_above - days_below) / days_below) * 100
        else:
            percentage_diff = float('inf')

        statement = ""
        if percentage_diff > 0:
            statement = f"the 20-Day MA of the stock traded {percentage_diff:.2f}% more time over the 50-Day MA than Under, Indicative of good performance over the last {self.p}"
        else:
            statement = f"the 20-Day MA of the stock traded {abs(percentage_diff):.2f}% less time over the 50-Day MA than Under, Indicative of poor performance over the last {self.p}."

        return statement

    def plot_stock(self):
        # Creating a DataFrame for the Bollinger Bands and Moving Averages plot
        plot_data = pd.DataFrame({
            'Close': self.data['Close'],
            'MA20': self.data['MA20'],
            'MA50': self.data['MA50'],
            'Upper BB': self.data['UB'],
            'Lower BB': self.data['LB']
        }, index=self.data.index)

        st.line_chart(plot_data)
        st.markdown(f"### {self.ticker} Data over {self.p}")

        # Creating a DataFrame for the RSI plot
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_data = pd.DataFrame({'RSI': rsi}, index=self.data.index)

        st.line_chart(rsi_data)
        st.markdown(f"### {self.ticker} RSI over {self.p}")

    def predict_gbm(self, df, days_in_future, iterations):
        prices = tf.convert_to_tensor(df.values, dtype=tf.float32)
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

    def Monte_Carlo_Simulation(self, df, days, iterations):
        price_paths = self.predict_gbm(df, days, iterations)
        last_actual_price = df.iloc[-1]
        returns = price_paths[-1] / last_actual_price - 1
        mean_returns = returns.mean()
        sd_returns = returns.std()
        thresholds = [-40, -20, -10, -5, 0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820]
        table = PrettyTable()
        table.field_names = [f"Change over next {days} days (%)", "Probability (%)"]
        plot_x_values = []
        plot_y_values = []

        for threshold in thresholds:
            prob = norm.cdf(threshold / 100, mean_returns, sd_returns) * 100
            if prob < 99:
                table.add_row([threshold, f"{prob:.2f}%"])
                plot_x_values.append(prob)
                plot_y_values.append(threshold)

        print(table)
        plt.figure()
        plt.title("Probability distribution from Monte-Carlo Simulation")
        sns.lineplot(x=plot_x_values, y=plot_y_values, marker='o')
        plt.xlabel("Probability(%)")
        plt.ylabel(f"% Change over next {days} days")
        plt.grid(True)
        plt.show()

    def forecast_stock(self):
        prediction_years = 2
        iterations = 35
        predicted_prices = self.predict_gbm(self.data['Close'], 365 * prediction_years, iterations)
        predicted_avg_prices = np.mean(predicted_prices, axis=1)
        future_dates = [self.data.index[-1] + timedelta(days=x) for x in range(1, len(predicted_avg_prices) + 1)]
        full_dates = self.data.index.append(pd.Index(future_dates))
        full_prices = np.concatenate([self.data['Close'].values, predicted_avg_prices])
        full_prices_series = pd.Series(full_prices, index=full_dates)

        ma20 = full_prices_series.rolling(window=20).mean()
        ma50 = full_prices_series.rolling(window=50).mean()
        ma200 = full_prices_series.rolling(window=200).mean()

        # Creating a DataFrame for the Historical and Forecasted Prices plot
        forecast_data = pd.DataFrame({
            'Historical Close': self.data['Close'],
            'Forecast': pd.Series(predicted_avg_prices, index=future_dates)
        }, index=full_dates)

        st.line_chart(forecast_data)
        st.markdown(f"### {self.ticker} Historical and Forecasted Prices")

# Streamlit app
st.title("Comprehensive Analysis of Stock")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "").upper()

if ticker:
    st.write(f"Analyzing {ticker}...")
    analyzer = StonkGather(ticker)

    ratings, insider_sales, description, news = get_analyst_ratings(ticker)
    st.write(description)

    if len(ratings) > 0:
        st.write("Analyst Ratings:")  # Analyst Ratings
        st.write(pd.DataFrame(ratings))
    else: 
        st.markdown(f"No Analyst Ratings on {ticker}")

    st.markdown("")
    
    st.write("Insider Trades:")  # And then insider Trades
    if len(insider_sales) > 0: 
        st.write(pd.DataFrame(insider_sales))
    else:
        st.markdown("No Insider Trades Made by the CEO Recently, Indicative of Bullish Conviction at the Company's Head")
    
    if len(news) > 0:
        st.subheader("Relevant News")
        st.write(pd.DataFrame(news))
    

    st.subheader("Technical Analysis")  # short technical analysis
    statement = analyzer.technical_analysis()
    st.write(statement)

    st.subheader("Stock Plots")
    analyzer.plot_stock()

    st.session_state.setdefault('messages', []).append({"role": "assistant", "content": "Based on Monte-Carlo Simulation I've Approximately Modelled Future Stock Price Movements"})
    
    st.subheader("Forecasted Stock Price")
    analyzer.forecast_stock()
    
    st.subheader("Key Metrics")
    metrics_df = get_metrics(ticker)
    st.table(metrics_df)

    st.subheader("Other Balance Sheet Metrics")
    balance_sheet_df = get_balance_sheet_metrics(ticker)
    st.write(balance_sheet_df)

    st.subheader("Similar Stocks Metrics")
    similar_stocks_df = get_similar_stocks(ticker)
    if len(similar_stocks_df) > 0: 
        st.session_state.messages.append({"role": "assistant", "content": f"Here are some similar stocks to {ticker}"})
        st.write(similar_stocks_df)
    else:
        default = f"No similar stocks were found for {ticker}"
        st.session_state.messages.append({"role": "assistant", "content": default})

