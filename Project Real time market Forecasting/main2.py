import streamlit as st
# st.set_page_config() must be the first Streamlit call.
st.set_page_config(layout="wide", page_title="Real Time Stock & Sentiment Dashboard")

# ------------------------------
# IMPORT LIBRARIES
# ------------------------------
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from streamlit_autorefresh import st_autorefresh

# For sentiment-based forecast (Code 2)
from finvizfinance.quote import finvizfinance
from statsmodels.tsa.statespace.sarimax import SARIMAX
import holidays
from langchain_community.llms import Ollama

# ------------------------------
# GLOBAL DATA & SESSION STATE
# ------------------------------

# Nifty 50 Company Name to Ticker Mapping (used in Real Time Dashboard)
nifty_50_dict = {
    "Apple": "AAPL",
    "Adani Ports and SEZ": "ADANIPORTS.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC": "HDFC.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Indian Oil Corporation": "IOC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "ITC": "ITC.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Lupin": "LUPIN.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "M&M": "M&M.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Nestlé India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Shree Cement": "SHREECEM.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS"
}

# Initialize session state for Real Time Dashboard (if not already set)
if 'update_chart' not in st.session_state:
    st.session_state.update_chart = False

# ------------------------------
# CODE 1 FUNCTIONS (Real Time Stock Dashboard)
# ------------------------------

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period, interval):
    """
    Fetch historical stock data from yfinance.
    """
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

def process_data(data):
    """
    Process the DataFrame:
      - Flatten MultiIndex columns (or remove ticker suffixes if present).
      - Convert the index to a timezone-aware Datetime in Asia/Kolkata.
      - Reset the index and rename 'Date' to 'Datetime' if needed.
      - Remove rows with missing 'Close' values.
    """
    if data.empty:
        st.error("No data fetched for the given ticker.")
        return data

    # Flatten MultiIndex columns if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        new_columns = {col: col.split(',')[0] for col in data.columns if isinstance(col, str)}
        data.rename(columns=new_columns, inplace=True)

    # Ensure the index is timezone-aware.
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('Asia/Kolkata')
    data.reset_index(inplace=True)

    if 'Date' in data.columns and 'Datetime' not in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)

    data = data.dropna(subset=['Close'])
    return data

def calculate_metrics(data):
    """
    Calculate key metrics from the data.
    """
    if data.empty or 'Close' not in data.columns:
        return None, None, None, None, None, None
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
    high = data['High'].max() if 'High' in data.columns else None
    low = data['Low'].min() if 'Low' in data.columns else None
    volume = data['Volume'].sum() if 'Volume' in data.columns else None
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    """
    Add 20-period SMA and EMA to the data.
    """
    if data.empty or 'Close' not in data.columns:
        st.warning("Data is empty or missing 'Close' column for technical indicators.")
        return data

    data = data.dropna(subset=['Close'])
    if len(data) < 20:
        st.warning("Not enough data to calculate SMA or EMA (need at least 20 data points).")
        return data

    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data

# ------------------------------
# CODE 2 FUNCTIONS (News Sentiment Forecast)
# ------------------------------

# Connect to local Ollama server
llm = Ollama(model='llama3')

def classify_sentiment(title):
    """
    Use LLM to classify news title sentiment as 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.
    """
    output = llm.invoke(f"Classify the sentiment as 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL' with just that one word only, no additional words or reasoning: {title}")
    return output.strip()

def get_news_data(ticker):
    """
    Pull and process news data from finvizfinance.
    """
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()

    # Preprocess and classify sentiment
    news_df['Title'] = news_df['Title'].str.lower()
    news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)
    news_df['sentiment'] = news_df['sentiment'].str.upper()
    news_df = news_df[news_df['sentiment'] != 'NEUTRAL']
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['DateOnly'] = news_df['Date'].dt.date
    return news_df

def process_sentiment_data(news_df):
    """
    Reshape news sentiment data to group by date.
    """
    grouped = news_df.groupby(['DateOnly', 'sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)
    grouped['7day_avg_positive'] = grouped['POSITIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['NEGATIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_pct_positive'] = grouped['POSITIVE'] / (grouped['POSITIVE'] + grouped['NEGATIVE'])
    result_df = grouped.reset_index()
    return result_df

def get_stock_data_for_sentiment(ticker, start_date, end_date):
    """
    Fetch stock data and compute daily percentage change.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100
    return stock_data

def combine_data(result_df, stock_data):
    print(result_df_indexed.index)
    print(stock_data_indexed.index)
    # Reset the index so that we don't carry over any MultiIndex information.
    stock_data = stock_data.reset_index()  # This converts the original DateTimeIndex into a column (typically named 'Date').
    
    # Convert the 'Date' column to datetime if it's not already, then extract the date.
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['DateOnly'] = stock_data['Date'].dt.date
    
    # Now set the index to 'DateOnly' so that it's a simple, single-level index.
    stock_data_indexed = stock_data.set_index('DateOnly')
    
    # For the news sentiment data, ensure that 'DateOnly' is of the same type.
    result_df_indexed = result_df.set_index('DateOnly')
    
    # Join on the DateOnly index.
    combined_df = result_df_indexed.join(stock_data_indexed[['Pct_Change']], how='inner')

    # Create the lagged sentiment feature.
    combined_df['lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)
    
    return combined_df


def calculate_correlation(combined_df):
    """
    Calculate Pearson correlation between lagged sentiment and stock percentage change.
    """
    correlation_pct_change = combined_df[['lagged_7day_pct_positive', 'Pct_Change']].corr().iloc[0, 1]
    return correlation_pct_change

def get_future_dates(start_date, num_days):
    """
    Get a list of future dates, excluding weekends and US holidays.
    """
    us_holidays = holidays.IN()
    future_dates = []
    current_date = start_date
    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)
    return future_dates

def fit_and_forecast(combined_df, forecast_steps=3):
    """
    Fit an ARIMAX model to forecast future stock percentage change.
    """
    endog = combined_df['Pct_Change'].dropna()
    exog = combined_df['lagged_7day_pct_positive'].dropna()
    endog = endog.loc[exog.index]
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
    fit = model.fit(disp=False)
    
    future_dates = get_future_dates(combined_df.index[-1], forecast_steps)
    future_exog = combined_df['lagged_7day_pct_positive'][-forecast_steps:].values.reshape(-1, 1)
    
    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    return forecast_mean, forecast_ci, future_dates

def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):
    """
    Create a Plotly chart showing standardized sentiment, stock percentage change, and forecast.
    """
    sentiment_std = (combined_df['7day_pct_positive'] - combined_df['7day_pct_positive'].mean()) / combined_df['7day_pct_positive'].std()

    fig = go.Figure()

    # Standardized sentiment proportion
    fig.add_trace(go.Scatter(
        x=list(combined_df.index),
        y=sentiment_std,
        name='Standardized Sentiment Proportion',
        line=dict(color='blue'),
        mode='lines'
    ))

    # Stock percentage change
    fig.add_trace(go.Scatter(
        x=list(combined_df.index),
        y=combined_df['Pct_Change'],
        name='Stock Pct Change',
        line=dict(color='green'),
        yaxis='y2',
        mode='lines'
    ))

    # Forecasted stock percentage change
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted Pct Change',
        line=dict(color='red'),
        mode='lines'
    ))

    # Forecast confidence interval
    fig.add_trace(go.Scatter(
        x=list(np.concatenate([forecast_index, forecast_index[::-1]])),
        y=list(np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]])),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title='Sentiment Proportion and Stock Percentage Change with Forecast',
        xaxis_title='Date',
        yaxis=dict(
            title='Standardized Sentiment Proportion',
            titlefont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Stock Pct Change',
            titlefont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )
    st.plotly_chart(fig)

# ------------------------------
# APP MODE SELECTION
# ------------------------------

# Use a sidebar radio button to choose between functionalities
app_mode = st.sidebar.radio("Choose App Mode", ["Real Time Stock Dashboard", "News Sentiment Forecast"])

# ==============================
# MODE 1: Real Time Stock Dashboard (Code 1)
# ==============================
if app_mode == "Real Time Stock Dashboard":
    st.title("Real Time Stock Dashboard")
    
    # Sidebar: Chart Parameters
    st.sidebar.header("Chart Parameters")
    company_name = st.sidebar.selectbox("Select Company", list(nifty_50_dict.keys()))
    ticker = nifty_50_dict[company_name]
    time_period = st.sidebar.selectbox("Time Period", ["1d", "1wk", "1mo", "1y", "max"])
    chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
    indicators = st.sidebar.multiselect("Technical Indicators", ["SMA 20", "EMA 20"])

    # Mapping time period to interval
    interval_mapping = {
        "1d": "1m",
        "1wk": "30m",
        "1mo": "1d",
        "1y": "1wk",
        "max": "1wk"
    }

    # Update Chart button
    if st.sidebar.button("Update Chart"):
        st.session_state.update_chart = True

    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60000, key="real_time_data_refresh")

    # Main Content: Chart & Data Display
    if st.session_state.update_chart:
        data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
        data = process_data(data)
        if data.empty:
            st.warning(f"No data available for ticker: {ticker}")
        else:
            data = add_technical_indicators(data)
            last_close, change, pct_change, high, low, volume = calculate_metrics(data)
            if last_close is None:
                st.error("Insufficient data to display metrics.")
            else:
                st.metric(label=f"{company_name} Last Price", value=f"{last_close:.2f} INR",
                          delta=f"{change:.2f} ({pct_change:.2f}%)")
                col1, col2, col3 = st.columns(3)
                col1.metric("High", f"{high:.2f} INR" if high is not None else "N/A")
                col2.metric("Low", f"{low:.2f} INR" if low is not None else "N/A")
                col3.metric("Volume", f"{volume:,}" if volume is not None else "N/A")
                
                # Create the stock price chart.
                fig = go.Figure()
                if chart_type == "Candlestick":
                    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                        fig.add_trace(go.Candlestick(x=data['Datetime'],
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'],
                                                     name=ticker))
                    else:
                        st.error("Required columns for Candlestick chart are missing.")
                else:
                    if 'Close' in data.columns:
                        fig = px.line(data, x='Datetime', y='Close', title=f'{ticker} Price')
                    else:
                        st.error("'Close' column is missing in data.")
                
                # Add technical indicators if selected.
                for indicator in indicators:
                    if indicator == "SMA 20" and "SMA_20" in data.columns:
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'],
                                                 mode='lines', name='SMA 20'))
                    elif indicator == "EMA 20" and "EMA_20" in data.columns:
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'],
                                                 mode='lines', name='EMA 20'))
                fig.update_layout(
                    title=f'{ticker} {time_period.upper()} Chart',
                    xaxis_title='Time',
                    yaxis_title='Price (INR)',
                    xaxis_rangeslider_visible=True,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display historical data.
                st.subheader("Historical Data")
                display_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in display_cols if col in data.columns]
                st.dataframe(data[available_cols])
                
                # Display technical indicators data.
                st.subheader("Technical Indicators")
                tech_cols = ['Datetime', 'SMA_20', 'EMA_20']
                available_tech = [col for col in tech_cols if col in data.columns]
                st.dataframe(data[available_tech][40:])
    
    # Sidebar: Real-Time Stock Prices
    st.sidebar.header("Real-Time Stock Prices")
    stock_symbols = ["HDFC Bank", "HDFC Bank", "State Bank of India"]
    for symbol in stock_symbols:
        tick_sym = nifty_50_dict[symbol]
        rt_data = yf.download(tick_sym, period='1d', interval='1m')
        rt_data = process_data(rt_data)
        if rt_data.empty or 'Open' not in rt_data.columns:
            st.sidebar.write(f"Data for {symbol} is not available.")
        else:
            try:
                first_open = float(rt_data['Open'].iloc[0])
            except Exception as e:
                st.sidebar.write(f"Error processing open price for {symbol}: {e}")
                continue

            if pd.notna(first_open) and first_open != 0:
                last_price = float(rt_data['Close'].iloc[-1])
                change = last_price - first_open
                pct_change = (change / first_open) * 100
            else:
                last_price = float(rt_data['Close'].iloc[-1])
                change = 0
                pct_change = 0

            st.sidebar.metric(f"{symbol}", f"{last_price:.2f} INR",
                              f"{change:.2f} ({pct_change:.2f}%)")
    
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This dashboard provides real-time and historical stock data with technical indicators. "
        "Data refreshes automatically every minute."
    )

# ==============================
# MODE 2: News Sentiment Forecast (Code 2)
# ==============================
elif app_mode == "News Sentiment Forecast":
    st.title("Predicting Stock Prices by News Sentiment")
    ticker_sent = st.sidebar.text_input("Enter stock ticker (e.g., SBUX):", value="SBUX")
    run_button = st.sidebar.button("Run Analysis")

    if run_button:
        news_df = get_news_data(ticker_sent)
        if news_df.empty:
            st.warning("No news data found.")
        else:
            result_df = process_sentiment_data(news_df)
            start_date = result_df['DateOnly'].min().strftime("%Y-%m-%d")
            end_date = result_df['DateOnly'].max().strftime("%Y-%m-%d")
            stock_data_sent = get_stock_data_for_sentiment(ticker_sent, start_date, end_date)
            combined_df = combine_data(result_df, stock_data_sent)
            correlation_pct_change = calculate_correlation(combined_df)
            st.write(f"Pearson correlation between lagged sentiment score and stock percentage change: {correlation_pct_change}")
            forecast_mean, forecast_ci, forecast_index = fit_and_forecast(combined_df)
            create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)
