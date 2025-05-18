import yfinance as yf

# Define the stock tickers and the date range
stock_tickers = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'META', 'NVDA', 'INTC', 'NFLX', 'IBM'
]
start_date = '2007-01-01'
end_date = '2023-12-31'

# Download the data from Yahoo Finance
stock_data = yf.download(stock_tickers, start=start_date, end=end_date)['Close']

# Show the first few rows of the dataset
stock_data.head()

# Clean up the data by removing rows with NaN values
stock_data.dropna(inplace=True)

# Print the shape of the dataset
print(stock_data.shape)
