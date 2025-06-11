import pandas as pd
import yfinance as yf

def preprocess_data(sales_data):
    sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%d-%m-%Y')
    sales_data['Year'] = sales_data['Date'].dt.year
    sales_data['Month'] = sales_data['Date'].dt.month
    sales_data['Day'] = sales_data['Date'].dt.day
    sales_data['WeekOfYear'] = sales_data['Date'].dt.isocalendar().week

    start_date = sales_data['Date'].min()
    end_date = sales_data['Date'].max()

    sp500_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
    sp500_data.reset_index(inplace=True)
    sp500_data_weekly = sp500_data.resample('W-FRI', on='Date').last().reset_index()

    merged_data = pd.merge(sales_data, sp500_data_weekly[['Date', 'Close']], on='Date', how='inner')
    merged_data.rename(columns={'Close': 'SP500_Close'}, inplace=True)
    merged_data['SP500_Close'] = merged_data['SP500_Close'].ffill()

    merged_data['Temp_Fuel'] = merged_data['Temperature'] * merged_data['Fuel_Price']
    merged_data['CPI_Unemployment'] = merged_data['CPI'] * merged_data['Unemployment']
    merged_data['Temp_Day'] = merged_data['Temperature'] * merged_data['Day']
    merged_data['Sales_Lag1'] = merged_data['Weekly_Sales'].shift(1)
    merged_data['Sales_Lag2'] = merged_data['Weekly_Sales'].shift(2)
    merged_data['Sales_MA_4'] = merged_data['Weekly_Sales'].rolling(window=4).mean()
    merged_data['Sales_MA_12'] = merged_data['Weekly_Sales'].rolling(window=12).mean()

    merged_data.dropna(inplace=True)
    return merged_data

def remove_date_column(merged_data):
    if 'Date' in merged_data.columns:
        merged_data.drop(['Date'], axis=1, inplace=True)
    return merged_data

def save_preprocessed_data(merged_data, output_path='preprocessed_dataframe.csv'):
    merged_data.to_csv(output_path, index=False)

def run_preprocessing_pipeline(file_path='Walmart.csv', output_path='preprocessed_dataframe.csv'):
    sales_data = pd.read_csv(file_path)
    merged_data = preprocess_data(sales_data)
    merged_data = remove_date_column(merged_data)
    save_preprocessed_data(merged_data, output_path)
    return merged_data
