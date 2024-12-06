import pandas as pd
import numpy as np

# Load the datasets
sales_train = pd.read_csv('data/sales_train_validation_afcs2024.csv')
calendar = pd.read_csv('data/calendar_afcs2024.csv')
sell_prices = pd.read_csv('data/sell_prices_afcs2024.csv')

# Melt the sales data to long format
sales_long = sales_train.melt(
    id_vars=['id'],
    var_name='d',
    value_name='sales'
)

# Extract day numbers (e.g., d_1 → 1) for alignment
sales_long['d'] = sales_long['d'].str.extract(r'(\d+)').astype(int)

# Add a `date` column to align sales with calendar data
sales_long = sales_long.merge(calendar, left_on='d', right_on=calendar.index + 1, how='left')

# Extract `store_id` and `item_id` from the `id` column
sales_long['store_id'] = sales_long['id'].str.split('_').str[-1]
sales_long['item_id'] = sales_long['id'].str.split('_').str[:-1].str.join('_')

# Merge with sell prices using `store_id`, `item_id`, and `wm_yr_wk`
sales_long = sales_long.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Feature Engineering
sales_long['day_of_week'] = sales_long['weekday']
sales_long['is_weekend'] = sales_long['wday'].isin([1, 7]).astype(int)

# Lag Features
lags = [1, 7, 28]
for lag in lags:
    sales_long[f'sales_lag_{lag}'] = sales_long.groupby('id')['sales'].shift(lag)

# Rolling Features
rolling_windows = [7, 28, 90]
for window in rolling_windows:
    sales_long[f'rolling_mean_{window}'] = sales_long.groupby('id')['sales'].transform(lambda x: x.rolling(window).mean())
    sales_long[f'rolling_std_{window}'] = sales_long.groupby('id')['sales'].transform(lambda x: x.rolling(window).std())

# Price Features
sales_long['price_change'] = sales_long.groupby('item_id')['sell_price'].pct_change()
sales_long['price_norm'] = sales_long['sell_price'] / sales_long.groupby('item_id')['sell_price'].transform('mean')

# Event Features
sales_long = pd.get_dummies(sales_long, columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], drop_first=True)

# Cumulative Sales
sales_long['cumulative_sales'] = sales_long.groupby('id')['sales'].cumsum()

# Seasonal Features
sales_long['sine_month'] = np.sin(2 * np.pi * sales_long['month'] / 12)
sales_long['cosine_month'] = np.cos(2 * np.pi * sales_long['month'] / 12)

# Fill Missing Values
sales_long['sell_price'] = sales_long['sell_price'].fillna(method='ffill')
sales_long['sales'] = sales_long['sales'].fillna(0)

# Final Dataset
feature_columns = [
    'day_of_week', 'is_weekend',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_28',
    'rolling_mean_7', 'rolling_mean_28', 'rolling_mean_90',
    'rolling_std_7', 'rolling_std_28', 'rolling_std_90',
    'price_change', 'price_norm',
    'cumulative_sales', 'sine_month', 'cosine_month'
] + [col for col in sales_long.columns if 'event_' in col]

final_dataset = sales_long[['id', 'date', 'sales'] + feature_columns]
final_dataset.to_parquet("engineered_features.parquet", index=False)


print("Submission file created: 'submission.parquet'")


print("Feature engineering completed. Dataset saved as 'engineered_features.parquet'.")
