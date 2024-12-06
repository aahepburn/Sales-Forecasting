# Sales-Forecasting


https://www.overleaf.com/1899317571ppvvzmhnyhnm#dd1be8




Engineered Dataset Structure
1. Key Columns
id: Unique identifier for each product and store combination (e.g., FOODS_3_001_TX_3_validation).
date: The calendar date associated with each sales entry.
sales: The number of units sold for the corresponding id and date.
2. Temporal Features
day_of_week: Name of the weekday (e.g., "Saturday").
is_weekend: Binary indicator where 1 denotes a weekend (Saturday or Sunday), and 0 denotes a weekday.
3. Lag Features
Used to capture recent sales trends:

sales_lag_1: Sales from 1 day ago.
sales_lag_7: Sales from 7 days ago.
sales_lag_28: Sales from 28 days ago (captures weekly and monthly patterns).
4. Rolling Features
Aggregate metrics over a rolling window of days:

rolling_mean_7: Average sales over the last 7 days.
rolling_mean_28: Average sales over the last 28 days.
rolling_mean_90: Average sales over the last 90 days.
rolling_std_7: Standard deviation of sales over the last 7 days.
rolling_std_28: Standard deviation of sales over the last 28 days.
rolling_std_90: Standard deviation of sales over the last 90 days.
5. Price Features
To model the influence of pricing:

sell_price: The selling price of the product for the given id and date.
price_change: Percentage change in price compared to the previous period.
price_norm: Normalized price relative to the average price of the same product.
6. Event Features
To incorporate the effect of special events:

event_name_1_*: Binary flags for the first event on the given date.
event_type_1_*: Binary flags for the type of the first event (e.g., holiday, cultural).
event_name_2_*: Binary flags for the second event on the given date.
event_type_2_*: Binary flags for the type of the second event.
7. Seasonal Features
To model seasonality patterns:

sine_month: Cyclical encoding of the month using sine transformation.
cosine_month: Cyclical encoding of the month using cosine transformation.
8. Cumulative Features
To track cumulative sales trends:

cumulative_sales: Running total of sales for the given id.
