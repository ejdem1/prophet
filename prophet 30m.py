import pandas as pd
from datetime import datetime
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Załaduj dane z pliku Excel
file_path = 'DE40_M10_202112080240_202410182300.xlsx'
data = pd.read_excel(file_path)

# Skonsoliduj kolumny 'DATE' i 'TIME' do jednej kolumny 'ds'
data['ds'] = pd.to_datetime(data['DATE'].astype(str) + ' ' + data['TIME'].astype(str))

# Przemianuj kolumnę 'CLOSE' na 'y'
data = data.rename(columns={'CLOSE': 'y'})

# Sprawdzenie zakresu dat w danych
print(f"Zakres dat: {data['ds'].min()} do {data['ds'].max()}")
print(data.head())
print(data.tail())

# Pominięcie weekendów w danych
data = data[~data['ds'].dt.dayofweek.isin([5, 6])]

# Podziel dane na zestawy treningowe i testowe na podstawie końca 2023 roku
train = data[data['ds'] <= '2024-7-19']
test = data[data['ds'] > '2024-7-20']

# Dostosowanie parametrów modelu Prophet z optymalizacją hiperparametrów
model = Prophet(
    changepoint_prior_scale=0.01,  # Konserwatywna wartość, aby uniknąć nadmiernej elastyczności
    seasonality_prior_scale=1.0,  # Standardowa wartość dla sezonowości
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,  # Wyłączamy domyślną codzienną sezonowość, dodamy własną
    changepoint_range=0.9  # Zwiększenie zakresu changepoints do 90% danych treningowych
)

# Dodanie niestandardowych sezonowości z umiarkowanym fourier_order
model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
model.add_seasonality(name='ten_minutely', period=10/60/24, fourier_order=20)  # Sezonowość 10-minutowa
model.add_seasonality(name='hourly', period=1/24, fourier_order=15)  # Sezonowość godzinowa
model.add_seasonality(name='daily', period=1, fourier_order=10)  # Sezonowość dzienna

# Dopasowanie modelu na danych treningowych
model.fit(train)

# Generowanie przyszłych dat dla danych testowych
future_dates = model.make_future_dataframe(periods=len(test), freq='10min', include_history=True)

# Usuń soboty i niedziele z przyszłych dat
future_dates = future_dates[~future_dates['ds'].dt.dayofweek.isin([5, 6])]

# Prognozowanie przyszłych danych
forecast = model.predict(future_dates)

# Sprawdzenie zakresu dat w prognozie
print(f"Zakres prognozy: {forecast['ds'].min()} do {forecast['ds'].max()}")
print(forecast.head())
print(forecast.tail())

# Wyodrębnienie prognozowanych wartości dla danych testowych
forecast_test = forecast[['ds', 'yhat']]

# Dopasowanie indeksu testowych danych do prognoz
forecast_test = forecast_test.set_index('ds')
test = test.set_index('ds')
test_filtered = test[test.index.isin(forecast_test.index)]

# Upewnienie się, że indeksy są zgodne
forecast_test = forecast_test[forecast_test.index.isin(test_filtered.index)]

# Sprawdzenie długości danych
print(f"Długość danych testowych: {len(test_filtered)}")
print(f"Długość prognoz testowych: {len(forecast_test)}")

# Obliczenie reszt i metryk wydajności
residuals = test_filtered['y'] - forecast_test['yhat']
mse_prophet = mean_squared_error(test_filtered['y'], forecast_test['yhat'])
mae_prophet = mean_absolute_error(test_filtered['y'], forecast_test['yhat'])

# Wykres prognoz z punktami zmiany i pełnym zakresem dat
fig, ax = plt.subplots(figsize=(14, 7))
model.plot(forecast, ax=ax)
add_changepoints_to_plot(ax, model, forecast)
ax.plot(train['ds'], train['y'], label='Train')
ax.plot(test_filtered.index, test_filtered['y'], label='Test', color='orange')
ax.plot(forecast_test.index, forecast_test['yhat'], label='Forecast', color='green', linestyle='dashed')
ax.set_xlabel('Data')
ax.set_ylabel('Cena zamknięcia')
ax.set_title('Model Prophet z codzienną, godzinową i 10-minutową sezonowością - Prognoza vs Rzeczywistość')
ax.legend()
plt.show()

print(f'MSE: {mse_prophet}')
print(f'MAE: {mae_prophet}')
