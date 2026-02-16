from prophet import Prophet
import pandas as pd
import os
from datetime import timedelta


FORECAST_DIR = 'data/forecasts'
os.makedirs(FORECAST_DIR, exist_ok=True)

class ProphetForecaster:
    def __init__(self, data_type):
        self.data_type = data_type
        self.model = None
        self.forecast_df = None
        self.last_trained_date = None
        
    def train(self, df):
        """Обучает модель на всех исторических данных"""
        prophet_df = df.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
        
        # Оптимальные параметры из тюнинга
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=3,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            changepoint_range=0.85,
            n_changepoints=35,
            seasonality_prior_scale=0.1,
            seasonality_mode='additive'
        )
        
        self.model.fit(prophet_df)
        self.last_trained_date = df.index.max()
        return self.model
    
    def ensure_forecast_until(self, target_date):
        """Гарантирует наличие прогноза до указанной даты"""
        target_date = pd.to_datetime(target_date)
        
        last_historical_date = pd.to_datetime('2026-01-31')
    
        # Если дата меньше или равна последней исторической - прогноз не нужен
        if target_date <= last_historical_date:
            return
        
        # Загружаем существующий прогноз
        self.load_forecast()
        
        # Загружаем исторические данные для обучения
        data_file = f'data/{self.data_type}.csv'
        df = pd.read_csv(data_file, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df = df.asfreq('D').sort_index()
        df['value'] = df['value'].interpolate()

        # Если прогноза нет, создаем с нуля
        if self.forecast_df is None:
            print(f"Создаю новый прогноз...")
            self.train(df)
            self.forecast_df = self._forecast_until(target_date)
            self.save_forecast()
            return
        
        # Проверяем, достаточно ли далеко прогноз
        last_forecast_date = self.forecast_df['ds'].max()
        
        if last_forecast_date < target_date:
            # Нужно продлить прогноз
            days_needed = (target_date - last_forecast_date).days + 7  # +7 запас
            print(f"Продлеваю прогноз на {days_needed} дней до {target_date.date()}...")
            
            # Переобучаем модель на всех данных
            self.train(df)
            
            # Делаем новый прогноз до нужной даты
            self.forecast_df = self._forecast_until(target_date)
            self.save_forecast()
            print(f"Прогноз продлен до {target_date.date()}")
    
    def _forecast_until(self, target_date):
        """Внутренний метод: делает прогноз до указанной даты"""
        last_date = pd.to_datetime(self.last_trained_date)
        days_to_forecast = (pd.to_datetime(target_date) - last_date).days
        
        if days_to_forecast <= 0:
            days_to_forecast = 30  # минимум 30 дней
            
        future = self.model.make_future_dataframe(periods=days_to_forecast, freq='D')
        forecast = self.model.predict(future)
        
        # Берем только будущие даты (после последней обученной)
        forecast = forecast[forecast['ds'] > last_date]
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def save_forecast(self):
        """Сохраняет прогноз в CSV"""
        filename = f"{FORECAST_DIR}/prophet_forecast_{self.data_type}.csv"
        self.forecast_df.to_csv(filename, index=False)
        return filename
    
    def load_forecast(self):
        """Загружает существующий прогноз"""
        filename = f"{FORECAST_DIR}/prophet_forecast_{self.data_type}.csv"
        if os.path.exists(filename):
            self.forecast_df = pd.read_csv(filename, parse_dates=['ds'])
            return True
        return False

    def get_forecast_for_date(self, target_date):
        """Возвращает данные на конкретную дату (история или прогноз)"""
        target = pd.to_datetime(target_date).date()
        last_historical = pd.to_datetime('2026-01-31').date()
        
        # Если дата в историческом периоде
        if target <= last_historical:
            # Загружаем исторические данные
            hist_df = pd.read_csv(f'data/{self.data_type}.csv', parse_dates=['date'])
            hist_df.set_index('date', inplace=True)
            
            if target in hist_df.index:
                value = hist_df.loc[str(target), 'value']
                return {
                    'date': target.strftime('%Y-%m-%d'),
                    'value': round(value, 2),
                    'source': 'history'
                }
            return None
        
        # Иначе - из прогноза
        if self.forecast_df is None:
            return None
        
        forecast_date = pd.to_datetime(target_date)
        mask = self.forecast_df['ds'].dt.date == forecast_date.date()
        
        if mask.any():
            row = self.forecast_df[mask].iloc[0]
            return {
                'date': row['ds'].strftime('%Y-%m-%d'),
                'value': round(row['yhat'], 2),
                'lower': round(row['yhat_lower'], 2),
                'upper': round(row['yhat_upper'], 2),
                'source': 'forecast'
            }
        return None

    def get_forecast_for_period(self, start_date, end_date):
        """Возвращает данные за период (история + прогноз если нужно)"""
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        last_historical = pd.to_datetime('2026-01-31').date()
        
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'total': 0,
            'avg': 0,
            'min': float('inf'),
            'max': float('-inf'),
            'days': 0,
            'sources': []
        }
        
        # Загружаем исторические данные
        hist_df = pd.read_csv(f'data/{self.data_type}.csv', parse_dates=['date'])
        hist_df.set_index('date', inplace=True)
        
        # Часть 1: История (до 2026-01-31)
        hist_end = min(end, last_historical)
        if start <= hist_end:
            hist_mask = (hist_df.index.date >= start) & (hist_df.index.date <= hist_end)
            hist_data = hist_df[hist_mask]
            
            if len(hist_data) > 0:
                result['total'] += hist_data['value'].sum()
                result['min'] = min(result['min'], hist_data['value'].min())
                result['max'] = max(result['max'], hist_data['value'].max())
                result['days'] += len(hist_data)
                result['sources'].append('history')
        
        # Часть 2: Прогноз (после 2026-01-31)
        forecast_start = max(start, last_historical + timedelta(days=1))
        if forecast_start <= end and self.forecast_df is not None:
            forecast_mask = (self.forecast_df['ds'].dt.date >= forecast_start) & \
                            (self.forecast_df['ds'].dt.date <= end)
            forecast_data = self.forecast_df[forecast_mask]
            
            if len(forecast_data) > 0:
                result['total'] += forecast_data['yhat'].sum()
                result['min'] = min(result['min'], forecast_data['yhat'].min())
                result['max'] = max(result['max'], forecast_data['yhat'].max())
                result['days'] += len(forecast_data)
                result['sources'].append('forecast')
        
        if result['days'] > 0:
            result['avg'] = round(result['total'] / result['days'], 2)
            result['total'] = round(result['total'], 2)
            result['min'] = round(result['min'], 2)
            result['max'] = round(result['max'], 2)
            result['sources'] = '+'.join(set(result['sources']))
            return result
        
        return None
    