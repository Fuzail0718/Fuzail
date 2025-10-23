import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io

class CovidDataLoader:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
        
    def load_data(self):
        """Load COVID-19 data from Johns Hopkins GitHub repository"""
        try:
            # Load confirmed cases
            confirmed_url = self.base_url + "time_series_covid19_confirmed_global.csv"
            deaths_url = self.base_url + "time_series_covid19_deaths_global.csv"
            recovered_url = self.base_url + "time_series_covid19_recovered_global.csv"
            
            confirmed_df = pd.read_csv(confirmed_url)
            deaths_df = pd.read_csv(deaths_url)
            recovered_df = pd.read_csv(recovered_url)
            
            return confirmed_df, deaths_df, recovered_df
        except:
            # Fallback to sample data if download fails
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['US', 'India', 'Brazil', 'Russia', 'UK', 'France', 'Germany', 'Italy', 'Spain', 'Canada']
        
        # Create sample data for each metric
        np.random.seed(42)
        
        confirmed_data = {}
        deaths_data = {}
        recovered_data = {}
        
        for country in countries:
            # Simulate COVID spread with logistic growth
            base_cases = np.random.randint(1000, 10000)
            growth_rate = np.random.uniform(0.1, 0.3)
            
            confirmed = base_cases * np.exp(growth_rate * np.arange(len(dates)) / 100)
            deaths = confirmed * np.random.uniform(0.01, 0.05)
            recovered = confirmed * np.random.uniform(0.7, 0.9)
            
            # Add some noise
            confirmed += np.random.normal(0, confirmed * 0.1, len(dates))
            deaths += np.random.normal(0, deaths * 0.1, len(dates))
            recovered += np.random.normal(0, recovered * 0.1, len(dates))
            
            confirmed_data[country] = np.maximum(confirmed, 0)
            deaths_data[country] = np.maximum(deaths, 0)
            recovered_data[country] = np.maximum(recovered, 0)
        
        # Create DataFrames
        confirmed_df = pd.DataFrame(confirmed_data, index=dates)
        deaths_df = pd.DataFrame(deaths_data, index=dates)
        recovered_df = pd.DataFrame(recovered_data, index=dates)
        
        return confirmed_df, deaths_df, recovered_df
    
    def process_data(self, confirmed_df, deaths_df, recovered_df):
        """Process and clean the COVID-19 data"""
        # For time series data (non-geographical)
        if isinstance(confirmed_df, pd.DataFrame) and 'Country/Region' in confirmed_df.columns:
            # Process geographical data format
            confirmed_melted = self._melt_geographical_data(confirmed_df, 'Confirmed')
            deaths_melted = self._melt_geographical_data(deaths_df, 'Deaths')
            recovered_melted = self._melt_geographical_data(recovered_df, 'Recovered')
            
            # Merge all data
            merged_df = confirmed_melted.merge(
                deaths_melted, on=['Country/Region', 'Date'], how='left'
            ).merge(
                recovered_melted, on=['Country/Region', 'Date'], how='left'
            )
            
            # Calculate derived features
            merged_df['Active Cases'] = merged_df['Confirmed'] - merged_df['Deaths'] - merged_df['Recovered']
            merged_df['Death Rate'] = (merged_df['Deaths'] / merged_df['Confirmed'] * 100).fillna(0)
            merged_df['Recovery Rate'] = (merged_df['Recovered'] / merged_df['Confirmed'] * 100).fillna(0)
            
            # Handle missing values
            merged_df.fillna(0, inplace=True)
            
            return merged_df
        else:
            # Process time series format
            dates = confirmed_df.index
            countries = confirmed_df.columns
            
            processed_data = []
            for country in countries:
                for i, date in enumerate(dates):
                    confirmed = confirmed_df[country].iloc[i] if i < len(confirmed_df) else 0
                    deaths = deaths_df[country].iloc[i] if i < len(deaths_df) else 0
                    recovered = recovered_df[country].iloc[i] if i < len(recovered_df) else 0
                    
                    active = max(confirmed - deaths - recovered, 0)
                    death_rate = (deaths / confirmed * 100) if confirmed > 0 else 0
                    recovery_rate = (recovered / confirmed * 100) if confirmed > 0 else 0
                    
                    processed_data.append({
                        'Country/Region': country,
                        'Date': date,
                        'Confirmed': confirmed,
                        'Deaths': deaths,
                        'Recovered': recovered,
                        'Active Cases': active,
                        'Death Rate': death_rate,
                        'Recovery Rate': recovery_rate
                    })
            
            return pd.DataFrame(processed_data)
    
    def _melt_geographical_data(self, df, metric_name):
        """Melt geographical data from wide to long format"""
        id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
        date_columns = [col for col in df.columns if col not in id_vars]
        
        melted = pd.melt(
            df, 
            id_vars=id_vars, 
            value_vars=date_columns,
            var_name='Date', 
            value_name=metric_name
        )
        
        melted['Date'] = pd.to_datetime(melted['Date'])
        return melted
