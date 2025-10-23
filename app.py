# COVID-19 Dashboard - No External Dependencies
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CovidDataLoader:
    def create_sample_data(self):
        """Create sample COVID-19 data without external dependencies"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['United States', 'India', 'Brazil', 'Russia', 'United Kingdom', 
                    'France', 'Germany', 'Italy', 'Spain', 'Canada']
        
        np.random.seed(42)
        processed_data = []
        
        for country in countries:
            base_cases = np.random.randint(50000, 200000)
            t = np.arange(len(dates)) / 100
            
            # Simulate COVID growth
            growth = base_cases / (1 + np.exp(-0.15 * (t - 5)))
            wave1 = 0.4 * np.sin(2 * np.pi * t / 2) * growth
            wave2 = 0.3 * np.sin(2 * np.pi * (t - 1.5) / 3) * growth
            
            confirmed = growth + wave1 + wave2
            deaths = confirmed * np.random.uniform(0.015, 0.035)
            recovered = confirmed * np.random.uniform(0.65, 0.85)
            
            # Add noise
            confirmed = np.maximum(confirmed + np.random.normal(0, confirmed * 0.08, len(dates)), 0)
            deaths = np.maximum(deaths + np.random.normal(0, deaths * 0.12, len(dates)), 0)
            recovered = np.maximum(recovered + np.random.normal(0, recovered * 0.1, len(dates)), 0)
            active = np.maximum(confirmed - deaths - recovered, 0)
            
            for i, date in enumerate(dates):
                death_rate = (deaths[i] / confirmed[i] * 100) if confirmed[i] > 0 else 0
                recovery_rate = (recovered[i] / confirmed[i] * 100) if confirmed[i] > 0 else 0
                
                processed_data.append({
                    'Country/Region': country,
                    'Date': date,
                    'Confirmed': confirmed[i],
                    'Deaths': deaths[i],
                    'Recovered': recovered[i],
                    'Active Cases': active[i],
                    'Death Rate': death_rate,
                    'Recovery Rate': recovery_rate
                })
        
        return pd.DataFrame(processed_data)

class CovidDashboard:
    def __init__(self):
        self.loader = CovidDataLoader()
        self.data = None
        
    def load_data(self):
        """Load COVID-19 data"""
        self.data = self.loader.create_sample_data()
            
    def run(self):
        """Run the dashboard application"""
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 EDA Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; color: #666;'>
        <p>Interactive dashboard for COVID-19 data analysis and visualization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        if self.data is None:
            self.load_data()
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Apply filters
        filtered_data = self.apply_filters()
        
        # Main content
        self.display_overview(filtered_data)
        self.display_eda_section(filtered_data)
        self.display_interactive_charts(filtered_data)
        
    def setup_sidebar(self):
        """Setup sidebar filters"""
        st.sidebar.title("🔧 Filters & Controls")
        st.sidebar.markdown("---")
        
        min_date = self.data['Date'].min()
        max_date = self.data['Date'].max()
        
        # Date range
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Country selection
        countries = sorted(self.data['Country/Region'].unique())
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            countries,
            default=countries[:5]
        )
        
        # Metric selection
        self.selected_metric = st.sidebar.selectbox(
            "Select Metric",
            ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 Use filters to customize the analysis")
    
    def apply_filters(self):
        """Apply filters to data"""
        filtered_data = self.data.copy()
        
        # Apply country filter
        selected_countries = st.session_state.get("countries", self.data['Country/Region'].unique()[:5])
        filtered_data = filtered_data[filtered_data['Country/Region'].isin(selected_countries)]
        
        return filtered_data
    
    def display_overview(self, data):
        """Display overview metrics"""
        st.header("📊 Overview Metrics")
        
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        total_confirmed = latest_data['Confirmed'].sum()
        total_deaths = latest_data['Deaths'].sum()
        total_recovered = latest_data['Recovered'].sum()
        total_active = latest_data['Active Cases'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Confirmed", f"{total_confirmed:,.0f}")
        with col2:
            st.metric("Total Deaths", f"{total_deaths:,.0f}")
        with col3:
            st.metric("Total Recovered", f"{total_recovered:,.0f}")
        with col4:
            st.metric("Active Cases", f"{total_active:,.0f}")
        
        st.markdown("---")
    
    def display_eda_section(self, data):
        """Display EDA section"""
        st.header("🔍 Exploratory Data Analysis")
        
        tab1, tab2 = st.tabs(["Descriptive Statistics", "Data Distributions"])
        
        with tab1:
            self.display_descriptive_stats(data)
        with tab2:
            self.display_distributions(data)
    
    def display_descriptive_stats(self, data):
        """Display descriptive statistics"""
        st.subheader("Descriptive Statistics")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        stats_df = data[numeric_cols].describe()
        st.dataframe(stats_df.style.format("{:,.2f}"))
        
        # Correlation matrix using Streamlit's built-in
        st.subheader("Correlation Matrix")
        correlation_matrix = data[numeric_cols].corr()
        st.dataframe(correlation_matrix.style.background_gradient(cmap='Blues'))
    
    def display_distributions(self, data):
        """Display data distributions"""
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confirmed Cases Distribution**")
            st.bar_chart(data.groupby('Country/Region')['Confirmed'].max())
        
        with col2:
            st.write("**Death Rate Statistics**")
            death_stats = data.groupby('Country/Region')['Death Rate'].agg(['mean', 'max']).round(2)
            st.dataframe(death_stats)
    
    def display_interactive_charts(self, data):
        """Display interactive charts using Streamlit native charts"""
        st.header("📈 Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Time Series", "Country Comparison", "Metrics Analysis"])
        
        with tab1:
            self.display_time_series_charts(data)
        with tab2:
            self.display_comparison_charts(data)
        with tab3:
            self.display_metrics_analysis(data)
    
    def display_time_series_charts(self, data):
        """Display time series charts using Streamlit"""
        st.subheader("Time Series Analysis")
        
        # Pivot data for line chart
        pivot_data = data.pivot_table(
            index='Date', 
            columns='Country/Region', 
            values=self.selected_metric,
            aggfunc='sum'
        ).fillna(0)
        
        st.line_chart(pivot_data)
        
        # Moving average
        st.subheader(f"{self.selected_metric} - Moving Average (7-day)")
        ma_data = pivot_data.rolling(window=7).mean()
        st.line_chart(ma_data)
    
    def display_comparison_charts(self, data):
        """Display comparison charts"""
        st.subheader("Country Comparison")
        
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Countries by Confirmed Cases**")
            top_countries = latest_data.nlargest(10, 'Confirmed')[['Country/Region', 'Confirmed']]
            st.bar_chart(top_countries.set_index('Country/Region'))
        
        with col2:
            st.write("**Death Rate by Country**")
            death_rates = latest_data[['Country/Region', 'Death Rate']].set_index('Country/Region')
            st.bar_chart(death_rates)
    
    def display_metrics_analysis(self, data):
        """Display metrics analysis"""
        st.subheader("Metrics Correlation")
        
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        # Scatter plot using Streamlit
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confirmed vs Death Rate**")
            scatter_data = latest_data[['Confirmed', 'Death Rate']]
            st.scatter_chart(scatter_data)
        
        with col2:
            st.write("**Active Cases vs Recovery Rate**")
            scatter_data2 = latest_data[['Active Cases', 'Recovery Rate']]
            st.scatter_chart(scatter_data2)
        
        # Data table
        st.subheader("Latest Data by Country")
        st.dataframe(latest_data.style.format({
            'Confirmed': '{:,.0f}',
            'Deaths': '{:,.0f}', 
            'Recovered': '{:,.0f}',
            'Active Cases': '{:,.0f}',
            'Death Rate': '{:.2f}%',
            'Recovery Rate': '{:.2f}%'
        }))

# Run the app
if __name__ == "__main__":
    dashboard = CovidDashboard()
    dashboard.run()
