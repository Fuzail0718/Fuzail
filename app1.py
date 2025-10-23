# app.py - COVID-19 EDA Dashboard (Simplified Dependencies)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
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
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
        
    def create_sample_data(self):
        """Create comprehensive sample COVID-19 data"""
        st.info("📊 Using sample COVID-19 data for demonstration")
        
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['United States', 'India', 'Brazil', 'Russia', 'United Kingdom', 
                    'France', 'Germany', 'Italy', 'Spain', 'Canada', 
                    'Japan', 'Australia', 'Mexico', 'South Korea', 'Indonesia']
        
        np.random.seed(42)
        processed_data = []
        
        for country in countries:
            # Create realistic COVID growth patterns
            base_cases = np.random.randint(50000, 200000)
            
            # Simulate multiple waves
            t = np.arange(len(dates)) / 100
            
            # Main logistic growth
            main_growth = base_cases / (1 + np.exp(-0.15 * (t - 5)))
            
            # Add seasonal waves
            wave1 = 0.4 * np.sin(2 * np.pi * t / 2) * main_growth
            wave2 = 0.3 * np.sin(2 * np.pi * (t - 1.5) / 3) * main_growth
            wave3 = 0.2 * np.sin(2 * np.pi * (t - 3) / 4) * main_growth
            
            confirmed = main_growth + wave1 + wave2 + wave3
            deaths = confirmed * np.random.uniform(0.015, 0.035)
            recovered = confirmed * np.random.uniform(0.65, 0.85)
            
            # Add realistic noise
            confirmed_noise = np.random.normal(0, confirmed * 0.08, len(dates))
            deaths_noise = np.random.normal(0, deaths * 0.12, len(dates))
            recovered_noise = np.random.normal(0, recovered * 0.1, len(dates))
            
            confirmed = np.maximum(confirmed + confirmed_noise, 0)
            deaths = np.maximum(deaths + deaths_noise, 0)
            recovered = np.maximum(recovered + recovered_noise, 0)
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
        with st.spinner('Loading COVID-19 data...'):
            self.data = self.loader.create_sample_data()
            
    def run(self):
        """Run the dashboard application"""
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 Exploratory Data Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; color: #666;'>
        <p style='font-size: 1.1rem;'>An interactive dashboard for exploring COVID-19 data with comprehensive analysis and visualization</p>
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
        """Setup sidebar filters and controls"""
        st.sidebar.title("🔧 Navigation & Filters")
        st.sidebar.markdown("---")
        
        # Date range filter
        min_date = self.data['Date'].min()
        max_date = self.data['Date'].max()
        
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Select analysis period",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Country filter
        st.sidebar.subheader("Country Selection")
        countries = sorted(self.data['Country/Region'].unique())
        selected_countries = st.sidebar.multiselect(
            "Choose countries to analyze",
            countries,
            default=countries[:6]
        )
        
        # Metric selection
        st.sidebar.subheader("Analysis Metric")
        self.selected_metric = st.sidebar.selectbox(
            "Primary metric for analysis",
            ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Tip**: Hover over charts for detailed information and use the range selector to zoom in on specific time periods.")
    
    def apply_filters(self):
        """Apply sidebar filters to data"""
        date_range = st.session_state.get("date_range", [self.data['Date'].min(), self.data['Date'].max()])
        selected_countries = st.session_state.get("countries", self.data['Country/Region'].unique()[:6])
        
        if len(date_range) == 2:
            filtered_data = self.data[
                (self.data['Date'] >= pd.to_datetime(date_range[0])) &
                (self.data['Date'] <= pd.to_datetime(date_range[1])) &
                (self.data['Country/Region'].isin(selected_countries))
            ]
        else:
            filtered_data = self.data[self.data['Country/Region'].isin(selected_countries)]
        
        return filtered_data
    
    def display_overview(self, data):
        """Display overview metrics"""
        st.markdown('<h2 class="section-header">📊 Overview Metrics</h2>', unsafe_allow_html=True)
        
        # Calculate key metrics using the latest data for each country
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        total_confirmed = latest_data['Confirmed'].sum()
        total_deaths = latest_data['Deaths'].sum()
        total_recovered = latest_data['Recovered'].sum()
        total_active = latest_data['Active Cases'].sum()
        avg_death_rate = latest_data['Death Rate'].mean()
        
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
        """Display Exploratory Data Analysis section"""
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            "📋 Descriptive Statistics", "🔗 Correlation Analysis", "📊 Data Distributions"
        ])
        
        with tab1:
            self.display_descriptive_stats(data)
        
        with tab2:
            self.display_correlation_analysis(data)
        
        with tab3:
            self.display_distributions(data)
    
    def display_descriptive_stats(self, data):
        """Display descriptive statistics"""
        st.subheader("Descriptive Statistics")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        stats_df = data[numeric_cols].describe()
        
        st.dataframe(stats_df.style.format("{:,.2f}"))
        
        # Additional statistics by country
        st.subheader("Statistics by Country")
        country_stats = data.groupby('Country/Region')[numeric_cols].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(country_stats)
    
    def display_correlation_analysis(self, data):
        """Display correlation analysis"""
        st.subheader("Correlation Matrix")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        correlation_matrix = data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation values
        st.subheader("Correlation Values")
        st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))
    
    def display_distributions(self, data):
        """Display data distributions"""
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                data, 
                x='Confirmed',
                title="Distribution of Confirmed Cases",
                nbins=50,
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                data,
                y='Death Rate',
                title="Distribution of Death Rates",
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Recovery Rate distribution
        fig_recovery = px.histogram(
            data,
            x='Recovery Rate',
            title="Distribution of Recovery Rates",
            nbins=50,
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_recovery, use_container_width=True)
    
    def display_interactive_charts(self, data):
        """Display interactive charts section"""
        st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            "⏰ Time Series Analysis", "🌍 Geographical Views", "📊 Comparative Analysis"
        ])
        
        with tab1:
            self.display_time_series_charts(data)
        
        with tab2:
            self.display_geographical_charts(data)
        
        with tab3:
            self.display_comparative_charts(data)
    
    def display_time_series_charts(self, data):
        """Display time series charts with range selector"""
        st.subheader("Time Series Analysis")
        
        # Time series line chart
        fig = px.line(
            data,
            x='Date',
            y=self.selected_metric,
            color='Country/Region',
            title=f"{self.selected_metric} Cases Over Time",
            height=500
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Moving average chart
        st.subheader("Moving Average Analysis")
        window = st.slider("Moving Average Window (days)", 7, 90, 30)
        
        ma_data = data.copy()
        for country in ma_data['Country/Region'].unique():
            country_mask = ma_data['Country/Region'] == country
            country_data = ma_data[country_mask].sort_values('Date')
            ma_values = country_data[self.selected_metric].rolling(window=window, min_periods=1).mean()
            ma_data.loc[country_mask, f'{self.selected_metric}_MA'] = ma_values.values
        
        fig_ma = px.line(
            ma_data,
            x='Date',
            y=f'{self.selected_metric}_MA',
            color='Country/Region',
            title=f"{self.selected_metric} - {window}-day Moving Average",
            height=400
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    
    def display_geographical_charts(self, data):
        """Display geographical distribution charts"""
        st.subheader("Geographical Distribution")
        
        # Aggregate data by country (latest data)
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        # World map
        fig_map = px.choropleth(
            latest_data,
            locations='Country/Region',
            locationmode='country names',
            color='Confirmed',
            hover_name='Country/Region',
            hover_data=['Deaths', 'Recovered', 'Death Rate', 'Recovery Rate'],
            title="Global COVID-19 Confirmed Cases Distribution",
            color_continuous_scale="Viridis",
            projection="natural earth"
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Bar chart of top countries
        st.subheader("Top Countries Analysis")
        top_countries = latest_data.nlargest(10, 'Confirmed')
        fig_bar = px.bar(
            top_countries,
            x='Confirmed',
            y='Country/Region',
            orientation='h',
            title="Top 10 Countries by Confirmed Cases",
            color='Confirmed',
            color_continuous_scale="Blues",
            hover_data=['Deaths', 'Death Rate']
        )
        fig_bar.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_comparative_charts(self, data):
        """Display comparative analysis charts"""
        st.subheader("Comparative Analysis")
        
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig_scatter = px.scatter(
                latest_data,
                x='Confirmed',
                y='Death Rate',
                size='Deaths',
                color='Country/Region',
                hover_name='Country/Region',
                hover_data=['Recovered', 'Active Cases', 'Recovery Rate'],
                title="Confirmed Cases vs Death Rate",
                size_max=60
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Bubble chart
            fig_bubble = px.scatter(
                latest_data,
                x='Recovered',
                y='Active Cases',
                size='Confirmed',
                color='Death Rate',
                hover_name='Country/Region',
                title="Recovery vs Active Cases Analysis",
                color_continuous_scale="RdYlBu_r"
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("Growth Rate Analysis")
        
        growth_data = data.copy()
        growth_data = growth_data.sort_values(['Country/Region', 'Date'])
        growth_data['Daily Growth Rate'] = growth_data.groupby('Country/Region')['Confirmed'].pct_change() * 100
        
        # Remove outliers
        growth_data = growth_data[np.isfinite(growth_data['Daily Growth Rate'])]
        growth_data = growth_data[growth_data['Daily Growth Rate'].abs() < 100]
        
        fig_growth = px.line(
            growth_data,
            x='Date',
            y='Daily Growth Rate',
            color='Country/Region',
            title="Daily Growth Rate of Confirmed Cases",
            height=400
        )
        st.plotly_chart(fig_growth, use_container_width=True)

# Run the dashboard
if __name__ == "__main__":
    try:
        dashboard = CovidDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page to try again.")
