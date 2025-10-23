import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.data_loader import CovidDataLoader

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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CovidDashboard:
    def __init__(self):
        self.loader = CovidDataLoader()
        self.data = None
        
    def load_and_process_data(self):
        """Load and process COVID-19 data"""
        with st.spinner('Loading COVID-19 data...'):
            confirmed_df, deaths_df, recovered_df = self.loader.load_data()
            self.data = self.loader.process_data(confirmed_df, deaths_df, recovered_df)
            
    def run(self):
        """Run the dashboard application"""
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 Exploratory Data Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Load data
        if self.data is None:
            self.load_and_process_data()
        
        # Sidebar
        st.sidebar.title("Navigation & Filters")
        
        # Date range filter
        min_date = self.data['Date'].min()
        max_date = self.data['Date'].max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Country filter
        countries = sorted(self.data['Country/Region'].unique())
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            countries,
            default=countries[:5] if len(countries) > 5 else countries
        )
        
        # Metric selection
        metric = st.sidebar.selectbox(
            "Select Metric",
            ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        )
        
        # Filter data based on selections
        filtered_data = self.data[
            (self.data['Date'] >= pd.to_datetime(date_range[0])) &
            (self.data['Date'] <= pd.to_datetime(date_range[1])) &
            (self.data['Country/Region'].isin(selected_countries))
        ]
        
        # Main content
        self.display_overview(filtered_data)
        self.display_eda_section(filtered_data, metric)
        self.display_interactive_charts(filtered_data, metric)
        
    def display_overview(self, data):
        """Display overview metrics"""
        st.header("📊 Overview Metrics")
        
        # Calculate key metrics
        total_confirmed = data['Confirmed'].max()
        total_deaths = data['Deaths'].max()
        total_recovered = data['Recovered'].max()
        avg_death_rate = data['Death Rate'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Confirmed Cases", f"{total_confirmed:,.0f}")
        with col2:
            st.metric("Total Deaths", f"{total_deaths:,.0f}")
        with col3:
            st.metric("Total Recovered", f"{total_recovered:,.0f}")
        with col4:
            st.metric("Average Death Rate", f"{avg_death_rate:.2f}%")
    
    def display_eda_section(self, data, metric):
        """Display Exploratory Data Analysis section"""
        st.header("🔍 Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Descriptive Statistics", "Correlation Analysis", 
            "Time Series Decomposition", "Data Distributions"
        ])
        
        with tab1:
            self.display_descriptive_stats(data)
        
        with tab2:
            self.display_correlation_analysis(data)
        
        with tab3:
            self.display_time_series_decomposition(data, metric)
        
        with tab4:
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
        st.subheader("Correlation Analysis")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        correlation_matrix = data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation values
        st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu_r'))
    
    def display_time_series_decomposition(self, data, metric):
        """Display time series decomposition"""
        st.subheader("Time Series Decomposition")
        
        if len(data) < 30:
            st.warning("Need at least 30 days of data for decomposition")
            return
            
        # Prepare data for decomposition
        country_data = data.groupby('Date')[metric].sum().reset_index()
        country_data = country_data.set_index('Date')
        
        # Perform decomposition
        try:
            decomposition = seasonal_decompose(country_data[metric], model='additive', period=30)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual']
            )
            
            # Original
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.observed, name='Original'),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.trend, name='Trend'),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.seasonal, name='Seasonal'),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.resid, name='Residual'),
                row=4, col=1
            )
            
            fig.update_layout(height=800, title_text="Time Series Decomposition")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Decomposition error: {e}")
    
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
                nbins=50
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                data,
                y='Death Rate',
                title="Distribution of Death Rates"
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    def display_interactive_charts(self, data, metric):
        """Display interactive charts section"""
        st.header("📈 Interactive Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Time Series Analysis", "Geographical Distribution", 
            "Comparative Analysis", "Advanced Analytics"
        ])
        
        with tab1:
            self.display_time_series_charts(data, metric)
        
        with tab2:
            self.display_geographical_charts(data)
        
        with tab3:
            self.display_comparative_charts(data)
        
        with tab4:
            self.display_advanced_analytics(data)
    
    def display_time_series_charts(self, data, metric):
        """Display time series charts with range selector"""
        st.subheader("Time Series Analysis")
        
        # Time series line chart
        fig = px.line(
            data,
            x='Date',
            y=metric,
            color='Country/Region',
            title=f"{metric} Cases Over Time",
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
                        dict(step="all")
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
            ma_data.loc[country_mask, f'{metric}_MA'] = (
                ma_data[country_mask][metric].rolling(window=window).mean()
            )
        
        fig_ma = px.line(
            ma_data,
            x='Date',
            y=f'{metric}_MA',
            color='Country/Region',
            title=f"{metric} - {window}-day Moving Average",
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
            hover_data=['Deaths', 'Recovered', 'Death Rate'],
            title="Global COVID-19 Confirmed Cases Distribution",
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Bar chart of top countries
        top_countries = latest_data.nlargest(10, 'Confirmed')
        fig_bar = px.bar(
            top_countries,
            x='Country/Region',
            y='Confirmed',
            title="Top 10 Countries by Confirmed Cases",
            color='Confirmed',
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_comparative_charts(self, data):
        """Display comparative analysis charts"""
        st.subheader("Comparative Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with hover information
            latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
            
            fig_scatter = px.scatter(
                latest_data,
                x='Confirmed',
                y='Death Rate',
                size='Deaths',
                color='Country/Region',
                hover_name='Country/Region',
                hover_data=['Recovered', 'Active Cases'],
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
    
    def display_advanced_analytics(self, data):
        """Display advanced analytics charts"""
        st.subheader("Advanced Analytics")
        
        # Growth rate analysis
        st.subheader("Growth Rate Analysis")
        
        # Calculate daily growth rates
        growth_data = data.copy()
        growth_data = growth_data.sort_values(['Country/Region', 'Date'])
        growth_data['Daily Growth Rate'] = growth_data.groupby('Country/Region')['Confirmed'].pct_change() * 100
        
        fig_growth = px.line(
            growth_data,
            x='Date',
            y='Daily Growth Rate',
            color='Country/Region',
            title="Daily Growth Rate of Confirmed Cases",
            height=400
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Correlation scatter matrix
        st.subheader("Multi-variable Correlation Analysis")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        fig_matrix = px.scatter_matrix(
            latest_data,
            dimensions=numeric_cols[:4],
            color='Country/Region',
            title="Scatter Matrix of COVID-19 Metrics",
            height=600
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = CovidDashboard()
    dashboard.run()
