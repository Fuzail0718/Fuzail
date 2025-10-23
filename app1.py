# app.py - Complete COVID-19 EDA Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import io
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

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
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
        except Exception as e:
            st.warning(f"Could not fetch live data: {e}. Using sample data for demonstration.")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['US', 'India', 'Brazil', 'Russia', 'UK', 'France', 'Germany', 'Italy', 'Spain', 'Canada', 
                    'Japan', 'Australia', 'Mexico', 'South Korea', 'Indonesia', 'Netherlands', 'Turkey', 'Saudi Arabia']
        
        # Create sample data for each metric
        np.random.seed(42)
        
        confirmed_data = {}
        deaths_data = {}
        recovered_data = {}
        
        for country in countries:
            # Simulate COVID spread with logistic growth
            base_cases = np.random.randint(1000, 50000)
            growth_rate = np.random.uniform(0.05, 0.2)
            
            # Create more realistic growth pattern
            t = np.arange(len(dates)) / 100
            confirmed = base_cases / (1 + np.exp(-growth_rate * (t - 5)))  # Logistic function
            
            # Add multiple waves
            wave1 = 0.3 * np.sin(2 * np.pi * t / 2)  # First wave
            wave2 = 0.2 * np.sin(2 * np.pi * (t - 1) / 3)  # Second wave
            confirmed = confirmed * (1 + wave1 + wave2)
            
            deaths = confirmed * np.random.uniform(0.01, 0.03)
            recovered = confirmed * np.random.uniform(0.6, 0.85)
            
            # Add some noise
            confirmed += np.random.normal(0, confirmed * 0.05, len(dates))
            deaths += np.random.normal(0, deaths * 0.1, len(dates))
            recovered += np.random.normal(0, recovered * 0.1, len(dates))
            
            confirmed_data[country] = np.maximum(confirmed, 0)
            deaths_data[country] = np.maximum(deaths, 0)
            recovered_data[country] = np.maximum(recovered, 0)
        
        # Create DataFrames
        confirmed_df = pd.DataFrame(confirmed_data, index=dates)
        deaths_df = pd.DataFrame(deaths_data, index=dates)
        recovered_df = pd.DataFrame(recovered_data, index=dates)
        
        # Reset index to have Date as a column for consistency
        confirmed_df = confirmed_df.reset_index().rename(columns={'index': 'Date'})
        deaths_df = deaths_df.reset_index().rename(columns={'index': 'Date'})
        recovered_df = recovered_df.reset_index().rename(columns={'index': 'Date'})
        
        return confirmed_df, deaths_df, recovered_df
    
    def process_data(self, confirmed_df, deaths_df, recovered_df):
        """Process and clean the COVID-19 data"""
        try:
            # Check if data is in geographical format or time series format
            if 'Country/Region' in confirmed_df.columns:
                # Process geographical data format
                return self._process_geographical_data(confirmed_df, deaths_df, recovered_df)
            else:
                # Process time series format
                return self._process_timeseries_data(confirmed_df, deaths_df, recovered_df)
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return self._create_processed_sample_data()
    
    def _process_geographical_data(self, confirmed_df, deaths_df, recovered_df):
        """Process geographical data format"""
        # Melt data from wide to long format
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
        
        # Handle missing values and negative cases
        merged_df.fillna(0, inplace=True)
        merged_df['Active Cases'] = merged_df['Active Cases'].clip(lower=0)
        
        return merged_df
    
    def _process_timeseries_data(self, confirmed_df, deaths_df, recovered_df):
        """Process time series format data"""
        processed_data = []
        
        for country in confirmed_df.columns:
            if country == 'Date':
                continue
                
            for i in range(len(confirmed_df)):
                date = confirmed_df['Date'].iloc[i]
                confirmed = confirmed_df[country].iloc[i]
                deaths = deaths_df[country].iloc[i] if country in deaths_df.columns else 0
                recovered = recovered_df[country].iloc[i] if country in recovered_df.columns else 0
                
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
        # Aggregate by country (summing provinces/states)
        melted = melted.groupby(['Country/Region', 'Date'])[metric_name].sum().reset_index()
        
        return melted
    
    def _create_processed_sample_data(self):
        """Create processed sample data as fallback"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['US', 'India', 'Brazil', 'Russia', 'UK', 'France', 'Germany', 'Italy', 'Spain', 'Canada']
        
        np.random.seed(42)
        processed_data = []
        
        for country in countries:
            base_cases = np.random.randint(10000, 100000)
            growth_rate = np.random.uniform(0.1, 0.3)
            
            for i, date in enumerate(dates):
                confirmed = base_cases * np.exp(growth_rate * i / 100)
                deaths = confirmed * np.random.uniform(0.01, 0.05)
                recovered = confirmed * np.random.uniform(0.7, 0.9)
                active = confirmed - deaths - recovered
                
                # Add noise
                confirmed += np.random.normal(0, confirmed * 0.1)
                deaths += np.random.normal(0, deaths * 0.1)
                recovered += np.random.normal(0, recovered * 0.1)
                
                processed_data.append({
                    'Country/Region': country,
                    'Date': date,
                    'Confirmed': max(confirmed, 0),
                    'Deaths': max(deaths, 0),
                    'Recovered': max(recovered, 0),
                    'Active Cases': max(active, 0),
                    'Death Rate': (deaths / confirmed * 100) if confirmed > 0 else 0,
                    'Recovery Rate': (recovered / confirmed * 100) if confirmed > 0 else 0
                })
        
        return pd.DataFrame(processed_data)

class CovidDashboard:
    def __init__(self):
        self.loader = CovidDataLoader()
        self.data = None
        
    def load_and_process_data(self):
        """Load and process COVID-19 data"""
        with st.spinner('Loading COVID-19 data... This may take a few moments.'):
            confirmed_df, deaths_df, recovered_df = self.loader.load_data()
            self.data = self.loader.process_data(confirmed_df, deaths_df, recovered_df)
            
    def run(self):
        """Run the dashboard application"""
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 Exploratory Data Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        An interactive dashboard for exploring COVID-19 data with comprehensive analysis and visualization
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        if self.data is None:
            self.load_and_process_data()
        
        # Sidebar
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
            max_value=max_date,
            key="date_range"
        )
        
        # Country filter
        st.sidebar.subheader("Country Selection")
        countries = sorted(self.data['Country/Region'].unique())
        selected_countries = st.sidebar.multiselect(
            "Choose countries to analyze",
            countries,
            default=countries[:8] if len(countries) > 8 else countries,
            key="countries"
        )
        
        # Metric selection
        st.sidebar.subheader("Analysis Metric")
        self.selected_metric = st.sidebar.selectbox(
            "Primary metric for analysis",
            ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate'],
            key="metric"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "💡 **Tip**: Use the interactive charts to zoom, hover for details, "
            "and click on legend items to toggle countries."
        )
    
    def apply_filters(self):
        """Apply sidebar filters to data"""
        date_range = st.session_state.get("date_range", [self.data['Date'].min(), self.data['Date'].max()])
        selected_countries = st.session_state.get("countries", self.data['Country/Region'].unique()[:8])
        
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
            st.metric(
                "Total Confirmed Cases", 
                f"{total_confirmed:,.0f}",
                help="Sum of confirmed cases across selected countries"
            )
        with col2:
            st.metric(
                "Total Deaths", 
                f"{total_deaths:,.0f}",
                delta=f"{avg_death_rate:.1f}% avg rate",
                help="Sum of deaths across selected countries"
            )
        with col3:
            st.metric(
                "Total Recovered", 
                f"{total_recovered:,.0f}",
                help="Sum of recovered cases across selected countries"
            )
        with col4:
            st.metric(
                "Active Cases", 
                f"{total_active:,.0f}",
                help="Current active cases across selected countries"
            )
        
        st.markdown("---")
    
    def display_eda_section(self, data):
        """Display Exploratory Data Analysis section"""
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Descriptive Statistics", "🔗 Correlation Analysis", 
            "📈 Time Series Decomposition", "📊 Data Distributions"
        ])
        
        with tab1:
            self.display_descriptive_stats(data)
        
        with tab2:
            self.display_correlation_analysis(data)
        
        with tab3:
            self.display_time_series_decomposition(data)
        
        with tab4:
            self.display_distributions(data)
    
    def display_descriptive_stats(self, data):
        """Display descriptive statistics"""
        st.subheader("Descriptive Statistics")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        stats_df = data[numeric_cols].describe()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
        
        with col2:
            st.info("""
            **Statistics Guide:**
            - **count**: Number of observations
            - **mean**: Average value
            - **std**: Standard deviation
            - **min/max**: Range of values
            - **25/50/75%**: Percentiles
            """)
        
        # Additional statistics by country
        st.subheader("Country-wise Statistics")
        country_stats = data.groupby('Country/Region')[numeric_cols].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(country_stats, use_container_width=True)
    
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
        st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)
        
        # Correlation interpretation
        st.info("""
        **Correlation Interpretation:**
        - **+1.0**: Perfect positive correlation
        - **+0.7 to +1.0**: Strong positive correlation  
        - **+0.3 to +0.7**: Moderate positive correlation
        - **-0.3 to +0.3**: Weak or no correlation
        - **-0.7 to -0.3**: Moderate negative correlation
        - **-1.0 to -0.7**: Strong negative correlation
        """)
    
    def display_time_series_decomposition(self, data):
        """Display time series decomposition"""
        st.subheader("Time Series Decomposition")
        
        if len(data) < 30:
            st.warning("Need at least 30 days of data for meaningful decomposition")
            return
            
        # Prepare data for decomposition - use aggregated data across countries
        country_data = data.groupby('Date')[self.selected_metric].sum().reset_index()
        country_data = country_data.set_index('Date')
        
        # Ensure we have a regular time series
        country_data = country_data.asfreq('D').fillna(method='ffill')
        
        if len(country_data) < 30:
            st.warning("Not enough data points after processing for decomposition")
            return
            
        # Perform decomposition
        try:
            with st.spinner('Performing time series decomposition...'):
                decomposition = seasonal_decompose(country_data[self.selected_metric], model='additive', period=30)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component'],
                vertical_spacing=0.08
            )
            
            # Original
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.observed, name='Original', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.trend, name='Trend', line=dict(color='red')),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.seasonal, name='Seasonal', line=dict(color='green')),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=country_data.index, y=decomposition.resid, name='Residual', line=dict(color='orange')),
                row=4, col=1
            )
            
            fig.update_layout(
                height=800, 
                title_text=f"Time Series Decomposition - {self.selected_metric}",
                showlegend=False
            )
            
            # Update y-axis titles
            fig.update_yaxes(title_text="Original", row=1, col=1)
            fig.update_yaxes(title_text="Trend", row=2, col=1)
            fig.update_yaxes(title_text="Seasonal", row=3, col=1)
            fig.update_yaxes(title_text="Residual", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.info("""
            **Decomposition Interpretation:**
            - **Trend**: Long-term progression of the series
            - **Seasonal**: Regular periodic fluctuations
            - **Residual**: Irregular component (noise)
            """)
            
        except Exception as e:
            st.error(f"Decomposition error: {str(e)}")
            st.info("This might occur with insufficient data or missing values in the time series.")
    
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
            fig_hist.update_layout(
                xaxis_title="Confirmed Cases",
                yaxis_title="Frequency"
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
            fig_box.update_layout(
                yaxis_title="Death Rate (%)"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Additional distribution - Recovery Rate
        fig_recovery = px.histogram(
            data,
            x='Recovery Rate',
            title="Distribution of Recovery Rates",
            nbins=50,
            color_discrete_sequence=['#2ca02c']
        )
        fig_recovery.update_layout(
            xaxis_title="Recovery Rate (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_recovery, use_container_width=True)
    
    def display_interactive_charts(self, data):
        """Display interactive charts section"""
        st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "⏰ Time Series Analysis", "🌍 Geographical Distribution", 
            "📊 Comparative Analysis", "🔬 Advanced Analytics"
        ])
        
        with tab1:
            self.display_time_series_charts(data)
        
        with tab2:
            self.display_geographical_charts(data)
        
        with tab3:
            self.display_comparative_charts(data)
        
        with tab4:
            self.display_advanced_analytics(data)
    
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
            height=500,
            labels={self.selected_metric: self.selected_metric, 'Date': 'Date'}
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
        window = st.slider("Moving Average Window (days)", 7, 90, 30, key="ma_window")
        
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
            height=400,
            labels={f'{self.selected_metric}_MA': f'{self.selected_metric} ({window}-day MA)'}
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    
    def display_geographical_charts(self, data):
        """Display geographical distribution charts"""
        st.subheader("Geographical Distribution")
        
        # Aggregate data by country (latest data)
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            # Bar chart of top countries
            top_countries = latest_data.nlargest(15, 'Confirmed')
            fig_bar = px.bar(
                top_countries,
                x='Confirmed',
                y='Country/Region',
                orientation='h',
                title="Top Countries by Confirmed Cases",
                color='Confirmed',
                color_continuous_scale="Blues",
                hover_data=['Deaths', 'Death Rate']
            )
            fig_bar.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_comparative_charts(self, data):
        """Display comparative analysis charts"""
        st.subheader("Comparative Analysis")
        
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with hover information
            fig_scatter = px.scatter(
                latest_data,
                x='Confirmed',
                y='Death Rate',
                size='Deaths',
                color='Country/Region',
                hover_name='Country/Region',
                hover_data=['Recovered', 'Active Cases', 'Recovery Rate'],
                title="Confirmed Cases vs Death Rate",
                size_max=60,
                labels={'Confirmed': 'Confirmed Cases', 'Death Rate': 'Death Rate (%)'}
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
                color_continuous_scale="RdYlBu_r",
                labels={'Recovered': 'Recovered Cases', 'Active Cases': 'Active Cases'}
            )
            fig_bubble.update_layout(showlegend=False)
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Additional comparison: Death Rate vs Recovery Rate
        fig_comparison = px.scatter(
            latest_data,
            x='Death Rate',
            y='Recovery Rate',
            size='Confirmed',
            color='Country/Region',
            hover_name='Country/Region',
            title="Death Rate vs Recovery Rate Comparison",
            labels={'Death Rate': 'Death Rate (%)', 'Recovery Rate': 'Recovery Rate (%)'}
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    def display_advanced_analytics(self, data):
        """Display advanced analytics charts"""
        st.subheader("Advanced Analytics")
        
        # Growth rate analysis
        st.subheader("Growth Rate Analysis")
        
        # Calculate daily growth rates
        growth_data = data.copy()
        growth_data = growth_data.sort_values(['Country/Region', 'Date'])
        growth_data['Daily Growth Rate'] = growth_data.groupby('Country/Region')['Confirmed'].pct_change() * 100
        
        # Remove infinite values and outliers
        growth_data = growth_data[np.isfinite(growth_data['Daily Growth Rate'])]
        growth_data = growth_data[growth_data['Daily Growth Rate'].abs() < 100]  # Remove extreme outliers
        
        fig_growth = px.line(
            growth_data,
            x='Date',
            y='Daily Growth Rate',
            color='Country/Region',
            title="Daily Growth Rate of Confirmed Cases",
            height=400,
            labels={'Daily Growth Rate': 'Daily Growth Rate (%)'}
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Correlation scatter matrix
        st.subheader("Multi-variable Correlation Analysis")
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active Cases', 'Death Rate', 'Recovery Rate']
        latest_data = data.sort_values('Date').groupby('Country/Region').last().reset_index()
        
        # Select only 4 variables for clarity
        selected_vars = numeric_cols[:4]
        
        fig_matrix = px.scatter_matrix(
            latest_data,
            dimensions=selected_vars,
            color='Country/Region',
            title="Scatter Matrix of COVID-19 Metrics",
            height=600
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(latest_data[selected_vars].describe(), use_container_width=True)

# Run the dashboard
if __name__ == "__main__":
    try:
        dashboard = CovidDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page or check your internet connection.")
