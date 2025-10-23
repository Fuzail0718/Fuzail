# COVID-19 EDA Dashboard - Streamlit Cloud Compatible
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
</style>
""", unsafe_allow_html=True)


class CovidDataLoader:
    def create_sample_data(self):
        """Create comprehensive sample COVID-19 data"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        countries = ['United States', 'India', 'Brazil', 'Russia', 'United Kingdom',
                     'France', 'Germany', 'Italy', 'Spain', 'Canada']

        np.random.seed(42)
        processed_data = []

        for country in countries:
            base_cases = np.random.randint(50000, 200000)
            t = np.arange(len(dates)) / 100

            main_growth = base_cases / (1 + np.exp(-0.15 * (t - 5)))
            wave1 = 0.4 * np.sin(2 * np.pi * t / 2) * main_growth
            wave2 = 0.3 * np.sin(2 * np.pi * (t - 1.5) / 3) * main_growth

            confirmed = main_growth + wave1 + wave2
            deaths = confirmed * np.random.uniform(0.015, 0.035)
            recovered = confirmed * np.random.uniform(0.65, 0.85)

            confirmed = np.maximum(
                confirmed + np.random.normal(0, confirmed * 0.08, len(dates)), 0)
            deaths = np.maximum(
                deaths + np.random.normal(0, deaths * 0.12, len(dates)), 0)
            recovered = np.maximum(
                recovered + np.random.normal(0, recovered * 0.1, len(dates)), 0)
            active = np.maximum(confirmed - deaths - recovered, 0)

            for i, date in enumerate(dates):
                death_rate = (deaths[i] / confirmed[i] *
                              100) if confirmed[i] > 0 else 0
                recovery_rate = (
                    recovered[i] / confirmed[i] * 100) if confirmed[i] > 0 else 0

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
        st.markdown(
            '<h1 class="main-header">🦠 COVID-19 EDA Dashboard</h1>', unsafe_allow_html=True)

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
            ['Confirmed', 'Deaths', 'Recovered',
                'Active Cases', 'Death Rate', 'Recovery Rate']
        )

        st.sidebar.markdown("---")
        st.sidebar.info("💡 Hover over charts for details")

    def apply_filters(self):
        """Apply filters to data"""
        date_range = st.session_state.get(
            "date_range", [self.data['Date'].min(), self.data['Date'].max()])
        selected_countries = st.session_state.get(
            "countries", self.data['Country/Region'].unique()[:5])

        if len(date_range) == 2:
            filtered_data = self.data[
                (self.data['Date'] >= pd.to_datetime(date_range[0])) &
                (self.data['Date'] <= pd.to_datetime(date_range[1])) &
                (self.data['Country/Region'].isin(selected_countries))
            ]
        else:
            filtered_data = self.data[self.data['Country/Region'].isin(
                selected_countries)]

        return filtered_data

    def display_overview(self, data):
        """Display overview metrics"""
        st.markdown(
            '<h2 class="section-header">📊 Overview Metrics</h2>', unsafe_allow_html=True)

        latest_data = data.sort_values('Date').groupby(
            'Country/Region').last().reset_index()

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
        st.markdown(
            '<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(
            ["Statistics", "Correlations", "Distributions"])

        with tab1:
            self.display_descriptive_stats(data)
        with tab2:
            self.display_correlation_analysis(data)
        with tab3:
            self.display_distributions(data)

    def display_descriptive_stats(self, data):
        """Display descriptive statistics"""
        st.subheader("Descriptive Statistics")

        numeric_cols = ['Confirmed', 'Deaths', 'Recovered',
                        'Active Cases', 'Death Rate', 'Recovery Rate']
        stats_df = data[numeric_cols].describe()
        st.dataframe(stats_df.style.format("{:,.2f}"))

    def display_correlation_analysis(self, data):
        """Display correlation analysis"""
        st.subheader("Correlation Analysis")

        numeric_cols = ['Confirmed', 'Deaths', 'Recovered',
                        'Active Cases', 'Death Rate', 'Recovery Rate']
        correlation_matrix = data[numeric_cols].corr()

        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_distributions(self, data):
        """Display data distributions"""
        st.subheader("Data Distributions")

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                data, x='Confirmed', title="Confirmed Cases Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(data, y='Death Rate',
                             title="Death Rate Distribution")
            st.plotly_chart(fig_box, use_container_width=True)

    def display_interactive_charts(self, data):
        """Display interactive charts"""
        st.markdown(
            '<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(
            ["Time Series", "Geographical", "Comparative"])

        with tab1:
            self.display_time_series_charts(data)
        with tab2:
            self.display_geographical_charts(data)
        with tab3:
            self.display_comparative_charts(data)

    def display_time_series_charts(self, data):
        """Display time series charts"""
        st.subheader("Time Series Analysis")

        fig = px.line(
            data,
            x='Date',
            y=self.selected_metric,
            color='Country/Region',
            title=f"{self.selected_metric} Over Time",
            height=500
        )

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6m", step="month",
                             stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True)
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_geographical_charts(self, data):
        """Display geographical charts"""
        st.subheader("Geographical Distribution")

        latest_data = data.sort_values('Date').groupby(
            'Country/Region').last().reset_index()

        fig_map = px.choropleth(
            latest_data,
            locations='Country/Region',
            locationmode='country names',
            color='Confirmed',
            hover_name='Country/Region',
            title="Global COVID-19 Cases",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    def display_comparative_charts(self, data):
        """Display comparative charts"""
        st.subheader("Comparative Analysis")

        latest_data = data.sort_values('Date').groupby(
            'Country/Region').last().reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = px.scatter(
                latest_data,
                x='Confirmed',
                y='Death Rate',
                size='Deaths',
                color='Country/Region',
                title="Confirmed vs Death Rate"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                latest_data.nlargest(10, 'Confirmed'),
                x='Confirmed',
                y='Country/Region',
                title="Top 10 Countries",
                orientation='h'
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# Run the app
if __name__ == "__main__":
    dashboard = CovidDashboard()
    dashboard.run()
