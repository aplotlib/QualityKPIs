import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quality KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- CUSTOM STYLING (CSS) ---
st.markdown("""
<style>
    /* Main app background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main {
        background-color: #0f1116;
    }
    /* Main title */
    h1 {
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
        padding-bottom: 1rem;
    }
    /* Subheaders */
    h2, h3 {
        color: #E0E0E0;
        text-align: left;
    }
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2c2f3b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #a0a4b8;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 600;
    }
    div[data-testid="stMetricDelta"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #a0a4b8;
        font-size: 0.9rem;
        padding-top: 2rem;
    }
    .footer a {
        color: #1c83e1;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING AND TRANSFORMATION ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_and_transform_data():
    """
    Connects to Google Sheets, loads the multi-year/multi-channel data,
    and transforms it into a clean, long-format DataFrame suitable for analysis.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Read the raw data block without a header
        raw_data = conn.read(worksheet=799906691, header=None)

        # --- Data Transformation Logic ---
        # 1. Identify the start of the actual data table
        header_row_index = raw_data[raw_data[1] == '2025'].index[0]
        months = raw_data.iloc[header_row_index - 1, 2:].dropna().tolist()
        
        # 2. Slice the DataFrame to get only the relevant data rows and columns
        data = raw_data.iloc[header_row_index:, 1:].reset_index(drop=True)
        data.columns = ['Category'] + months
        
        # 3. Forward-fill the year values and create a 'Channel' column
        data['Year'] = data['Category'].where(data['Category'].astype(str).str.isnumeric()).ffill().astype(int)
        data['Channel'] = data['Category'].where(~data['Category'].astype(str).str.isnumeric())
        
        # 4. Filter to keep only the channel rows and drop unnecessary columns
        data = data.dropna(subset=['Channel']).drop(columns=['Category'])
        
        # 5. "Melt" the DataFrame from wide to long format
        id_vars = ['Year', 'Channel']
        value_vars = months
        long_df = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='Month', value_name='Return Rate')
        
        # 6. Clean and convert data types
        long_df['Return Rate'] = pd.to_numeric(long_df['Return Rate'].astype(str).str.replace('%', ''), errors='coerce') / 100.0
        long_df.dropna(subset=['Return Rate'], inplace=True)
        
        # 7. Create a proper datetime object for plotting and sorting
        long_df['Date'] = pd.to_datetime(long_df['Year'].astype(str) + '-' + long_df['Month'], format='%Y-%b')
        
        return long_df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred while loading or transforming data: {e}")
        return pd.DataFrame()

# --- HELPER FUNCTIONS FOR VISUALIZATIONS ---
def create_sparkline(data_series):
    """Creates a sparkline chart for a given data series."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(data_series))),
        y=data_series,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#1c83e1', width=2),
        fillcolor='rgba(28, 131, 225, 0.2)'
    ))
    fig.update_layout(
        width=200, height=80,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=4),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

# --- MAIN DASHBOARD ---
st.title("📊 Quality Department KPI Dashboard")

df = load_and_transform_data()

if not df.empty:
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Year selector based on available data
        available_years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Year", available_years)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            """
            <div class="footer">
                <p>For questions or feedback, please <a href='mailto:alexander.popoff@vivehealth.com'>contact the Quality Dept</a>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Filter DataFrame based on selected year
    df_filtered = df[df['Year'] == selected_year]

    # --- KPI OVERVIEW ---
    st.subheader(f"Executive Summary: {selected_year} Performance")
    
    # Get latest data for each channel in the selected year
    latest_amazon = df_filtered[df_filtered['Channel'] == 'Amazon'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'Amazon'].empty else None
    latest_b2b = df_filtered[df_filtered['Channel'] == 'B2B'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'B2B'].empty else None

    col1, col2 = st.columns(2)
    with col1:
        if latest_amazon is not None:
            st.metric(
                label=f"Amazon Return Rate ({latest_amazon['Month']})",
                value=f"{latest_amazon['Return Rate']:.2%}"
            )
            st.plotly_chart(create_sparkline(df_filtered[df_filtered['Channel'] == 'Amazon']['Return Rate']), use_container_width=True)
        else:
            st.info("No Amazon data available for the selected year.")
            
    with col2:
        if latest_b2b is not None:
            st.metric(
                label=f"B2B Return Rate ({latest_b2b['Month']})",
                value=f"{latest_b2b['Return Rate']:.2%}"
            )
            st.plotly_chart(create_sparkline(df_filtered[df_filtered['Channel'] == 'B2B']['Return Rate']), use_container_width=True)
        else:
            st.info("No B2B data available for the selected year.")

    st.markdown("---")

    # --- TREND ANALYSIS ---
    st.subheader(f"Channel Performance Trends for {selected_year}")
    
    fig = px.line(
        df_filtered,
        x='Date',
        y='Return Rate',
        color='Channel',
        markers=True,
        template="plotly_dark",
        labels={'Return Rate': 'Rate', 'Date': 'Month', 'Channel': 'Sales Channel'},
        color_discrete_map={'Amazon': '#ff9900', 'B2B': '#1c83e1'}
    )
    fig.update_layout(
        yaxis_tickformat='.2%',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RAW DATA DISPLAY ---
    with st.expander("View Transformed Data Table"):
        st.dataframe(df)
else:
    st.warning("Data could not be loaded. Please check the Google Sheet is shared correctly and the format is as expected.")
