import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quality KPI Dashboard",
    page_icon="�",
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
    Connects to Google Sheets, loads the multi-metric/multi-year data,
    and transforms it into a clean, long-format DataFrame suitable for analysis.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        raw_data = conn.read(worksheet=799906691, header=None)

        # --- Data Transformation Logic ---
        all_metrics_df = pd.DataFrame()
        
        # Find the rows where a new metric block starts (non-empty in column C)
        metric_start_indices = raw_data[raw_data[2].notna()].index

        for i in range(len(metric_start_indices)):
            # Determine the start and end of the current metric block
            start_index = metric_start_indices[i]
            end_index = metric_start_indices[i+1] if (i+1) < len(metric_start_indices) else len(raw_data)
            
            metric_block = raw_data.iloc[start_index:end_index]
            metric_name = metric_block.iloc[0, 2]
            
            # Find the header row for months within this block
            month_header_row = metric_block[metric_block[4] == 'Jan'].index[0]
            months = metric_block.loc[month_header_row, 5:].dropna().tolist()
            
            # Get the data part of the block
            data_start_row = month_header_row + 1
            data = metric_block.loc[data_start_row:, 3:].reset_index(drop=True)
            data.columns = ['Year', 'Channel'] + months
            
            data['Year'] = data['Year'].ffill().astype(int)
            data = data.dropna(subset=['Channel'])
            
            # Melt the block into a long format
            long_df = pd.melt(data, id_vars=['Year', 'Channel'], value_vars=months, var_name='Month', value_name='Value')
            long_df['Metric'] = metric_name
            
            all_metrics_df = pd.concat([all_metrics_df, long_df], ignore_index=True)

        # --- Final Cleaning and Calculations ---
        # Identify percentage vs. numeric metrics
        pct_metrics = [m for m in all_metrics_df['Metric'].unique() if '%' in m]
        
        # Convert values to numeric, handling percentages correctly
        def clean_value(row):
            val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
            val_num = pd.to_numeric(val_str, errors='coerce')
            if row['Metric'] in pct_metrics:
                return val_num / 100.0
            return val_num
            
        all_metrics_df['Value'] = all_metrics_df.apply(clean_value, axis=1)
        all_metrics_df.dropna(subset=['Value'], inplace=True)
        
        all_metrics_df['Date'] = pd.to_datetime(all_metrics_df['Year'].astype(str) + '-' + all_metrics_df['Month'], format='%Y-%b')
        
        all_metrics_df = all_metrics_df.sort_values(by=['Metric', 'Channel', 'Date'])
        all_metrics_df['MoM Change'] = all_metrics_df.groupby(['Metric', 'Channel'])['Value'].diff()
        all_metrics_df['YoY Change'] = all_metrics_df.groupby(['Metric', 'Channel', all_metrics_df['Date'].dt.month])['Value'].diff()
        
        return all_metrics_df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred while loading or transforming data: {e}")
        return pd.DataFrame()

# --- MAIN DASHBOARD ---
st.title("📊 Quality Department KPI Dashboard")

df = load_and_transform_data()

if not df.empty:
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Dashboard Controls")
        
        available_metrics = df['Metric'].unique()
        selected_metric = st.selectbox("Select KPI to Analyze", available_metrics)
        
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

    # --- FILTER DATA BASED ON SELECTIONS ---
    df_metric = df[df['Metric'] == selected_metric]
    df_filtered = df_metric[df_metric['Year'] == selected_year]
    
    # Determine if lower values are better (for color-coding deltas)
    lower_is_better = 'rate' in selected_metric.lower() or 'cost' in selected_metric.lower()

    # --- KPI OVERVIEW ---
    st.subheader(f"Executive Summary: {selected_metric} ({selected_year})")
    
    latest_amazon = df_filtered[df_filtered['Channel'] == 'Amazon'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'Amazon'].empty else None
    latest_b2b = df_filtered[df_filtered['Channel'] == 'B2B'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'B2B'].empty else None

    # Determine value format
    is_percent = '%' in selected_metric
    is_currency = 'cost' in selected_metric.lower()
    value_format = "{:,.2%}" if is_percent else ("${:,.2f}" if is_currency else "{:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        if latest_amazon is not None:
            st.metric(
                label=f"Amazon ({latest_amazon['Month']})",
                value=value_format.format(latest_amazon['Value']),
                delta=f"{latest_amazon['MoM Change']:.2%}" if is_percent and pd.notna(latest_amazon['MoM Change']) else None,
                delta_color="inverse" if lower_is_better else "normal"
            )
            yoy_change = latest_amazon['YoY Change']
            if pd.notna(yoy_change):
                yoy_color = "green" if (yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better) else "red"
                st.markdown(f"<p style='color:{yoy_color}; font-size: 1.1rem; font-weight: 600;'>{yoy_change:+.2%} vs. Same Month Last Year</p>", unsafe_allow_html=True)
        else:
            st.info(f"No Amazon data for {selected_year}.")
            
    with col2:
        if latest_b2b is not None:
            st.metric(
                label=f"B2B ({latest_b2b['Month']})",
                value=value_format.format(latest_b2b['Value']),
                delta=f"{latest_b2b['MoM Change']:.2%}" if is_percent and pd.notna(latest_b2b['MoM Change']) else None,
                delta_color="inverse" if lower_is_better else "normal"
            )
            yoy_change = latest_b2b['YoY Change']
            if pd.notna(yoy_change):
                yoy_color = "green" if (yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better) else "red"
                st.markdown(f"<p style='color:{yoy_color}; font-size: 1.1rem; font-weight: 600;'>{yoy_change:+.2%} vs. Same Month Last Year</p>", unsafe_allow_html=True)
        else:
            st.info(f"No B2B data for {selected_year}.")

    # --- TREND ANALYSIS ---
    st.subheader(f"Performance Trends: {selected_year} vs. {selected_year - 1}")
    
    if not df_filtered.empty:
        df_previous_year = df_metric[df_metric['Year'] == selected_year - 1]
        fig = go.Figure()

        for channel, color in [('Amazon', '#ff9900'), ('B2B', '#1c83e1')]:
            # Current year data
            channel_df = df_filtered[df_filtered['Channel'] == channel]
            if not channel_df.empty:
                fig.add_trace(go.Scatter(x=channel_df['Date'].dt.month, y=channel_df['Value'], name=f'{channel} ({selected_year})', mode='lines+markers', line=dict(color=color, width=3)))
            # Previous year data
            channel_df_prev = df_previous_year[df_previous_year['Channel'] == channel]
            if not channel_df_prev.empty:
                fig.add_trace(go.Scatter(x=channel_df_prev['Date'].dt.month, y=channel_df_prev['Value'], name=f'{channel} ({selected_year - 1})', mode='lines', line=dict(color=color, width=2, dash='dash')))

        fig.update_layout(
            title_text=f"{selected_metric} Trend",
            template="plotly_dark",
            yaxis_tickformat= ('.2%' if is_percent else (None if is_currency else ',.0f')),
            yaxis_title=selected_metric,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No data to display for {selected_metric} in {selected_year}")

else:
    st.warning("Data could not be loaded. Please check the Google Sheet is shared correctly and the format is as expected.")
