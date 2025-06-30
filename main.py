import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import traceback

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
    Connects to Google Sheets, loads the multi-metric/multi-year data,
    and transforms it into a clean, long-format DataFrame suitable for analysis.
    This version is robustly designed to parse the specific sheet structure.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        raw_data = conn.read(worksheet=799906691, header=None)
        
        # Robust pre-processing: Fill all NaN/None values with empty strings
        raw_data = raw_data.fillna('')
        # Convert entire dataframe to string to ensure consistent comparisons
        raw_data = raw_data.astype(str)

        all_metrics_data = []

        # Find rows that define the start of a new metric block (value in Column C, empty in D)
        metric_header_indices = raw_data[raw_data[2].ne('') & raw_data[3].eq('')].index

        if metric_header_indices.empty:
            st.error("Parsing Error: Could not identify any metric header rows. Please ensure that in your Google Sheet, metric titles (e.g., 'Overall Return Rate') are in Column C, and the corresponding cell in Column D is empty.")
            return pd.DataFrame()

        for i, start_row_idx in enumerate(metric_header_indices):
            metric_name = raw_data.iloc[start_row_idx, 2]
            
            # The month header is one row below the metric name row
            month_header_row_idx = start_row_idx + 1
            # Months start in Column F (index 5)
            months = raw_data.iloc[month_header_row_idx, 5:].tolist()
            
            # Data for this metric starts one row after the month header
            data_start_row_idx = month_header_row_idx + 1
            
            # Determine the end of the current block
            next_metric_start_idx = metric_header_indices[i+1] if (i + 1) < len(metric_header_indices) else len(raw_data)
            
            current_metric_data = raw_data.iloc[data_start_row_idx:next_metric_start_idx]

            # Process each row in the current metric's data block
            for _, row in current_metric_data.iterrows():
                # Year is in Column D (index 3)
                year_val = row[3]
                # Channel is in Column E (index 4)
                channel = row[4]
                
                if year_val == '' or channel == '':
                    continue

                # Monthly values start from Column F (index 5)
                values = row[5:].tolist()
                
                for month, value in zip(months, values):
                    if month != '' and value != '':
                        all_metrics_data.append({
                            'Metric': metric_name,
                            'Year': int(year_val),
                            'Channel': channel,
                            'Month': month,
                            'Value': value
                        })

        if not all_metrics_data:
            st.error("Could not parse any data from the sheet. Please verify the sheet structure.")
            return pd.DataFrame()

        df = pd.DataFrame(all_metrics_data)

        # --- Final Cleaning and Calculations ---
        pct_metrics = [m for m in df['Metric'].unique() if '%' in m]
        
        def clean_value(row):
            val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
            val_num = pd.to_numeric(val_str, errors='coerce')
            if row['Metric'] in pct_metrics:
                return val_num / 100.0
            return val_num
            
        df['Value'] = df.apply(clean_value, axis=1)
        df.dropna(subset=['Value'], inplace=True)
        
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
        
        df = df.sort_values(by=['Metric', 'Channel', 'Date'])
        df['MoM Change'] = df.groupby(['Metric', 'Channel'])['Value'].diff()
        df['YoY Change'] = df.groupby(['Metric', 'Channel', df['Date'].dt.month])['Value'].diff()
        
        return df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred while loading or transforming data: {e}")
        st.code(traceback.format_exc())
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
    
    lower_is_better = 'rate' in selected_metric.lower() or 'cost' in selected_metric.lower()

    # --- KPI OVERVIEW ---
    st.subheader(f"Executive Summary: {selected_metric} ({selected_year})")
    
    latest_amazon = df_filtered[df_filtered['Channel'] == 'Amazon'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'Amazon'].empty else None
    latest_b2b = df_filtered[df_filtered['Channel'] == 'B2B'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'B2B'].empty else None

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
            channel_df = df_filtered[df_filtered['Channel'] == channel]
            if not channel_df.empty:
                fig.add_trace(go.Scatter(x=channel_df['Date'].dt.month, y=channel_df['Value'], name=f'{channel} ({selected_year})', mode='lines+markers', line=dict(color=color, width=3)))
            channel_df_prev = df_previous_year[df_previous_year['Channel'] == channel]
            if not channel_df_prev.empty:
                fig.add_trace(go.Scatter(x=channel_df_prev['Date'].dt.month, y=channel_df_prev['Value'], name=f'{channel} ({selected_year - 1})', mode='lines', line=dict(color=color, width=2, dash='dash')))

        fig.update_layout(
            title_text=f"{selected_metric} Trend",
            template="plotly_dark",
            yaxis_tickformat=('.2%' if is_percent else ('${:,.2f}' if is_currency else ',.0f')),
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
