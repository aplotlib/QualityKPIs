import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- PAGE CONFIGURATION ---
# Set the page configuration for a professional look suitable for a large screen
st.set_page_config(
    page_title="Quality KPI Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING (CSS) ---
# A more robust way to style is to target Streamlit's data-testid attributes
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #0f1116;
    }
    /* Main title */
    h1 {
        font-size: 3.5rem !important;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
    }
    /* Subheaders */
    h2, h3 {
        color: #E0E0E0;
        text-align: center;
    }
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2c2f3b;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.15);
    }
    /* Metric label */
    div[data-testid="stMetricLabel"] {
        font-size: 1.25rem;
        color: #a0a4b8;
    }
    /* Metric value */
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 600;
    }
    /* Metric delta (positive is bad, so red) */
    div[data-testid="stMetricDelta"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #a0a4b8;
        font-size: 1rem;
        padding-top: 2rem;
    }
    .footer a {
        color: #1c83e1;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING AND CLEANING ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_and_clean_data():
    """
    Connects to Google Sheets, loads the data, and performs necessary cleaning.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        worksheet_name = "Quality "
        data = conn.read(
            worksheet=worksheet_name,
            usecols=[0, 1, 2],
            header=28
        )
        data.dropna(how='all', inplace=True)
        data.rename(columns={
            'Company Wide Return Rate %': 'return_rate_by_qty',
            'Returns % of Revenue': 'return_rate_by_revenue'
        }, inplace=True)
        for col in ['return_rate_by_qty', 'return_rate_by_revenue']:
            data[col] = data[col].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0
        
        # Handle year changes correctly
        def assign_date(month_str):
            # This logic assumes the sheet is for the current or previous year
            current_month = datetime.now().month
            current_year = datetime.now().year
            month_num = datetime.strptime(month_str, "%b").month
            year = current_year if month_num <= current_month else current_year -1
            return datetime(year, month_num, 1)

        data['Date'] = data['Month'].apply(assign_date)
        data = data.sort_values(by='Date').reset_index(drop=True)
        return data

    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame(columns=['Month', 'Date', 'return_rate_by_qty', 'return_rate_by_revenue'])


# --- UI / DASHBOARD LAYOUT ---
df = load_and_clean_data()

st.title("📊 Quality Department KPI Dashboard")

if not df.empty:
    # --- HEADER ---
    last_update_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.markdown(f"### Last Updated: {last_update_time}")
    st.divider()

    # --- KEY METRICS DISPLAY ---
    latest_data = df.iloc[-1]
    
    # Calculate delta from the previous month
    delta_qty, delta_rev = None, None
    if len(df) > 1:
        previous_data = df.iloc[-2]
        delta_qty = latest_data['return_rate_by_qty'] - previous_data['return_rate_by_qty']
        delta_rev = latest_data['return_rate_by_revenue'] - previous_data['return_rate_by_revenue']

    st.subheader(f"Key Metrics for {latest_data['Month']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Return Rate (by Quantity)",
            value=f"{latest_data['return_rate_by_qty']:.2%}",
            delta=f"{delta_qty:.2%}" if delta_qty is not None else None,
            delta_color="inverse", # Lower is better
            help="The percentage of total units shipped that were returned."
        )
    with col2:
        st.metric(
            label="Return Rate (% of Revenue)",
            value=f"{latest_data['return_rate_by_revenue']:.2%}",
            delta=f"{delta_rev:.2%}" if delta_rev is not None else None,
            delta_color="inverse", # Lower is better
            help="The value of returned goods as a percentage of total revenue."
        )

    st.divider()

    # --- DATA VISUALIZATION ---
    st.subheader("Monthly Performance Trends")

    # Create a line chart using Plotly Express with a dark theme
    fig_line = px.line(
        df,
        x='Date',
        y=['return_rate_by_qty', 'return_rate_by_revenue'],
        markers=True,
        template="plotly_dark",
        labels={'value': 'Rate', 'Date': 'Month'},
    )
    fig_line.update_layout(
        title_text="Return Rates Over Time",
        legend_title_text='Metric',
        yaxis_tickformat='.1%',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig_line.update_traces(hovertemplate='<b>%{x|%B %Y}</b><br>Rate: %{y:.2%}')
    new_names = {
        'return_rate_by_qty': 'By Quantity', 
        'return_rate_by_revenue': '% of Revenue'
    }
    fig_line.for_each_trace(lambda t: t.update(name=new_names[t.name]))
    
    st.plotly_chart(fig_line, use_container_width=True)

    # --- RAW DATA DISPLAY ---
    with st.expander("View Raw Data Table"):
        st.dataframe(df)

else:
    st.warning("Data could not be loaded. Please check the Google Sheet is shared correctly and the format is as expected.")

# --- FOOTER ---
st.markdown(
    """
    <div class="footer">
        <p>For questions or feedback, please <a href='mailto:alexander.popoff@vivehealth.com'>contact the Quality Department</a>.</p>
    </div>
    """,
    unsafe_allow_html=True
)
