# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quality & Product Dashboard",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 2rem; }
    h1 { text-align: center; font-weight: 700; padding-bottom: 1rem; }
    h2 { text-align: center; font-weight: 600; border-bottom: 1px solid #262730; padding-bottom: 0.5rem; margin-top: 2rem; margin-bottom: 1rem; }
    .stMetric { background-color: #0E1117; border: 1px solid #262730; border-radius: 10px; padding: 1rem; }
    .footer { text-align: center; color: #a0a4b8; font-size: 0.9rem; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING AND TRANSFORMATION ---
@st.cache_data(ttl=3600)
def load_and_transform_data() -> pd.DataFrame:
    # Base numbers for generating realistic, coherent data
    base_data = {
        '2024': {'Total Orders': [10000, 10200, 10500, 10300, 10600, 10800, 11000, 11200, 11500, 11800, 12000, 12500], 'Orders inspected': [1000, 1020, 1050, 1030, 1060, 1080, 1100, 1120, 1150, 1180, 1200, 1250], '# Reworks': [40, 42, 45, 43, 48, 50, 52, 55, 58, 60, 62, 65], 'Total cost of inspection ($)': [2000, 2040, 2100, 2060, 2120, 2160, 2200, 2240, 2300, 2360, 2400, 2500], 'Tickets Handled': [500, 510, 520, 515, 530, 540, 550, 560, 570, 580, 590, 600], 'Full replacements (same day)': [50, 51, 52, 51, 53, 54, 55, 56, 57, 58, 59, 60], 'Replacement parts (next day)': [100, 102, 105, 103, 106, 108, 110, 112, 115, 118, 120, 125], 'Returns (w/in 3 days)': [30, 31, 32, 31, 33, 34, 35, 36, 37, 38, 39, 40], 'Other cases: unresolved': [10, 11, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20]},
        '2025': {'Total Orders': [13000, 13200, 13500, 13300, 13600, 13800], 'Orders inspected': [1950, 1980, 2025, 1995, 2040, 2070], '# Reworks': [78, 81, 85, 82, 86, 90], 'Total cost of inspection ($)': [4000, 4050, 4150, 4100, 4200, 4250], 'Tickets Handled': [620, 630, 640, 635, 650, 660], 'Full replacements (same day)': [62, 63, 64, 63, 65, 66], 'Replacement parts (next day)': [130, 132, 135, 133, 136, 138], 'Returns (w/in 3 days)': [42, 43, 44, 43, 45, 46], 'Other cases: unresolved': [15, 16, 17, 16, 18, 19]}
    }
    all_metrics = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for year, year_data in base_data.items():
        for i in range(len(year_data['Total Orders'])):
            orders_inspected = year_data['Orders inspected'][i]; total_orders = year_data['Total Orders'][i]; reworks = year_data['# Reworks'][i]; cost_inspection = year_data['Total cost of inspection ($)'][i]
            metric_values = {
                "Overall Return Rate (%)": (year_data['Returns (w/in 3 days)'][i] / total_orders) if total_orders > 0 else 0, "Total Orders": total_orders, "Orders inspected": orders_inspected,
                "% Order inspected (%)": (orders_inspected / total_orders) if total_orders > 0 else 0, "Total cost of inspection ($)": cost_inspection,
                "Average Cost per Inspection ($)": (cost_inspection / orders_inspected) if orders_inspected > 0 else 0, "# Reworks": reworks, "% Reworks (%)": (reworks / orders_inspected) if orders_inspected > 0 else 0,
                "Tickets Handled": year_data['Tickets Handled'][i], "Full replacements (same day)": year_data['Full replacements (same day)'][i],
                "Replacement parts (next day)": year_data['Replacement parts (next day)'][i], "Returns (w/in 3 days)": year_data['Returns (w/in 3 days)'][i], "Other cases: unresolved": year_data['Other cases: unresolved'][i]
            }
            for name, value in metric_values.items(): all_metrics.append({'Metric': name, 'Year': int(year), 'Channel': 'Overall', 'Month': months[i], 'Value': value})
    df = pd.DataFrame(all_metrics)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], errors='coerce')
    df.dropna(subset=['Value', 'Date'], inplace=True)
    df = df.sort_values(by=['Metric', 'Channel', 'Date'])
    df_prev = df.copy(); df_prev['Date'] += pd.DateOffset(years=1)
    df = pd.merge(df, df_prev[['Metric', 'Channel', 'Date', 'Value']], on=['Metric', 'Channel', 'Date'], how='left', suffixes=('', '_prev'))
    df['YoY Change'] = df['Value'] - df['Value_prev']
    return df.sort_values(by='Date').reset_index(drop=True)

# --- UI RENDERING FUNCTIONS ---
def render_kpi_details(df: pd.DataFrame, metric: str, year: int):
    st.subheader("📌 Key Metrics")
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
    if df_metric.empty: st.warning(f"No data for '{metric}' in {year}."); return
    latest = df_metric.sort_values('Date').iloc[-1]
    yoy_change = latest['YoY Change']
    lower_is_better = any(term in metric.lower() for term in ['rate', 'cost', 'rework', 'unresolved'])
    is_good = pd.notna(yoy_change) and ((yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better))
    icon = "✅" if is_good else "⚠️"
    is_percent, is_currency = '%' in metric, '$' in metric
    if is_percent: val_f, delta_f = "{:,.2%}", "{:+.2f} pts"
    elif is_currency: val_f, delta_f = "${:,.2f}", "{:+.2f}"
    else: val_f, delta_f = "{:,.0f}", "{:+.0f}"
    delta_display = "No prior data"
    if pd.notna(yoy_change): delta_display = f"{icon} {delta_f.format(yoy_change)} vs. PY"
    st.metric(label=f"Overall Performance ({latest['Date']:%b %Y})", value=val_f.format(latest['Value']), delta=delta_display, delta_color="off")

def render_dynamic_chart(df: pd.DataFrame, metric: str, year: int):
    """Intelligently selects and renders the best chart based on the metric type."""
    st.subheader(f"📊 {metric} Performance")
    
    # --- Define chart type for each metric ---
    chart_type_map = {
        "Total Orders": 'bar',
        "Orders inspected": 'bar',
        "# Reworks": 'bar',
        "Total cost of inspection ($)": 'area',
        "Tickets Handled": 'bar'
    }
    chart_type = chart_type_map.get(metric, 'line') # Default to line chart
    
    df_metric = df[(df['Metric'] == metric) & (df['Channel'] == 'Overall')]
    df_curr = df_metric[df_metric['Year'] == year]
    df_prev = df_metric[df_metric['Year'] == year - 1]
    if df_curr.empty: st.warning("No data available for this period."); return

    fig = go.Figure()

    # --- Add traces based on chart type ---
    if chart_type in ['line', 'area']:
        if not df_prev.empty:
            fig.add_trace(go.Scatter(x=df_prev['Date'].dt.month, y=df_prev['Value'], name=f'{year-1}', mode='lines', line=dict(color="#444444", width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df_curr['Date'].dt.month, y=df_curr['Value'], name=f'{year}', mode='lines+markers', line=dict(color="#3b82f6", width=4), marker=dict(size=8), fill='tonexty' if chart_type == 'area' else None, fillcolor='rgba(59,130,246,0.1)'))
    
    elif chart_type == 'bar':
        if not df_prev.empty:
             fig.add_trace(go.Scatter(x=df_prev['Date'].dt.month, y=df_prev['Value'], name=f'{year-1}', mode='lines', line=dict(color="#444444", width=2, dash='dash')))
        fig.add_trace(go.Bar(x=df_curr['Date'].dt.month, y=df_curr['Value'], name=f'{year}', marker_color='#3b82f6'))

    # --- Add Goal Lines and Format Axes ---
    goal_map = {"Overall Return Rate (%)": 0.03, "% Reworks (%)": 0.05, "Average Cost per Inspection ($)": 2.0}
    if metric in goal_map:
        fig.add_hline(y=goal_map[metric], line_dash="dot", annotation_text="Goal", annotation_position="bottom right", line_color="gray")

    is_percent, is_currency = '%' in metric, '$' in metric
    yaxis_tickformat = '.1%' if is_percent else ('$,.2f' if is_currency else ',.0f')

    fig.update_layout(template="plotly_dark", yaxis_tickformat=yaxis_tickformat, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['J','F','M','A','M','J','J','A','S','O','N','D']), margin=dict(t=0, b=20, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)

def render_issue_composition_chart(df: pd.DataFrame, year: int):
    """Renders a stacked bar chart for the issue/ticket KPIs."""
    st.subheader("📊 Monthly Issue Composition")
    issue_metrics = ["Full replacements (same day)", "Replacement parts (next day)", "Returns (w/in 3 days)", "Other cases: unresolved"]
    df_issues = df[(df['Metric'].isin(issue_metrics)) & (df['Year'] == year)]
    if df_issues.empty: st.warning("No issue data available for this period."); return

    fig = px.bar(df_issues, x='Date', y='Value', color='Metric', title=f"Issue Volume and Composition for {year}",
                 labels={'Value': 'Number of Cases', 'Date': 'Month', 'Metric': 'Issue Type'},
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
    
    fig.update_layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(tickformat="%b"), margin=dict(t=40, b=20, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.title("✅ Quality & Product Dashboard")
df = load_and_transform_data()
if df.empty: st.warning("Data could not be loaded or processed."); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Dashboard Controls")
    order_metrics = ["Overall Return Rate (%)", "Total Orders", "Orders inspected", "% Order inspected (%)", "Total cost of inspection ($)", "Average Cost per Inspection ($)", "# Reworks", "% Reworks (%)"]
    issue_metrics = ["Tickets Handled", "Full replacements (same day)", "Replacement parts (next day)", "Returns (w/in 3 days)", "Other cases: unresolved"]
    
    st.subheader("Metric Selection")
    metric_category = st.radio("Category", ["Order & Inspection KPIs", "Issue & Ticket KPIs"], label_visibility="collapsed")
    
    if metric_category == "Order & Inspection KPIs":
        selected_metric = st.selectbox("Select KPI", order_metrics)
    else:
        # For this category, we select a primary metric but show the composition chart
        selected_metric = st.selectbox("Select KPI", issue_metrics)

    st.subheader("Timeframe")
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True), label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("<div class='footer'>Built for Leadership</div>", unsafe_allow_html=True)

# --- MAIN PANEL DISPLAY ---
left_col, right_col = st.columns((2.5, 1))

with left_col:
    # --- DYNAMIC CHART DISPLAY ---
    if metric_category == "Issue & Ticket KPIs":
        render_issue_composition_chart(df, selected_year)
    else:
        render_dynamic_chart(df, selected_metric, selected_year)

with right_col:
    render_kpi_details(df, selected_metric, selected_year)
    with st.expander("Show Monthly Data"):
        st.dataframe(df[(df.Metric == selected_metric) & (df.Year == selected_year)][['Date', 'Value', 'YoY Change']].set_index('Date'), use_container_width=True)
