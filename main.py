# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Dict, Any, Optional

# --- Dependency Check for AI Models ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
    h2 { text-align: left; font-weight: 600; border-bottom: 1px solid #262730; padding-bottom: 0.5rem; margin-top: 1rem; margin-bottom: 1rem; }
    .stMetric { background-color: #0E1117; border: 1px solid #262730; border-radius: 10px; padding: 1rem; }
    .st-emotion-cache-1f1G2gn { border-color: #4A5568; } /* Expander border */
    .footer { text-align: center; color: #a0a4b8; font-size: 0.9rem; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING AND TRANSFORMATION ---
@st.cache_data(ttl=3600)
def load_and_transform_data() -> pd.DataFrame:
    """
    Loads and transforms data from a stable, embedded sample source,
    built exclusively from the provided metric list.
    """
    data = [
        # 2024 Data
        {'Metric': 'Total Orders', 'Year': 2024, 'Month': 'Jan', 'Value': 10000}, {'Metric': 'Total Orders', 'Year': 2024, 'Month': 'Feb', 'Value': 10200}, {'Metric': 'Total Orders', 'Year': 2024, 'Month': 'Mar', 'Value': 10500},
        {'Metric': 'Orders inspected', 'Year': 2024, 'Month': 'Jan', 'Value': 1000}, {'Metric': 'Orders inspected', 'Year': 2024, 'Month': 'Feb', 'Value': 1020}, {'Metric': 'Orders inspected', 'Year': 2024, 'Month': 'Mar', 'Value': 1050},
        {'Metric': '# Reworks', 'Year': 2024, 'Month': 'Jan', 'Value': 40}, {'Metric': '# Reworks', 'Year': 2024, 'Month': 'Feb', 'Value': 42}, {'Metric': '# Reworks', 'Year': 2024, 'Month': 'Mar', 'Value': 45},
        {'Metric': 'Total cost of inspection ($)', 'Year': 2024, 'Month': 'Jan', 'Value': 2000}, {'Metric': 'Total cost of inspection ($)', 'Year': 2024, 'Month': 'Feb', 'Value': 2040}, {'Metric': 'Total cost of inspection ($)', 'Year': 2024, 'Month': 'Mar', 'Value': 2100},
        {'Metric': 'Tickets Handled', 'Year': 2024, 'Month': 'Jan', 'Value': 500}, {'Metric': 'Tickets Handled', 'Year': 2024, 'Month': 'Feb', 'Value': 510}, {'Metric': 'Tickets Handled', 'Year': 2024, 'Month': 'Mar', 'Value': 520},
        {'Metric': 'Full replacements (same day)', 'Year': 2024, 'Month': 'Jan', 'Value': 50}, {'Metric': 'Full replacements (same day)', 'Year': 2024, 'Month': 'Feb', 'Value': 51}, {'Metric': 'Full replacements (same day)', 'Year': 2024, 'Month': 'Mar', 'Value': 52},
        {'Metric': 'Replacement parts (next day)', 'Year': 2024, 'Month': 'Jan', 'Value': 100}, {'Metric': 'Replacement parts (next day)', 'Year': 2024, 'Month': 'Feb', 'Value': 102}, {'Metric': 'Replacement parts (next day)', 'Year': 2024, 'Month': 'Mar', 'Value': 105},
        {'Metric': 'Returns (w/in 3 days)', 'Year': 2024, 'Month': 'Jan', 'Value': 30}, {'Metric': 'Returns (w/in 3 days)', 'Year': 2024, 'Month': 'Feb', 'Value': 31}, {'Metric': 'Returns (w/in 3 days)', 'Year': 2024, 'Month': 'Mar', 'Value': 32},
        {'Metric': 'Other cases: unresolved', 'Year': 2024, 'Month': 'Jan', 'Value': 10}, {'Metric': 'Other cases: unresolved', 'Year': 2024, 'Month': 'Feb', 'Value': 11}, {'Metric': 'Other cases: unresolved', 'Year': 2024, 'Month': 'Mar', 'Value': 12},

        # 2025 Data
        {'Metric': 'Total Orders', 'Year': 2025, 'Month': 'Jan', 'Value': 13000}, {'Metric': 'Total Orders', 'Year': 2025, 'Month': 'Feb', 'Value': 13200}, {'Metric': 'Total Orders', 'Year': 2025, 'Month': 'Mar', 'Value': 13500},
        {'Metric': 'Orders inspected', 'Year': 2025, 'Month': 'Jan', 'Value': 1950}, {'Metric': 'Orders inspected', 'Year': 2025, 'Month': 'Feb', 'Value': 1980}, {'Metric': 'Orders inspected', 'Year': 2025, 'Month': 'Mar', 'Value': 2025},
        {'Metric': '# Reworks', 'Year': 2025, 'Month': 'Jan', 'Value': 78}, {'Metric': '# Reworks', 'Year': 2025, 'Month': 'Feb', 'Value': 81}, {'Metric': '# Reworks', 'Year': 2025, 'Month': 'Mar', 'Value': 85},
        {'Metric': 'Total cost of inspection ($)', 'Year': 2025, 'Month': 'Jan', 'Value': 4000}, {'Metric': 'Total cost of inspection ($)', 'Year': 2025, 'Month': 'Feb', 'Value': 4050}, {'Metric': 'Total cost of inspection ($)', 'Year': 2025, 'Month': 'Mar', 'Value': 4150},
        {'Metric': 'Tickets Handled', 'Year': 2025, 'Month': 'Jan', 'Value': 620}, {'Metric': 'Tickets Handled', 'Year': 2025, 'Month': 'Feb', 'Value': 630}, {'Metric': 'Tickets Handled', 'Year': 2025, 'Month': 'Mar', 'Value': 640},
        {'Metric': 'Full replacements (same day)', 'Year': 2025, 'Month': 'Jan', 'Value': 62}, {'Metric': 'Full replacements (same day)', 'Year': 2025, 'Month': 'Feb', 'Value': 63}, {'Metric': 'Full replacements (same day)', 'Year': 2025, 'Month': 'Mar', 'Value': 64},
        {'Metric': 'Replacement parts (next day)', 'Year': 2025, 'Month': 'Jan', 'Value': 130}, {'Metric': 'Replacement parts (next day)', 'Year': 2025, 'Month': 'Feb', 'Value': 132}, {'Metric': 'Replacement parts (next day)', 'Year': 2025, 'Month': 'Mar', 'Value': 135},
        {'Metric': 'Returns (w/in 3 days)', 'Year': 2025, 'Month': 'Jan', 'Value': 42}, {'Metric': 'Returns (w/in 3 days)', 'Year': 2025, 'Month': 'Feb', 'Value': 43}, {'Metric': 'Returns (w/in 3 days)', 'Year': 2025, 'Month': 'Mar', 'Value': 44},
        {'Metric': 'Other cases: unresolved', 'Year': 2025, 'Month': 'Jan', 'Value': 15}, {'Metric': 'Other cases: unresolved', 'Year': 2025, 'Month': 'Feb', 'Value': 16}, {'Metric': 'Other cases: unresolved', 'Year': 2025, 'Month': 'Mar', 'Value': 17},
    ]
    df = pd.DataFrame(data)

    # --- Calculate Derived Metrics ---
    derived_metrics = []
    for year in df['Year'].unique():
        for month in df['Month'].unique():
            df_month = df[(df['Year'] == year) & (df['Month'] == month)]
            if df_month.empty: continue
            
            # Use .get(0) to gracefully handle cases where a metric might be missing for a month
            total_orders = df_month.loc[df_month['Metric'] == 'Total Orders', 'Value'].values[0] if not df_month[df_month['Metric'] == 'Total Orders'].empty else None
            orders_inspected = df_month.loc[df_month['Metric'] == 'Orders inspected', 'Value'].values[0] if not df_month[df_month['Metric'] == 'Orders inspected'].empty else None
            reworks = df_month.loc[df_month['Metric'] == '# Reworks', 'Value'].values[0] if not df_month[df_month['Metric'] == '# Reworks'].empty else None
            cost_inspection = df_month.loc[df_month['Metric'] == 'Total cost of inspection ($)', 'Value'].values[0] if not df_month[df_month['Metric'] == 'Total cost of inspection ($)'].empty else None
            returns = df_month.loc[df_month['Metric'] == 'Returns (w/in 3 days)', 'Value'].values[0] if not df_month[df_month['Metric'] == 'Returns (w/in 3 days)'].empty else None
            
            if all(v is not None for v in [total_orders, orders_inspected, reworks, cost_inspection, returns]):
                derived_values = {
                    "% Order inspected (%)": (orders_inspected / total_orders) if total_orders else 0,
                    "% Reworks (%)": (reworks / orders_inspected) if orders_inspected else 0,
                    "Average Cost per Inspection ($)": (cost_inspection / orders_inspected) if orders_inspected else 0,
                    "Overall Return Rate (%)": (returns / total_orders) if total_orders else 0,
                }
                for name, value in derived_values.items():
                    derived_metrics.append({'Metric': name, 'Year': year, 'Month': month, 'Value': value})
    
    if derived_metrics:
        df = pd.concat([df, pd.DataFrame(derived_metrics)], ignore_index=True)

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
    df.dropna(subset=['Value', 'Date'], inplace=True)
    df = df.sort_values(by=['Metric', 'Date'])
    df_prev = df.copy(); df_prev['Date'] += pd.DateOffset(years=1)
    df = pd.merge(df, df_prev[['Metric', 'Date', 'Value']], on=['Metric', 'Date'], how='left', suffixes=('', '_prev'))
    df['YoY Change'] = df['Value'] - df['Value_prev']
    return df.sort_values(by='Date').reset_index(drop=True)

# --- AI ANALYTICS ENGINE ---
class AIAnalyzer:
    @staticmethod
    def get_data_summary(df, metric, year):
        df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
        if df_metric.empty: return "No data available."
        latest = df_metric.sort_values('Date').iloc[-1]
        summary = f"Analysis for '{metric}' for {year}: Latest value is {latest['Value']:.2f} ({latest['Date']:%b %Y}), with a YoY change of {latest['YoY Change']:.2f}. The last 3 months are: {df_metric.tail(3)['Value'].tolist()}"
        return summary

    @staticmethod
    def generate_insights(data_summary: str, metric_name: str, api_key: str) -> Optional[List[str]]:
        if not OPENAI_AVAILABLE: return ["Error: The `openai` library is not installed."]
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""As a Principal Data Analyst, provide 1-2 sharp, actionable insights for a business owner based on the following data summary for the KPI '{metric_name}'. Address the owner directly and be concise. Return your response as a JSON object with a single key "insights" which contains a list of strings.

            Data Summary:
            {data_summary}
            """
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            content = response.choices[0].message.content
            insights_data = json.loads(content)
            return insights_data.get("insights", ["AI returned an unexpected format."])
        except openai.AuthenticationError:
            return ["Authentication Error: Your OpenAI API key is invalid or has expired."]
        except Exception as e:
            return [f"AI interaction failed: {e}"]

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

def render_chart(df: pd.DataFrame, metric: str, year: int, issue_composition_mode=False):
    st.subheader(f"📊 Visual Analysis")
    
    if issue_composition_mode:
        issue_metrics = ["Full replacements (same day)", "Replacement parts (next day)", "Returns (w/in 3 days)", "Other cases: unresolved"]
        df_issues = df[(df['Metric'].isin(issue_metrics)) & (df['Year'] == year)]
        if df_issues.empty: st.warning("No issue data available."); return
        
        fig = px.bar(df_issues, x='Date', y='Value', color='Metric', title=f"Monthly Issue Composition for {year}", template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(tickformat="%b"), margin=dict(t=40, b=20, l=40, r=20))
    else:
        chart_type = 'bar' if any(m in metric for m in ['Total Orders', 'Orders inspected', '# Reworks', 'Tickets Handled']) else 'line'
        df_metric = df[df['Metric'] == metric]
        df_curr, df_prev = df_metric[df_metric['Year'] == year], df_metric[df_metric['Year'] == year - 1]
        if df_curr.empty: st.warning("No data available."); return
        
        fig = go.Figure()
        if not df_prev.empty: fig.add_trace(go.Scatter(x=df_prev['Date'].dt.month, y=df_prev['Value'], name=f'{year-1}', mode='lines', line=dict(color="#444444", width=2, dash='dash')))
        
        if chart_type == 'line':
            fig.add_trace(go.Scatter(x=df_curr['Date'].dt.month, y=df_curr['Value'], name=f'{year}', mode='lines+markers', line=dict(color="#3b82f6", width=4), marker=dict(size=8), fill='tonexty', fillcolor='rgba(59,130,246,0.1)'))
        else:
            fig.add_trace(go.Bar(x=df_curr['Date'].dt.month, y=df_curr['Value'], name=f'{year}', marker_color='#3b82f6'))

        goal_map = {"Overall Return Rate (%)": 0.03, "% Reworks (%)": 0.05, "Average Cost per Inspection ($)": 2.0}
        if metric in goal_map: fig.add_hline(y=goal_map[metric], line_dash="dot", annotation_text="Goal", annotation_position="bottom right", line_color="gray")
        
        is_percent, is_currency = '%' in metric, '$' in metric
        yaxis_tickformat = '.1%' if is_percent else ('$,.2f' if is_currency else ',.0f')
        fig.update_layout(template="plotly_dark", yaxis_tickformat=yaxis_tickformat, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['J','F','M','A','M','J','J','A','S','O','N','D']), margin=dict(t=0, b=20, l=40, r=20))
        
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.title("✅ Quality & Product Dashboard")
df = load_and_transform_data()
if df.empty: st.error("Fatal Error: Data could not be loaded or processed."); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Dashboard Controls")
    order_metrics = ["Overall Return Rate (%)", "Total Orders", "Orders inspected", "% Order inspected (%)", "Total cost of inspection ($)", "Average Cost per Inspection ($)", "# Reworks", "% Reworks (%)"]
    issue_metrics = ["Tickets Handled", "Full replacements (same day)", "Replacement parts (next day)", "Returns (w/in 3 days)", "Other cases: unresolved"]
    
    st.subheader("Metric Selection")
    metric_category = st.radio("Category", ["Order & Inspection KPIs", "Issue & Ticket KPIs"], label_visibility="collapsed")
    
    is_issue_category = metric_category == "Issue & Ticket KPIs"
    if is_issue_category: selected_metric = st.selectbox("Select KPI", issue_metrics)
    else: selected_metric = st.selectbox("Select KPI", order_metrics)

    st.subheader("Timeframe")
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True), label_visibility="collapsed")
    
    st.markdown("---")
    st.header("AI Analysis")
    # --- FIX: Check for both uppercase and lowercase versions of the API key ---
    api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    
    if st.button("🤖 Generate AI Insights", disabled=not api_key, use_container_width=True):
        summary = AIAnalyzer.get_data_summary(df, selected_metric, selected_year)
        with st.spinner("Analyzing data..."): st.session_state.ai_insights = AIAnalyzer.generate_insights(summary, selected_metric, api_key)
        st.session_state.insights_for_metric = selected_metric
    
    # --- Secrets & AI Status Debugger ---
    with st.expander("Secrets & AI Status Debugger"):
        st.write("Use this to verify your Streamlit secrets setup.")
        if api_key:
            st.success("OpenAI API Key Found!")
            st.info("The AI Insights button above is enabled.")
        else:
            st.error("OpenAI API Key Not Found.")
            st.info("The app looked for `OPENAI_API_KEY` or `openai_api_key` but found neither. Please add one to your secrets to enable AI.")
        
        st.write("All available secret keys found by the app:")
        st.json([key for key in st.secrets.keys()])

    st.markdown("---")
    st.markdown("<div class='footer'>Built for Leadership</div>", unsafe_allow_html=True)

# --- MAIN PANEL DISPLAY ---
if 'ai_insights' in st.session_state and st.session_state.get('insights_for_metric') == selected_metric:
    st.subheader("💡 AI-Powered Insights")
    container = st.container(border=True)
    for insight in st.session_state.ai_insights: container.markdown(f"&bull; {insight}")
    st.markdown("---")

left_col, right_col = st.columns((2.5, 1))
with left_col:
    render_chart(df, selected_metric, selected_year, issue_composition_mode=is_issue_category)

with right_col:
    render_kpi_details(df, selected_metric, selected_year)
    with st.expander("Show Monthly Data"):
        display_df = df[(df.Metric == selected_metric) & (df.Year == selected_year)]
        if not display_df.empty:
            st.dataframe(display_df[['Date', 'Value', 'YoY Change']].set_index('Date'), use_container_width=True)
