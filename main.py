# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Optional

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
    .footer { text-align: center; color: #a0a4b8; font-size: 0.9rem; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# --- DATA PARSING ENGINE for UPLOADED FILE ---
@st.cache_data
def parse_and_transform_data(uploaded_file) -> pd.DataFrame:
    """
    Reads the user's uploaded messy CSV file and transforms it into a clean, tidy DataFrame.
    This is the core logic for handling the specific multi-table format.
    """
    try:
        raw_df = pd.read_csv(uploaded_file, header=None).fillna('')
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        return pd.DataFrame()

    all_metrics_data = []
    current_metric, current_months, current_year = None, [], None

    for _, row_series in raw_df.iterrows():
        row = row_series.tolist()
        # Condition 1: Is this a METRIC TITLE row? (e.g., "Overall Return Rate")
        if row[2] and not row[3] and not row[4]:
            current_metric, current_months, current_year = row[2], [], None
            continue
        # Condition 2: Is this a MONTH HEADER row?
        if row[5] == 'Jan':
            current_months = [m for m in row[5:] if m and 'Total' not in m]
            continue
        
        if not (current_metric and current_months): continue
        if str(row[3]).isnumeric(): current_year = int(float(row[3]))
        
        # Condition 3: Is this a DATA row?
        if current_year and (row[4] or str(row[3]).isnumeric()):
            # All data is aggregated under 'Overall' channel for simplicity and accuracy
            for month, value in zip(current_months, row[5:]):
                if value:
                    all_metrics_data.append({
                        'Metric': current_metric.strip(),
                        'Year': current_year,
                        'Month': month,
                        'Value': value
                    })

    if not all_metrics_data:
        st.error("Could not parse any structured data from the file. Please ensure it follows the expected format.")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics_data)

    # --- Data Cleaning and Final Calculations ---
    df['Value'] = pd.to_numeric(df['Value'].astype(str).str.replace(r'[%,$]', '', regex=True), errors='coerce')
    df.dropna(subset=['Value'], inplace=True)
    
    # Identify which metrics were originally percentages to scale them correctly
    pct_metrics = [
        "Overall Return Rate", "% Order inspected", "% Reworks"
    ]
    for m in pct_metrics:
        df.loc[df['Metric'] == m, 'Value'] /= 100.0

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
    df.sort_values(by=['Metric', 'Date'], inplace=True)

    # Calculate YoY change by comparing to the value 12 periods ago
    df['YoY Change'] = df.groupby('Metric')['Value'].diff(12)
    return df.reset_index(drop=True)

# --- AI ANALYTICS ENGINE ---
class AIAnalyzer:
    @staticmethod
    def get_data_summary(df, metric, year):
        df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
        if df_metric.empty: return "No data available."
        latest = df_metric.sort_values('Date').iloc[-1]
        yoy_change_str = f"{latest['YoY Change']:.2f}" if pd.notna(latest['YoY Change']) else "N/A"
        summary = f"Analysis for '{metric}' for {year}: Latest value is {latest['Value']:.2f} ({latest['Date']:%b %Y}), with a YoY change of {yoy_change_str}. The last 3 months are: {df_metric.tail(3)['Value'].tolist()}"
        return summary

    @staticmethod
    def generate_insights(data_summary: str, metric_name: str, api_key: str) -> Optional[List[str]]:
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""As a Principal Data Analyst, provide 1-2 sharp, actionable insights for a business owner based on the following data summary for the KPI '{metric_name}'. Address the owner directly and be concise. Return your response as a JSON object with a single key "insights" which contains a list of strings.
            Data Summary: {data_summary}"""
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            insights_data = json.loads(response.choices[0].message.content)
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
        if not df_prev.empty: fig.add_trace(go.Scatter(x=df_prev['Date'], y=df_prev['Value'], name=f'{year-1}', mode='lines', line=dict(color="#444444", width=2, dash='dash')))
        
        if chart_type == 'line':
            fig.add_trace(go.Scatter(x=df_curr['Date'], y=df_curr['Value'], name=f'{year}', mode='lines+markers', line=dict(color="#3b82f6", width=4), marker=dict(size=8), fill='tonexty', fillcolor='rgba(59,130,246,0.1)'))
        else:
            fig.add_trace(go.Bar(x=df_curr['Date'], y=df_curr['Value'], name=f'{year}', marker_color='#3b82f6'))

        goal_map = {"Overall Return Rate": 0.03, "% Reworks": 0.05, "Average Cost per Inspection ($)": 2.0}
        if metric in goal_map: fig.add_hline(y=goal_map[metric], line_dash="dot", annotation_text="Goal", annotation_position="bottom right", line_color="gray")
        
        is_percent, is_currency = '%' in metric, '$' in metric
        yaxis_tickformat = '.1%' if is_percent else ('$,.2f' if is_currency else ',.0f')
        fig.update_layout(template="plotly_dark", yaxis_tickformat=yaxis_tickformat, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(tickformat="%b %Y"), margin=dict(t=0, b=20, l=40, r=20))
        
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN APP ---
st.title("✅ Quality & Product Dashboard")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose your Quality KPI CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload your data file to begin analysis.")
    st.stop()

# --- Process the uploaded file and render the dashboard ---
df = parse_and_transform_data(uploaded_file)

if df.empty:
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Dashboard Controls")
    # Dynamically get metric lists from the processed dataframe
    all_kpis = df['Metric'].unique()
    order_metrics = [kpi for kpi in all_kpis if any(term in kpi for term in ['Order', 'Inspect', 'Cost', 'Rework', 'Return'])]
    issue_metrics = [kpi for kpi in all_kpis if any(term in kpi for term in ['Ticket', 'replacement', 'Return', 'unresolved'])]
    
    st.subheader("Metric Selection")
    metric_category = st.radio("Category", ["Order & Inspection KPIs", "Issue & Ticket KPIs"], label_visibility="collapsed")
    
    is_issue_category = metric_category == "Issue & Ticket KPIs"
    if is_issue_category: selected_metric = st.selectbox("Select KPI", issue_metrics)
    else: selected_metric = st.selectbox("Select KPI", order_metrics)

    st.subheader("Timeframe")
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True), label_visibility="collapsed")
    
    st.markdown("---")
    st.header("AI Analysis")
    api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    if not api_key: st.warning("Set `OPENAI_API_KEY` in secrets to enable AI.")
    
    if st.button("🤖 Generate AI Insights", disabled=not api_key, use_container_width=True):
        summary = AIAnalyzer.get_data_summary(df, selected_metric, selected_year)
        with st.spinner("Analyzing data..."): st.session_state.ai_insights = AIAnalyzer.generate_insights(summary, selected_metric, api_key)
        st.session_state.insights_for_metric = selected_metric
    
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
