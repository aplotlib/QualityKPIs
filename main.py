# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from typing import List, Dict, Any, Optional

# --- Dependency Check for AI Models ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quality KPI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- THEME & STYLING ---
def apply_styling(mode="Interactive"):
    if mode == "Presentation (TV)":
        # High-contrast, large-font theme for TV screens
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class*="st-"] {
                font-family: 'Inter', sans-serif;
                background-color: #FFFFFF;
                color: #111111;
            }
            h1 {
                font-size: 4.5rem !important;
                font-weight: 700;
                text-align: center;
                color: #111111;
            }
            .stMetric {
                background-color: #F0F2F6;
                border: 1px solid #D1D5DB;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
            }
            .stMetricLabel {
                font-size: 2rem !important;
                color: #4B5563;
                font-weight: 600;
            }
            .stMetricValue {
                font-size: 6rem !important;
                font-weight: 700;
                color: #111111;
            }
            .stMetricDelta {
                font-size: 1.75rem !important;
                font-weight: 600;
            }
            .stPlotlyChart {
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Standard sleek dark theme for interactive analysis
        st.markdown("""
        <style>
            .main .block-container { padding: 2rem; }
            h1, h2, h3 { text-align: center; }
            .stMetric { background-color: #0E1117; border: 1px solid #262730; border-radius: 10px; padding: 1rem; }
        </style>
        """, unsafe_allow_html=True)

# --- DATA LOADING AND TRANSFORMATION ---
@st.cache_data(ttl=3600)
def load_and_transform_data() -> pd.DataFrame:
    # --- EMBEDDED SAMPLE DATA (Corrected for Amazon & B2B) ---
    sample_raw_data = [
        ['', '', 'Overall Return Rate (%)', '', ''],
        ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        ['', '', '', '2025', 'Amazon', '4.5%', '4.6%', '4.4%', '4.5%', '4.7%', '4.6%'],
        ['', '', '', '2025', 'B2B', '3.1%', '3.0%', '3.2%', '3.1%', '3.0%', '3.1%'],
        ['', '', '', '2025', 'Overall', '4.1%', '4.2%', '4.0%', '4.1%', '4.2%', '4.1%'],
        ['', '', '', '2024', 'Amazon', '5.1%', '5.3%', '5.0%', '4.8%', '4.9%', '5.2%', '5.1%', '4.9%', '5.0%', '5.2%', '5.4%', '5.3%'],
        ['', '', '', '2024', 'B2B', '3.5%', '3.4%', '3.6%', '3.5%', '3.4%', '3.5%', '3.6%', '3.5%', '3.4%', '3.5%', '3.6%', '3.5%'],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', 'Customer Satisfaction (CSAT)', '', ''],
        ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        ['', '', '', '2025', 'Phone Support', '4.3', '4.4', '4.2', '4.5', '4.4', '4.5'],
        ['', '', '', '2025', 'Email Support', '4.6', '4.7', '4.6', '4.7', '4.8', '4.7'],
        ['', '', '', '2025', 'Overall', '4.4', '4.5', '4.4', '4.6', '4.6', '4.6'],
        ['', '', '', '2024', 'Phone Support', '4.1', '4.2', '4.0', '4.2', '4.3', '4.1', '4.2', '4.3', '4.4', '4.2', '4.3', '4.4'],
        ['', '', '', '2024', 'Email Support', '4.5', '4.6', '4.5', '4.4', '4.5', '4.6', '4.7', '4.6', '4.5', '4.6', '4.7', '4.8'],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', 'First Contact Resolution (%)', '', ''],
        ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        ['', '', '', '2025', 'Phone Support', '78%', '79%', '77%', '80%', '81%', '80%'],
        ['', '', '', '2025', 'Email Support', '88%', '89%', '88%', '90%', '91%', '90%'],
        ['', '', '', '2025', 'Overall', '82%', '83%', '81%', '84%', '85%', '84%'],
        ['', '', '', '2024', 'Phone Support', '75%', '76%', '74%', '75%', '76%', '77%', '76%', '78%', '79%', '78%', '77%', '78%'],
        ['', '', '', '2024', 'Email Support', '85%', '86%', '86%', '87%', '88%', '87%', '88%', '89%', '90%', '89%', '88%', '89%'],
    ]
    raw_data = pd.DataFrame(sample_raw_data).fillna('')
    all_metrics, current_metric, current_months, current_year = [], None, [], None
    for _, row_series in raw_data.iterrows():
        row = row_series.tolist()
        if row[2] and not row[3] and not row[4]: current_metric, current_months, current_year = row[2], [], None; continue
        if row[5] == 'Jan': current_months = [m for m in row[5:] if m and 'Total' not in m]; continue
        if not (current_metric and current_months): continue
        if str(row[3]).isnumeric(): current_year = int(float(row[3]))
        channel = row[4] if row[4] else 'Overall'
        if current_year and (row[4] or str(row[3]).isnumeric()):
            for month, value in zip(current_months, row[5:]):
                if value: all_metrics.append({'Metric': current_metric, 'Year': current_year, 'Channel': channel, 'Month': month, 'Value': value})
    df = pd.DataFrame(all_metrics)
    pct_metrics = [m for m in df['Metric'].unique() if '%' in m]
    def clean_value(row):
        val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
        return pd.to_numeric(val_str, errors='coerce') / 100.0 if row['Metric'] in pct_metrics else pd.to_numeric(val_str, errors='coerce')
    df['Value'] = df.apply(clean_value, axis=1)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], errors='coerce')
    df.dropna(inplace=True)
    df = df.sort_values(by=['Metric', 'Channel', 'Date'])
    df_prev = df.copy(); df_prev['Date'] += pd.DateOffset(years=1)
    df = pd.merge(df, df_prev[['Metric', 'Channel', 'Date', 'Value']], on=['Metric', 'Channel', 'Date'], how='left', suffixes=('', '_prev'))
    df['YoY Change'] = df['Value'] - df['Value_prev']
    return df.sort_values(by='Date').reset_index(drop=True)

# --- AI ANALYTICS ENGINE (Omitted for brevity, same as previous version) ---
class AIDashboardGenerator:
    @staticmethod
    def get_data_summary_for_ai(df, metric, year):
        df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
        if df_metric.empty: return "No data available."
        summary_parts = []
        for channel in ['Overall', 'Amazon', 'B2B']:
            if channel not in df_metric['Channel'].unique(): continue
            channel_df = df_metric[df_metric['Channel'] == channel]
            latest = channel_df.sort_values('Date').iloc[-1]
            summary_parts.append(f"- For **{channel}**: Latest value is **{latest['Value']:.2f}** ({latest['Date']:%b %Y}), with a YoY change of **{latest['YoY Change']:.2f}**.")
        return "\n".join(summary_parts)
    @staticmethod
    def get_ai_layout(data_summary: str, metric_name: str, model_choice: str, api_key: str) -> Dict[str, Any]:
        client, model_name_api = (anthropic.Anthropic(api_key=api_key), "claude-3-5-sonnet-20240620") if "Claude" in model_choice else (openai.OpenAI(api_key=api_key), "gpt-4o")
        prompt = f"""You are a Principal Data Analyst reporting to a business owner. Your analysis must be sharp, concise, and business-focused.
        **Task**: 1. Analyze the following data summary for the KPI: **{metric_name}**. 2. **Compare the performance between channels.** This is the most critical part of your analysis. 3. Generate a bulleted list of 1-3 actionable insights. Address the owner directly (e.g., "Your return rate..."). 4. Create a JSON object containing two keys: "insights" and "layout". The layout must present the data to support your insights.
        **Data Summary**: {data_summary}
        **JSON Output Structure**: - `insights`: A list of your string insights. - `layout`: An array of dashboard components. Available components: "kpi_summary", "line_chart".
        """
        try:
            if "Claude" in model_choice:
                response = client.messages.create(model=model_name_api, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
                content = response.content[0].text
            else:
                response = client.chat.completions.create(model=model_name_api, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": f"AI interaction failed: {e}"}

# --- UI RENDERING FUNCTIONS ---
def render_kpi_details(df: pd.DataFrame, metric: str, year: int):
    st.subheader("📌 Key Metrics")
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
    channels = sorted([c for c in df_metric['Channel'].unique() if c != 'Overall']); channels.insert(0, 'Overall')
    lower_is_better = any(term in metric.lower() for term in ['rate', 'cost', 'time'])
    for channel in channels:
        if channel not in df_metric['Channel'].unique(): continue
        latest = df_metric[df_metric['Channel'] == channel].sort_values('Date').iloc[-1]
        yoy_change = latest['YoY Change']
        is_good = pd.notna(yoy_change) and ((yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better))
        icon = "✅" if is_good else "⚠️"
        is_percent, is_csat = '%' in metric, 'csat' in metric.lower()
        if is_percent: val_f, delta_f = "{:,.1%}", "{:+.1f} pts"
        elif is_csat: val_f, delta_f = "{:,.2f} ⭐", "{:+.2f}"
        else: val_f, delta_f = "{:,.0f}", "{:+.0f}"
        delta_display = "No prior data"
        if pd.notna(yoy_change): delta_display = f"{icon} {delta_f.format(yoy_change)} vs. PY"
        st.metric(label=f"{channel} ({latest['Date']:%b %Y})", value=val_f.format(latest['Value']), delta=delta_display, delta_color="off")
        if channel != channels[-1]: st.markdown("---")

def render_line_chart(df: pd.DataFrame, title: str, metric: str, year: int, tv_mode=False):
    if not tv_mode: st.subheader(f"📊 {title}")
    df_metric = df[df['Metric'] == metric]
    df_curr, df_prev = df_metric[df_metric['Year'] == year], df_metric[df_metric['Year'] == year - 1]
    if df_curr.empty: return
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    line_width, marker_size = (6, 12) if tv_mode else (4, 6)
    
    df_overall_prev = df_prev[df_prev['Channel'] == 'Overall'].sort_values('Date')
    if not df_overall_prev.empty:
        fig.add_trace(go.Scatter(x=df_overall_prev['Date'].dt.month, y=df_overall_prev['Value'], name=f'Overall ({year-1})', mode='lines', line=dict(color="#999999", width=line_width-2, dash='dash')))

    df_overall_curr = df_curr[df_curr['Channel'] == 'Overall'].sort_values('Date')
    if not df_overall_curr.empty:
        fig.add_trace(go.Scatter(x=df_overall_curr['Date'].dt.month, y=df_overall_curr['Value'], name=f'Overall ({year})', mode='lines+markers', line=dict(color="#3b82f6", width=line_width), marker=dict(size=marker_size), fill='tonexty', fillcolor='rgba(59,130,246,0.1)'))

    if not tv_mode:
        for i, channel in enumerate(sorted([c for c in df_curr['Channel'].unique() if c != 'Overall'])):
            df_ch = df_curr[df_curr['Channel'] == channel].sort_values('Date')
            fig.add_trace(go.Scatter(x=df_ch['Date'].dt.month, y=df_ch['Value'], name=channel, mode='lines', line=dict(color=colors[i+1], width=2), opacity=0.7))

    goal_map = {"Overall Return Rate (%)": 0.045, "First Contact Resolution (%)": 0.85, "Customer Satisfaction (CSAT)": 4.5}
    if metric in goal_map:
        fig.add_hline(y=goal_map[metric], line_dash="dot", annotation_text="Goal", annotation_position="bottom right", line_color="gray")

    font_size = 20 if tv_mode else 12
    template = "plotly_white" if tv_mode else "plotly_dark"
    fig.update_layout(template=template, font=dict(size=font_size), legend=dict(font_size=font_size),
                      yaxis_tickformat=('.1%' if '%' in metric else '.2f' if 'csat' in metric.lower() else ',.0f'),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['J','F','M','A','M','J','J','A','S','O','N','D']), margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

def render_insights_card(insights: List[str]):
    st.subheader("💡 AI-Powered Insights")
    container = st.container(border=True)
    for insight in insights: container.markdown(f"&bull; {insight}")

def render_presentation_view(df: pd.DataFrame, metric: str, year: int):
    st.title(metric)
    
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year) & (df['Channel'] == 'Overall')]
    if df_metric.empty: st.header("No data for this period."); return

    latest = df_metric.sort_values('Date').iloc[-1]
    yoy_change = latest['YoY Change']
    lower_is_better = any(term in metric.lower() for term in ['rate', 'cost', 'time'])
    is_good = pd.notna(yoy_change) and ((yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better))
    icon = "✅" if is_good else "⚠️"
    
    is_percent, is_csat = '%' in metric, 'csat' in metric.lower()
    if is_percent: val_f, delta_f = "{:,.1%}", "{:+.1f} pts"
    elif is_csat: val_f, delta_f = "{:,.2f} ⭐", "{:+.2f}"
    else: val_f, delta_f = "{:,.0f}", "{:+.0f}"
    
    delta_display = ""
    if pd.notna(yoy_change): delta_display = f"{icon} {delta_f.format(yoy_change)} vs. Prior Year"
    
    st.metric(label=f"Overall Performance ({latest['Date']:%b %Y})", value=val_f.format(latest['Value']), delta=delta_display, delta_color="off")
    
    st.markdown("---")
    render_line_chart(df, "", metric, year, tv_mode=True)


# --- MAIN APP ---
df = load_and_transform_data()
if df.empty: st.warning("Data could not be loaded."); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Dashboard Settings")
    display_mode = st.radio("Display Mode", ["Interactive", "Presentation (TV)"], key="display_mode")
    
    st.markdown("---")
    
    if st.session_state.display_mode == "Interactive":
        st.header("View Controls")
        analysis_mode = st.radio("Analysis Mode", ["Curated", "AI-Generated"], label_visibility="collapsed")
        st.markdown("---")
        st.header("Filters")
        if analysis_mode == "Curated":
            selected_metric = st.selectbox("Select KPI to Analyze", sorted(df['Metric'].unique()))
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True), label_visibility="collapsed")
        
        if analysis_mode == "AI-Generated":
            st.markdown("---")
            st.header("AI Analysis")
            ai_metric = st.selectbox("Select KPI for AI Analysis", sorted(df['Metric'].unique()))
            with st.expander("AI Model Configuration"):
                model_choice = st.radio("Choose AI Model", ["GPT-4o", "Claude 3.5 Sonnet"], horizontal=True, label_visibility="collapsed")
            secret_key_name = "OPENAI_API_KEY" if "GPT" in model_choice else "ANTHROPIC_API_KEY"
            api_key_to_use = st.secrets.get(secret_key_name)
            if not api_key_to_use: st.warning(f"Set `{secret_key_name}` in secrets.")

# --- MAIN PANEL DISPLAY ---
apply_styling(st.session_state.get("display_mode", "Interactive"))

if st.session_state.display_mode == "Interactive":
    st.title("🚀 Business Quality Dashboard")
    if analysis_mode == "Curated":
        left_col, right_col = st.columns((2.5, 1))
        with left_col:
            render_line_chart(df, f"{selected_metric} Performance", selected_metric, selected_year)
        with right_col:
            render_kpi_details(df, selected_metric, selected_year)
            with st.expander("Show Raw Data"):
                st.dataframe(df[(df.Metric == selected_metric) & (df.Year == selected_year)], use_container_width=True)
    
    elif analysis_mode == "AI-Generated":
        if not api_key_to_use: st.info("Please configure an AI API key in your Streamlit secrets.")
        else:
            st.subheader(f"AI Analysis of: {ai_metric}")
            with st.spinner(f"🚀 Asking {model_choice} to analyze {ai_metric}..."):
                data_summary = AIDashboardGenerator.get_data_summary_for_ai(df, ai_metric, selected_year)
                ai_response = AIDashboardGenerator.get_ai_layout(data_summary, ai_metric, model_choice, api_key=api_key_to_use)
                if "error" in ai_response: st.error(f"AI Generation Failed: {ai_response['error']}")
                else:
                    render_insights_card(ai_response.get("insights", ["The AI did not provide any insights."]))
                    st.markdown("---")
                    for component in ai_response.get("layout", []):
                        params = component.get("params", {})
                        if component.get("type") == "line_chart": render_line_chart(df, **params)
                        elif component.get("type") == "kpi_summary": render_kpi_details(df, **params)

elif st.session_state.display_mode == "Presentation (TV)":
    # --- SLIDESHOW LOGIC ---
    if 'metric_index' not in st.session_state:
        st.session_state.metric_index = 0
    
    all_metrics = sorted(df['Metric'].unique())
    current_metric_index = st.session_state.metric_index
    
    # Display the current metric
    metric_to_display = all_metrics[current_metric_index]
    year_to_display = df['Year'].max()
    render_presentation_view(df, metric_to_display, year_to_display)
    
    # Update index for the next run
    st.session_state.metric_index = (current_metric_index + 1) % len(all_metrics)
    
    # Pause and rerun for slideshow effect
    time.sleep(20)
    st.rerun()
