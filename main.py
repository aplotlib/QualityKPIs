# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import json
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
    page_icon="🤖",
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
    /* Main title */
    h1 {
        text-align: center;
        padding-bottom: 1rem;
    }
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #a0a4b8;
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
@st.cache_data(ttl=3600)
def load_and_transform_data() -> pd.DataFrame:
    """
    Parses a complex multi-table format from the expanded embedded sample data
    and returns a clean, tidy DataFrame for analysis.
    """
    try:
        # --- Constants for Sheet Structure ---
        COL_METRIC_TITLE, COL_YEAR, COL_CHANNEL, COL_MONTHS_START = 2, 3, 4, 5

        # --- EXPANDED EMBEDDED SAMPLE DATA ---
        sample_raw_data = [
            ['', '', 'Overall Return Rate (%)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2025', 'Amazon', '4.5%', '4.6%', '4.4%', '4.5%', '4.7%', '4.6%'],
            ['', '', '', '2025', 'Walmart', '5.8%', '5.9%', '5.7%', '5.6%', '5.8%', '5.7%'],
            ['', '', '', '2025', 'Overall', '5.0%', '5.1%', '4.9%', '4.9%', '5.1%', '5.0%'],
            ['', '', '', '2024', 'Amazon', '5.1%', '5.3%', '5.0%', '4.8%', '4.9%', '5.2%', '5.1%', '4.9%', '5.0%', '5.2%', '5.4%', '5.3%'],
            ['', '', '', '2024', 'Walmart', '6.2%', '6.0%', '6.1%', '5.9%', '5.8%', '6.3%', '6.2%', '6.1%', '6.0%', '6.2%', '6.4%', '6.3%'],
            ['', '', '', '2024', 'Overall', '5.5%', '5.6%', '5.4%', '5.2%', '5.3%', '5.7%', '5.6%', '5.4%', '5.4%', '5.6%', '5.8%', '5.7%'],
            ['', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', 'Customer Satisfaction (CSAT)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2025', 'Phone Support', '4.3', '4.4', '4.2', '4.5', '4.4', '4.5'],
            ['', '', '', '2025', 'Email Support', '4.6', '4.7', '4.6', '4.7', '4.8', '4.7'],
            ['', '', '', '2025', 'Overall', '4.4', '4.5', '4.4', '4.6', '4.6', '4.6'],
            ['', '', '', '2024', 'Phone Support', '4.1', '4.2', '4.0', '4.2', '4.3', '4.1', '4.2', '4.3', '4.4', '4.2', '4.3', '4.4'],
            ['', '', '', '2024', 'Email Support', '4.5', '4.6', '4.5', '4.4', '4.5', '4.6', '4.7', '4.6', '4.5', '4.6', '4.7', '4.8'],
            ['', '', '', '2024', 'Overall', '4.3', '4.4', '4.2', '4.3', '4.4', '4.3', '4.4', '4.4', '4.4', '4.4', '4.5', '4.6'],
            ['', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', 'First Contact Resolution (%)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2025', 'Phone Support', '78%', '79%', '77%', '80%', '81%', '80%'],
            ['', '', '', '2025', 'Email Support', '88%', '89%', '88%', '90%', '91%', '90%'],
            ['', '', '', '2025', 'Overall', '82%', '83%', '81%', '84%', '85%', '84%'],
            ['', '', '', '2024', 'Phone Support', '75%', '76%', '74%', '75%', '76%', '77%', '76%', '78%', '79%', '78%', '77%', '78%'],
            ['', '', '', '2024', 'Email Support', '85%', '86%', '86%', '87%', '88%', '87%', '88%', '89%', '90%', '89%', '88%', '89%'],
            ['', '', '', '2024', 'Overall', '79%', '80%', '79%', '80%', '81%', '81%', '81%', '82%', '83%', '82%', '81%', '82%'],
            ['', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', 'Average Handling Time (sec)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2025', 'Phone Support', '295', '298', '290', '285', '288', '292'],
            ['', '', '', '2025', 'Email Support', '450', '455', '448', '440', '445', '452'],
            ['', '', '', '2025', 'Overall', '373', '377', '369', '363', '367', '372'],
            ['', '', '', '2024', 'Phone Support', '310', '305', '308', '300', '299', '301', '303', '298', '295', '296', '294', '290'],
            ['', '', '', '2024', 'Email Support', '480', '475', '478', '470', '465', '468', '472', '460', '455', '458', '454', '450'],
            ['', '', '', '2024', 'Overall', '395', '390', '393', '385', '382', '385', '388', '379', '375', '377', '374', '370'],
        ]
        raw_data = pd.DataFrame(sample_raw_data).fillna('').astype(str)

        all_metrics_data = []
        current_metric: Optional[str] = None
        current_months: List[str] = []
        current_year: Optional[int] = None

        for _, row_series in raw_data.iterrows():
            row = row_series.tolist()
            if row[COL_METRIC_TITLE] and not row[COL_YEAR] and not row[COL_CHANNEL]:
                current_metric, current_months, current_year = row[COL_METRIC_TITLE], [], None
                continue
            if row[COL_MONTHS_START] == 'Jan':
                current_months = [m for m in row[COL_MONTHS_START:] if m and 'Total' not in m]
                continue
            if not (current_metric and current_months): continue
            if row[COL_YEAR].isnumeric(): current_year = int(float(row[COL_YEAR]))
            channel = row[COL_CHANNEL] if row[COL_CHANNEL] else 'Overall'
            if current_year and (row[COL_CHANNEL] or row[COL_YEAR].isnumeric()):
                for month, value in zip(current_months, row[COL_MONTHS_START:]):
                    if value:
                        all_metrics_data.append({'Metric': current_metric, 'Year': current_year, 'Channel': channel, 'Month': month, 'Value': value})
        
        df = pd.DataFrame(all_metrics_data)
        pct_metrics = [m for m in df['Metric'].unique() if '%' in m]
        
        def clean_value(row: pd.Series) -> Optional[float]:
            val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
            val_num = pd.to_numeric(val_str, errors='coerce')
            if pd.isna(val_num): return None
            return val_num / 100.0 if row['Metric'] in pct_metrics else val_num
        
        df['Value'] = df.apply(clean_value, axis=1)
        df.dropna(subset=['Value'], inplace=True)
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df = df.sort_values(by=['Metric', 'Channel', 'Date'])
        df_prev = df.copy()
        df_prev['Date'] += pd.DateOffset(years=1)
        df = pd.merge(df, df_prev[['Metric', 'Channel', 'Date', 'Value']], on=['Metric', 'Channel', 'Date'], how='left', suffixes=('', '_prev'))
        df['YoY Change'] = df['Value'] - df['Value_prev']
        return df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()


# --- AI DASHBOARD GENERATOR ---
class AIDashboardGenerator:
    """Uses Generative AI to dynamically create a dashboard layout."""
    @staticmethod
    def get_ai_layout(data_summary: str, model_choice: str, api_key: str) -> Dict[str, Any]:
        """Prompts an AI model to return a JSON layout for the dashboard."""
        client, model_name = None, ""
        try:
            if "Claude" in model_choice:
                if not ANTHROPIC_AVAILABLE: return {"error": "Anthropic library not installed."}
                client = anthropic.Anthropic(api_key=api_key)
                model_name = "claude-3-5-sonnet-20240620"
            elif "GPT" in model_choice:
                if not OPENAI_AVAILABLE: return {"error": "OpenAI library not installed."}
                client = openai.OpenAI(api_key=api_key)
                model_name = "gpt-4o"
            else:
                return {"error": "Invalid model choice."}
        except Exception as e:
            return {"error": f"Failed to initialize AI client: {e}. Is your API key correct?"}

        prompt = f"""
        You are a data visualization expert designing a Streamlit dashboard.
        Based on the data summary, create a layout. Return ONLY a valid JSON object.
        JSON: {{"layout": [{{"type": "component", "params": {{...}}}}]}}
        Components: "title", "kpi_summary", "line_chart".
        Params for kpi_summary & line_chart: "metric", "year".
        Data Summary: {data_summary}
        Design an insightful layout for the most recent year's data.
        """
        try:
            content = ""
            if "Claude" in model_choice:
                response = client.messages.create(model=model_name, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
                content = response.content[0].text
            else:
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = response.choices[0].message.content
            
            if content.strip().startswith("```json"):
                content = content.strip()[7:-4]
            return json.loads(content)
        except Exception as e:
            error_message = str(e)
            if "authentication" in error_message.lower():
                return {"error": "Authentication failed. Please check if your API key is correct and active."}
            return {"error": f"Failed to get or parse AI layout: {e}", "raw_response": str(content)}

    @staticmethod
    def render_dashboard(layout: Dict[str, Any], df: pd.DataFrame):
        if "error" in layout:
            st.error(f"AI Generation Failed: {layout['error']}")
            if 'raw_response' in layout and layout['raw_response']:
                st.code(layout['raw_response'], language="text")
            return
        for i, component in enumerate(layout.get("layout", [])):
            comp_type, params = component.get("type"), component.get("params", {})
            try:
                if comp_type == "title": st.title(params.get("text"))
                elif comp_type == "kpi_summary": render_kpi_summary(df, params.get("metric"), int(params.get("year")))
                elif comp_type == "line_chart": render_line_chart(df, params.get("title"), params.get("metric"), int(params.get("year")))
                if i < len(layout.get("layout", [])) - 1: st.markdown("---")
            except Exception as e:
                st.error(f"Error rendering AI component: `{comp_type}`. Reason: {e}")


# --- UI RENDERING FUNCTIONS ---
def render_kpi_summary(df: pd.DataFrame, metric: str, year: int):
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
    if df_metric.empty: st.warning(f"No data for '{metric}' in {year}."); return
    st.subheader(f"Executive Summary: {metric} ({year})")
    channels = sorted(df_metric['Channel'].unique())
    cols = st.columns(min(len(channels), 4)) if channels else [st.container()]
    lower_is_better = any(term in metric.lower() for term in ['rate', 'cost', 'time'])
    for i, channel in enumerate(channels):
        latest_data = df_metric[df_metric['Channel'] == channel].sort_values('Date').iloc[-1]
        is_percent, is_csat = '%' in metric, 'csat' in metric.lower()
        if is_percent: value_format, delta_format = "{:,.1%}", "{:+.1f} pts"
        elif is_csat: value_format, delta_format = "{:,.2f} ⭐", "{:+.2f}"
        else: value_format, delta_format = "{:,.0f}", "{:+.0f}"
        val_display = value_format.format(latest_data['Value'])
        delta_display = "No prior data"
        if pd.notna(latest_data['YoY Change']): delta_display = f"{delta_format.format(latest_data['YoY Change'])} vs. prior year"
        cols[i % len(cols)].metric(label=f"{channel} ({latest_data['Date']:%b %Y})", value=val_display, delta=delta_display, delta_color="inverse" if lower_is_better else "normal")

def render_line_chart(df: pd.DataFrame, title: str, metric: str, year: int):
    df_metric = df[df['Metric'] == metric]
    df_current, df_previous = df_metric[df_metric['Year'] == year], df_metric[df_metric['Year'] == year - 1]
    if df_current.empty: return
    st.subheader(title)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, channel in enumerate(sorted(df_current['Channel'].unique())):
        color = colors[i % len(colors)]
        df_ch_curr = df_current[df_current['Channel'] == channel].sort_values('Date')
        if not df_ch_curr.empty:
            fig.add_trace(go.Scatter(x=df_ch_curr['Date'].dt.month, y=df_ch_curr['Value'], name=f'{channel} ({year})', mode='lines+markers', line=dict(color=color, width=3)))
            latest = df_ch_curr.iloc[-1]
            anno_text = f"{latest['Value']:.1%}" if '%' in metric else f"{latest['Value']:.2f}" if 'csat' in metric.lower() else f"{latest['Value']:.0f}"
            fig.add_annotation(x=latest['Date'].month, y=latest['Value'], text=anno_text, showarrow=True, arrowhead=2, ax=0, ay=-40, bordercolor="#c7c7c7", borderwidth=2, bgcolor="rgba(255,255,255,0.8)")
        df_ch_prev = df_previous[df_previous['Channel'] == channel].sort_values('Date')
        if not df_ch_prev.empty:
            fig.add_trace(go.Scatter(x=df_ch_prev['Date'].dt.month, y=df_ch_prev['Value'], name=f'{channel} ({year - 1})', mode='lines', line=dict(color=color, width=2, dash='dash')))
    yaxis_tickformat = '.1%' if '%' in metric else '.2f' if 'csat' in metric.lower() else ',.0f'
    fig.update_layout(template="plotly_dark", yaxis_tickformat=yaxis_tickformat, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP LOGIC ---
st.title("🤖 Quality Department KPI Dashboard")
df = load_and_transform_data()
if df.empty: st.warning("Data could not be loaded."); st.stop()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Dashboard Controls")
    dashboard_mode = st.radio("Choose View", ["Curated", "AI-Generated"], label_visibility="collapsed")
    st.markdown("---")
    st.header("Filters")
    selected_metric = st.selectbox("Select KPI", sorted(df['Metric'].unique()))
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
    st.markdown("---")
    
    # --- AI Configuration Section ---
    if dashboard_mode == "AI-Generated":
        st.header("AI Configuration")
        model_choice = st.radio("Choose AI Model", ["GPT-4o", "Claude 3.5 Sonnet"], horizontal=True)
        
        # Determine the key for secrets (case-insensitive)
        secret_key_map = {"GPT-4o": "OPENAI_API_KEY", "Claude 3.5 Sonnet": "ANTHROPIC_API_KEY"}
        secret_key_name = secret_key_map[model_choice]

        # Allow user to input API key directly, overriding secrets
        user_api_key = st.text_input(
            label=f"Enter {model_choice} API Key",
            type="password",
            help=f"Your key is not stored. It overrides the secret key '{secret_key_name}' if provided."
        )
        
        api_key_to_use = None
        if user_api_key:
            api_key_to_use = user_api_key
            st.success("✅ Using API key provided above.")
        elif st.secrets.get(secret_key_name):
            api_key_to_use = st.secrets[secret_key_name]
            st.info(f"ℹ️ Using API key found in Streamlit secrets ('{secret_key_name}').")
        else:
            st.warning(f"⚠️ Please provide an API key above or set '{secret_key_name}' in your Streamlit secrets.")

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        with st.spinner("Clearing cache..."): st.cache_data.clear()
        st.rerun()
    st.markdown("<div class='footer'><p>For questions, contact the <a href='mailto:your.email@example.com'>Quality Dept</a>.</p></div>", unsafe_allow_html=True)

# --- MAIN PANEL DISPLAY ---
if dashboard_mode == "Curated":
    render_kpi_summary(df, selected_metric, selected_year)
    st.markdown("---")
    render_line_chart(df, f"Performance Trends: {selected_year} vs. {selected_year - 1}", selected_metric, selected_year)

elif dashboard_mode == "AI-Generated":
    st.header("AI-Generated Dashboard")
    if api_key_to_use:
        with st.spinner(f"🤖 Asking {model_choice} to design the dashboard..."):
            data_summary = f"Metrics: {df['Metric'].unique().tolist()}. Years: {df['Year'].unique().tolist()}."
            ai_layout = AIDashboardGenerator.get_ai_layout(data_summary, model_choice, api_key=api_key_to_use)
            AIDashboardGenerator.render_dashboard(ai_layout, df)
    else:
        st.info("Please configure an API key in the sidebar to generate an AI dashboard.")
