# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Optional, Any
import io

# --- Dependency Checks ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    
try:
    from streamlit_gsheets import GSheetsConnection
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intelligent Quality Dashboard",
    page_icon="🧠",
    layout="wide",
)

# --- STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 2rem; }
    h1, h2 { text-align: center; font-weight: 700; }
    h3 { text-align: left; font-weight: 600; border-bottom: 1px solid #262730; padding-bottom: 0.5rem; margin-top: 1rem; margin-bottom: 1rem; }
    .stMetric { background-color: #0E1117; border: 1px solid #262730; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# --- AI-POWERED PARSING AND ANALYSIS ENGINE ---
class AI_Data_Processor:
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            st.error("The `openai` library is not installed. Please add it to your requirements.txt.")
            st.stop()
        self.client = openai.OpenAI(api_key=api_key)

    def parse_with_ai(self, file_preview: str) -> Optional[pd.DataFrame]:
        """Uses AI to convert a raw text preview of a file into a tidy DataFrame."""
        prompt = f"""
        You are an expert data analyst. Your task is to parse the following raw text data, which is a preview from a spreadsheet, and transform it into a clean, "tidy" JSON format.

        Instructions:
        1.  Analyze the structure of the data below. Identify the time-based columns (Year, Month), metric names, and their corresponding values.
        2.  Ignore any empty rows or purely presentational headers like "Quality".
        3.  The goal is a tidy format: each row is a single observation.
        4.  Return a JSON object with a single key, "data", which contains a list of dictionaries. Each dictionary must have the keys: "Metric", "Year", "Month", and "Value".
        5.  Ensure 'Year' is a 4-digit integer and 'Value' is a number (integer or float), stripping any characters like '$' or '%'. Handle variations in metric names, like standardizing "Overall Return Rate" and "% Reworks".

        Raw Data Preview:
        ```
        {file_preview}
        ```
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content)
            tidy_data = parsed_json.get("data", [])
            if not tidy_data or not isinstance(tidy_data, list):
                st.error("AI parsing failed: The AI did not return the expected data structure.")
                return None
            
            df = pd.DataFrame(tidy_data)
            required_cols = {"Metric", "Year", "Month", "Value"}
            if not required_cols.issubset(df.columns):
                st.error(f"AI parsing failed: Returned data is missing required columns. Found: {df.columns.tolist()}")
                return None

            return df
        except Exception as e:
            st.error(f"An error occurred during AI parsing: {e}")
            return None

    def generate_insights(self, data_summary: str, metric_name: str) -> Optional[List[str]]:
        """Generates analytical insights based on a data summary."""
        prompt = f"""As a Principal Data Analyst, provide 1-2 sharp, actionable insights for a business owner based on the following data summary for the KPI '{metric_name}'. Address the owner directly and be concise. Return your response as a JSON object with a single key "insights" which contains a list of strings.
        Data Summary: {data_summary}"""
        try:
            response = self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            insights_data = json.loads(response.choices[0].message.content)
            return insights_data.get("insights", ["AI returned an unexpected format."])
        except openai.AuthenticationError:
            return ["Authentication Error: Your OpenAI API key is invalid or has expired."]
        except Exception as e:
            return [f"AI interaction failed: {e}"]


# --- DATA LOADING AND PROCESSING ---
def load_data_source(source: Any) -> Optional[pd.DataFrame]:
    """Reads an uploaded file (CSV, XLSX) or Google Sheet into a pandas DataFrame."""
    try:
        if isinstance(source, str) and "docs.google.com/spreadsheets" in source:
            if not GSHEETS_AVAILABLE: st.error("GSheets library not found. Please add `streamlit-gsheets-connection` to requirements.txt"); return None
            conn = st.connection("gsheets", type=GSheetsConnection)
            return conn.read(spreadsheet=source, header=None)
        elif hasattr(source, 'name') and source.name.endswith('.csv'):
            return pd.read_csv(source, header=None)
        elif hasattr(source, 'name') and source.name.endswith(('.xlsx', '.xls')):
            if not OPENPYXL_AVAILABLE: st.error("`openpyxl` library not found. Please add it to your requirements.txt"); return None
            return pd.read_excel(source, header=None)
        return None
    except Exception as e:
        st.error(f"Failed to read the data source: {e}"); return None

def process_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a clean DataFrame from the AI and performs final transformations like YoY calculations."""
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Value', 'Year', 'Month'], inplace=True)
    
    # Scale percentage values correctly
    pct_metrics = [m for m in df['Metric'].unique() if '%' in m or 'rate' in m.lower()]
    df.loc[df['Metric'].isin(pct_metrics) & (df['Value'] > 1), 'Value'] /= 100.0
    
    df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'], format='%Y-%b', errors='coerce')
    df.sort_values(by=['Metric', 'Date'], inplace=True)

    df['YoY Change'] = df.groupby('Metric')['Value'].diff(12)
    return df.reset_index(drop=True)


# --- UI RENDERING FUNCTIONS ---
def render_kpi_details(df: pd.DataFrame, metric: str, year: int):
    st.subheader("📌 Key Metrics")
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
    if df_metric.empty: st.warning(f"No data for '{metric}' in {year}."); return
    
    latest = df_metric.sort_values('Date').iloc[-1]
    yoy_change = latest['YoY Change']
    lower_is_better = any(term in metric.lower() for term in ['rate', 'cost', 'rework', 'unresolved', 'return'])
    is_good = pd.notna(yoy_change) and ((yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better))
    icon = "✅" if is_good else "⚠️"
    
    is_percent = '%' in metric or 'rate' in metric.lower()
    is_currency = '$' in metric
    if is_percent: val_f, delta_f = "{:,.2%}", "{:+.2f} pts"
    elif is_currency: val_f, delta_f = "${:,.2f}", "{:+.2f}"
    else: val_f, delta_f = "{:,.0f}", "{:+.0f}"
    
    delta_display = "No prior data"
    if pd.notna(yoy_change): delta_display = f"{icon} {delta_f.format(yoy_change)} vs. PY"
    st.metric(label=f"Latest Performance ({latest['Date']:%b %Y})", value=val_f.format(latest['Value']), delta=delta_display, delta_color="off")

def render_chart(df: pd.DataFrame, metric: str, year: int):
    st.subheader(f"📊 Visual Analysis")
    
    issue_metrics = ["Full replacements (same day)", "Replacement parts (next day)", "Returns (w/in 3 days)", "Other cases: unresolved"]
    is_issue_category = metric in issue_metrics

    if is_issue_category:
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
        
        is_percent = '%' in metric or 'rate' in metric.lower()
        is_currency = '$' in metric
        yaxis_tickformat = '.1%' if is_percent else ('$,.2f' if is_currency else ',.0f')
        fig.update_layout(template="plotly_dark", yaxis_tickformat=yaxis_tickformat, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(tickformat="%b %Y"), margin=dict(t=0, b=20, l=40, r=20))
        
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.title("🧠 Intelligent Quality Dashboard")

# --- Step 1: Get Data Source ---
st.subheader("1. Provide Your Data Source")
input_method = st.radio("Choose input method:", ["Upload File", "Google Sheet URL"], horizontal=True, label_visibility="collapsed")

data_source = None
if input_method == "Upload File":
    data_source = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx', 'xls'])
else:
    if GSHEETS_AVAILABLE: data_source = st.text_input("Enter your public Google Sheet URL")
    else: st.warning("Please install `streamlit-gsheets-connection` to use this feature.")

if not data_source: st.info("Please provide a data source to begin."); st.stop()

# --- Step 2: AI-Powered Parsing ---
st.subheader("2. Analyze Data Structure with AI")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
if not api_key: st.warning("Please add your `OPENAI_API_KEY` to your Streamlit secrets to enable analysis."); st.stop()

if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None

if st.button("🤖 Analyze with AI", disabled=not data_source, use_container_width=True, type="primary"):
    with st.spinner("AI is analyzing the structure of your file... Please wait."):
        raw_df = load_data_source(data_source)
        if raw_df is not None:
            file_preview = raw_df.head(50).to_string()
            processor = AI_Data_Processor(api_key=api_key)
            st.session_state.cleaned_df = processor.parse_with_ai(file_preview)
            st.session_state.ai_insights = None # Clear old insights

# --- Step 3: Display Dashboard ---
if st.session_state.cleaned_df is not None:
    st.success("✅ AI analysis complete. Your dashboard is ready.")
    st.markdown("---")
    df = process_and_transform(st.session_state.cleaned_df)

    with st.sidebar:
        st.header("Dashboard Controls")
        all_metrics = sorted(df['Metric'].unique())
        selected_metric = st.selectbox("Select KPI", all_metrics)
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
        st.markdown("---")
        st.header("Generate Insights")
        if st.button("💡 Get AI Insights", use_container_width=True):
            summary = AI_Data_Processor.get_data_summary(df, selected_metric, selected_year)
            with st.spinner("Generating insights..."):
                st.session_state.ai_insights = AI_Data_Processor(api_key).generate_insights(summary, selected_metric)
            st.session_state.insights_for_metric = selected_metric

    if 'ai_insights' in st.session_state and st.session_state.get('insights_for_metric') == selected_metric:
        st.subheader("💡 AI-Powered Insights")
        container = st.container(border=True)
        for insight in st.session_state.ai_insights: container.markdown(f"&bull; {insight}")
        st.markdown("---")

    left_col, right_col = st.columns((2.5, 1))
    with left_col:
        render_chart(df, selected_metric, selected_year)
    with right_col:
        render_kpi_details(df, selected_metric, selected_year)
        with st.expander("Show AI-Cleaned Data"):
            st.dataframe(df)
else:
    st.info("Click 'Analyze with AI' to process your data and build the dashboard.")
