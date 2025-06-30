import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import traceback
import json

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
    Connects to Google Sheets and robustly transforms the multi-table data
    into a single, clean DataFrame for analysis.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        raw_data = conn.read(worksheet=799906691, header=None).fillna('').astype(str)

        all_metrics_data = []
        current_metric = None
        current_months = []
        month_start_col = -1

        # Iterate through each row to parse the sheet contextually
        for index, row_series in raw_data.iterrows():
            row = row_series.tolist()
            
            # Condition 1: Is this a metric title row? (e.g., "Overall Return Rate")
            # It has text in Column C and is empty in D and E.
            if row[2] != '' and row[3] == '' and row[4] == '':
                current_metric = row[2]
                continue

            # Condition 2: Is this a month header row? (e.g., "Jan", "Feb", ...)
            # We look for "Jan" to identify the start of the months.
            if 'Jan' in row:
                try:
                    month_start_col = row.index('Jan')
                    # Filter out any summary columns like "Total/AVG"
                    current_months = [m for m in row[month_start_col:] if m != '' and 'Total' not in m and 'AVG' not in m]
                except ValueError:
                    # This row contains "Jan" but not as a distinct cell value, so ignore.
                    pass
                continue

            # Condition 3: Is this a data row?
            # It has a year in Column D and a channel in Column E.
            year_val = row[3]
            channel = row[4]
            if year_val.isnumeric() and channel != '' and current_metric is not None and current_months:
                values = row[month_start_col : month_start_col + len(current_months)]
                
                for month, value in zip(current_months, values):
                    if value != '':
                        all_metrics_data.append({
                            'Metric': current_metric,
                            'Year': int(float(year_val)),
                            'Channel': channel,
                            'Month': month,
                            'Value': value
                        })

        if not all_metrics_data:
            st.error("Could not parse any data from the sheet.")
            return pd.DataFrame()

        df = pd.DataFrame(all_metrics_data)

        pct_metrics = [m for m in df['Metric'].unique() if '%' in m]
        
        def clean_value(row):
            val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
            val_num = pd.to_numeric(val_str, errors='coerce')
            return val_num / 100.0 if row['Metric'] in pct_metrics else val_num
            
        df['Value'] = df.apply(clean_value, axis=1)
        df.dropna(subset=['Value'], inplace=True)
        
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'])
        
        df = df.sort_values(by=['Metric', 'Channel', 'Date'])
        df['MoM Change'] = df.groupby(['Metric', 'Channel'])['Value'].diff()
        df['YoY Change'] = df.groupby(['Metric', 'Channel', df['Date'].dt.month])['Value'].diff()
        
        return df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

# --- AI DASHBOARD GENERATOR ---
class AIDashboardGenerator:
    """Uses AI to dynamically generate a dashboard layout from data."""

    @staticmethod
    def get_ai_layout(data_summary, model_choice):
        """Prompts an AI model to return a JSON layout for a Streamlit dashboard."""
        api_key = None
        client = None
        model_name = ""

        if "Claude" in model_choice and ANTHROPIC_AVAILABLE:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=api_key)
            model_name = "claude-3-5-sonnet-20240620"
        elif "GPT" in model_choice and OPENAI_AVAILABLE:
            api_key = st.secrets.get("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)
            model_name = "gpt-4o"
        
        if not client:
            return {"error": "Selected AI model is not available or configured."}

        prompt = f"""
        You are a world-class data visualization expert and Streamlit developer. Your task is to design a dashboard layout based on the provided data summary.
        Return ONLY a valid JSON object that represents the dashboard layout. Do not include any other text, explanations, or markdown formatting.

        The JSON object must have a single key "layout" which is an array of component objects.
        Each component object must have a "type" and "params".

        Available component types and their required params:
        1. "title": {{ "text": "Your Title" }}
        2. "kpi_summary": {{ "metric": "Metric Name", "year": "Year" }} - This will create a 2-column summary for Amazon and B2B.
        3. "line_chart": {{ "title": "Chart Title", "metric": "Metric Name", "compare_years": true }}

        Here is the summary of the available data:
        {data_summary}

        Based on this data, design an insightful and clean dashboard layout. A good layout might include a title, a KPI summary for the most recent year, and a comparative line chart for a key metric.
        """
        
        try:
            if "Claude" in model_choice:
                response = client.messages.create(model=model_name, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
                content = response.content[0].text
            else: # OpenAI
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = response.choices[0].message.content

            return json.loads(content)
        except Exception as e:
            return {"error": f"Failed to get or parse AI layout: {e}", "raw_response": str(content)}

    @staticmethod
    def render_dashboard(layout, df):
        """Renders Streamlit components based on the AI-generated layout JSON."""
        if "error" in layout:
            st.error(f"AI Generation Failed: {layout['error']}")
            if 'raw_response' in layout:
                st.code(layout['raw_response'], language="text")
            return

        for component in layout.get("layout", []):
            comp_type = component.get("type")
            params = component.get("params", {})
            
            if comp_type == "title":
                st.title(params.get("text", "AI Generated Dashboard"))
            
            elif comp_type == "kpi_summary":
                render_kpi_summary(df, params.get("metric"), int(params.get("year")))
            
            elif comp_type == "line_chart":
                render_line_chart(df, params.get("title"), params.get("metric"), int(params.get("year")))
            
            st.markdown("---")

# --- UI RENDERING FUNCTIONS ---
def render_kpi_summary(df, metric, year):
    df_metric = df[df['Metric'] == metric]
    df_filtered = df_metric[df_metric['Year'] == year]
    
    st.subheader(f"Executive Summary: {metric} ({year})")
    
    latest_amazon = df_filtered[df_filtered['Channel'] == 'Amazon'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'Amazon'].empty else None
    latest_b2b = df_filtered[df_filtered['Channel'] == 'B2B'].iloc[-1] if not df_filtered[df_filtered['Channel'] == 'B2B'].empty else None

    is_percent = '%' in metric
    is_currency = 'cost' in metric.lower()
    value_format = "{:,.2%}" if is_percent else ("${:,.2f}" if is_currency else "{:,.0f}")
    lower_is_better = 'rate' in metric.lower() or 'cost' in metric.lower()

    col1, col2 = st.columns(2)
    with col1:
        if latest_amazon is not None:
            st.metric(label=f"Amazon ({latest_amazon['Month']})", value=value_format.format(latest_amazon['Value']))
            yoy_change = latest_amazon['YoY Change']
            if pd.notna(yoy_change):
                yoy_color = "green" if (yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better) else "red"
                st.markdown(f"<p style='color:{yoy_color};'>{yoy_change:+.2%} vs. Last Year</p>", unsafe_allow_html=True)
    with col2:
        if latest_b2b is not None:
            st.metric(label=f"B2B ({latest_b2b['Month']})", value=value_format.format(latest_b2b['Value']))
            yoy_change = latest_b2b['YoY Change']
            if pd.notna(yoy_change):
                yoy_color = "green" if (yoy_change < 0 and lower_is_better) or (yoy_change > 0 and not lower_is_better) else "red"
                st.markdown(f"<p style='color:{yoy_color};'>{yoy_change:+.2%} vs. Last Year</p>", unsafe_allow_html=True)

def render_line_chart(df, title, metric, year):
    df_metric = df[df['Metric'] == metric]
    df_current_year = df_metric[df_metric['Year'] == year]
    df_previous_year = df_metric[df_metric['Year'] == year - 1]
    
    st.subheader(title)
    fig = go.Figure()

    for channel, color in [('Amazon', '#ff9900'), ('B2B', '#1c83e1')]:
        # Current year
        df_ch = df_current_year[df_current_year['Channel'] == channel]
        if not df_ch.empty:
            fig.add_trace(go.Scatter(x=df_ch['Date'].dt.month, y=df_ch['Value'], name=f'{channel} ({year})', mode='lines+markers', line=dict(color=color, width=3)))
        # Previous year
        df_ch_prev = df_previous_year[df_previous_year['Channel'] == channel]
        if not df_ch_prev.empty:
            fig.add_trace(go.Scatter(x=df_ch_prev['Date'].dt.month, y=df_ch_prev['Value'], name=f'{channel} ({year - 1})', mode='lines', line=dict(color=color, width=2, dash='dash')))

    is_percent = '%' in metric
    is_currency = 'cost' in metric.lower()
    
    fig.update_layout(template="plotly_dark", yaxis_tickformat=('.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"), xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP LOGIC ---
st.title("📊 Quality Department KPI Dashboard")

df = load_and_transform_data()

if not df.empty:
    with st.sidebar:
        st.header("Dashboard Mode")
        dashboard_mode = st.radio("Choose Dashboard Type", ["Curated Dashboard", "AI-Generated Dashboard"], label_visibility="collapsed")
        st.markdown("---")

        st.header("Controls")
        available_metrics = sorted(df['Metric'].unique())
        selected_metric = st.selectbox("Select KPI", available_metrics)
        available_years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Year", available_years)
        
        st.markdown("---")
        st.header("AI Configuration")
        available_models = AIAnalyst.get_available_models()
        model_choice = st.selectbox("Choose AI Model", available_models) if available_models else None
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("<div class='footer'><p>For questions or feedback, please <a href='mailto:alexander.popoff@vivehealth.com'>contact the Quality Dept</a>.</p></div>", unsafe_allow_html=True)

    if dashboard_mode == "Curated Dashboard":
        render_kpi_summary(df, selected_metric, selected_year)
        st.markdown("---")
        render_line_chart(df, f"Performance Trends: {selected_year} vs. {selected_year - 1}", selected_metric, selected_year)
    
    elif dashboard_mode == "AI-Generated Dashboard":
        if not model_choice:
            st.warning("Please select an AI model in the sidebar to generate a dashboard.")
        else:
            with st.spinner(f"🤖 Asking {model_choice} to design the dashboard..."):
                data_summary = f"Available Metrics: {df['Metric'].unique().tolist()}. Available Years: {df['Year'].unique().tolist()}."
                ai_layout = AIDashboardGenerator.get_ai_layout(data_summary, model_choice)
                AIDashboardGenerator.render_dashboard(ai_layout, df)

else:
    st.warning("Data could not be loaded. Please check the Google Sheet is shared correctly and the format is as expected.")
