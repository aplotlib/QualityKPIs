# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from streamlit_gsheets import GSheetsConnection # No longer needed for now
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
    page_icon="📈",
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
    Parses a complex multi-table format from the embedded sample data
    and returns a clean, tidy DataFrame for analysis.
    """
    try:
        # --- Constants for Sheet Structure ---
        COL_METRIC_TITLE, COL_YEAR, COL_CHANNEL, COL_MONTHS_START = 2, 3, 4, 5

        # --- EMBEDDED SAMPLE DATA ---
        # This section replaces the live Google Sheets connection.
        # To switch back, remove this list and uncomment the 'st.connection' line.
        sample_raw_data = [
            ['', '', 'Overall Return Rate (%)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2024', 'Amazon', '5.1%', '5.3%', '5.0%', '4.8%', '4.9%', '5.2%'],
            ['', '', '', '2024', 'Walmart', '6.2%', '6.0%', '6.1%', '5.9%', '5.8%', '6.3%'],
            ['', '', '', '2024', 'Overall', '5.5%', '5.6%', '5.4%', '5.2%', '5.3%', '5.7%'],
            ['', '', '', '2023', 'Amazon', '4.8%', '4.9%', '4.7%', '4.6%', '4.7%', '4.8%', '4.9%', '5.0%', '5.1%', '5.2%', '5.3%', '5.4%'],
            ['', '', '', '2023', 'Walmart', '5.9%', '5.8%', '5.7%', '5.6%', '5.5%', '5.6%', '5.7%', '5.8%', '5.9%', '6.0%', '6.1%', '6.2%'],
            ['', '', '', '2023', 'Overall', '5.2%', '5.3%', '5.1%', '5.0%', '5.1%', '5.2%', '5.3%', '5.4%', '5.5%', '5.6%', '5.7%', '5.8%'],
            ['', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', 'Support Ticket Cost ($)', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['', '', '', '2024', 'Phone Support', '12.50', '12.55', '12.48', '12.60', '12.65', '12.70'],
            ['', '', '', '2024', 'Email Support', '8.10', '8.05', '8.15', '8.20', '8.25', '8.30'],
            ['', '', '', '2024', 'Overall', '10.30', '10.30', '10.32', '10.40', '10.45', '10.50'],
            ['', '', '', '2023', 'Phone Support', '11.80', '11.85', '11.90', '11.95', '12.00', '12.05', '12.10', '12.15', '12.20', '12.25', '12.30', '12.40'],
            ['', '', '', '2023', 'Email Support', '7.50', '7.55', '7.60', '7.65', '7.70', '7.75', '7.80', '7.85', '7.90', '7.95', '8.00', '8.05'],
            ['', '', '', '2023', 'Overall', '9.65', '9.70', '9.75', '9.80', '9.85', '9.90', '9.95', '10.00', '10.05', '10.10', '10.15', '10.23'],
        ]
        # Use the sample data to create a DataFrame
        raw_data = pd.DataFrame(sample_raw_data).fillna('').astype(str)

        # --- LIVE DATA CONNECTION (Currently disabled) ---
        # conn = st.connection("gsheets", type=GSheetsConnection)
        # raw_data = conn.read(worksheet=799906691, header=None).fillna('').astype(str)
        # ---

        all_metrics_data = []
        current_metric: Optional[str] = None
        current_months: List[str] = []
        current_year: Optional[int] = None

        for _, row_series in raw_data.iterrows():
            row = row_series.tolist()

            is_metric_title = row[COL_METRIC_TITLE] and not row[COL_YEAR] and not row[COL_CHANNEL]
            if is_metric_title:
                current_metric = row[COL_METRIC_TITLE]
                current_months, current_year = [], None
                continue

            is_month_header = row[COL_MONTHS_START] == 'Jan'
            if is_month_header:
                current_months = [m for m in row[COL_MONTHS_START:] if m and 'Total' not in m and 'AVG' not in m]
                continue

            if not (current_metric and current_months):
                continue

            if row[COL_YEAR].isnumeric():
                current_year = int(float(row[COL_YEAR]))

            channel = row[COL_CHANNEL] if row[COL_CHANNEL] else 'Overall'
            is_data_row = current_year is not None and (row[COL_CHANNEL] or row[COL_YEAR].isnumeric())

            if is_data_row:
                values = row[COL_MONTHS_START : COL_MONTHS_START + len(current_months)]
                for month, value in zip(current_months, values):
                    if value:
                        all_metrics_data.append({
                            'Metric': current_metric, 'Year': current_year, 'Channel': channel,
                            'Month': month, 'Value': value
                        })

        if not all_metrics_data:
            st.error("Could not parse any data from the embedded source.")
            return pd.DataFrame()

        df = pd.DataFrame(all_metrics_data)

        # --- Final Cleaning and Type Conversion ---
        pct_metrics = [m for m in df['Metric'].unique() if '%' in m]

        def clean_value(row: pd.Series) -> Optional[float]:
            val_str = str(row['Value']).replace('%', '').replace('$', '').replace(',', '')
            val_num = pd.to_numeric(val_str, errors='coerce')
            if pd.isna(val_num):
                return None
            return val_num / 100.0 if row['Metric'] in pct_metrics else val_num

        df['Value'] = df.apply(clean_value, axis=1)
        df.dropna(subset=['Value'], inplace=True)

        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        # --- Calculate YoY Change ---
        df = df.sort_values(by=['Metric', 'Channel', 'Date'])
        df_prev_year = df.copy()
        df_prev_year['Date'] = df_prev_year['Date'] + pd.DateOffset(years=1)
        df = pd.merge(
            df, df_prev_year[['Metric', 'Channel', 'Date', 'Value']],
            on=['Metric', 'Channel', 'Date'], how='left', suffixes=('', '_prev_year')
        )
        df['YoY Change'] = df['Value'] - df['Value_prev_year']

        return df.sort_values(by='Date').reset_index(drop=True)

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()


# --- AI DASHBOARD GENERATOR ---
class AIDashboardGenerator:
    """Uses Generative AI to dynamically create a dashboard layout."""

    @staticmethod
    def get_available_models() -> List[str]:
        """Checks for available AI models based on dependencies and secrets."""
        models = []
        if OPENAI_AVAILABLE and st.secrets.get("OPENAI_API_KEY"):
            models.append("GPT-4o")
        if ANTHROPIC_AVAILABLE and st.secrets.get("ANTHROPIC_API_KEY"):
            models.append("Claude 3.5 Sonnet")
        return models

    @staticmethod
    def get_ai_layout(data_summary: str, model_choice: str) -> Dict[str, Any]:
        """Prompts an AI model to return a JSON layout for the dashboard."""
        client, model_name = None, ""
        if "Claude" in model_choice and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=st.secrets.ANTHROPIC_API_KEY)
            model_name = "claude-3-5-sonnet-20240620"
        elif "GPT" in model_choice and OPENAI_AVAILABLE:
            client = openai.OpenAI(api_key=st.secrets.OPENAI_API_KEY)
            model_name = "gpt-4o"

        if not client:
            return {"error": "Selected AI model is not available or configured."}

        prompt = f"""
        You are a world-class data visualization expert designing a Streamlit dashboard.
        Based on the data summary below, create a layout for an executive KPI dashboard.
        Return ONLY a valid JSON object. Do not include any other text or markdown.

        The JSON must have a single key "layout", which is an array of component objects.
        Available component types and their required params:
        1. "title": {{"text": "Your Title"}}
        2. "kpi_summary": {{"metric": "Metric Name", "year": "Year"}}
        3. "line_chart": {{"title": "Chart Title", "metric": "Metric Name", "year": "Year"}}

        Data Summary:
        {data_summary}

        Design an insightful layout. A good design includes a title, a KPI summary for the
        most recent year, and a comparative line chart for a key metric.
        """

        try:
            content = ""
            if "Claude" in model_choice:
                response = client.messages.create(model=model_name, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
                content = response.content[0].text
            else: # OpenAI
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                content = response.choices[0].message.content

            if content.strip().startswith("```json"):
                content = content.strip()[7:-4]

            return json.loads(content)
        except Exception as e:
            return {"error": f"Failed to get or parse AI layout: {e}", "raw_response": str(content)}

    @staticmethod
    def render_dashboard(layout: Dict[str, Any], df: pd.DataFrame):
        """Renders Streamlit components based on the AI-generated layout."""
        if "error" in layout:
            st.error(f"AI Generation Failed: {layout['error']}")
            if 'raw_response' in layout and layout['raw_response']:
                st.code(layout['raw_response'], language="text")
            return

        for i, component in enumerate(layout.get("layout", [])):
            comp_type = component.get("type")
            params = component.get("params", {})
            try:
                if comp_type == "title":
                    st.title(params.get("text", "AI Generated Dashboard"))
                elif comp_type == "kpi_summary":
                    render_kpi_summary(df, params.get("metric"), int(params.get("year")))
                elif comp_type == "line_chart":
                    render_line_chart(df, params.get("title"), params.get("metric"), int(params.get("year")))
                if i < len(layout.get("layout", [])) -1:
                    st.markdown("---")
            except (KeyError, TypeError, ValueError) as e:
                st.error(f"Error rendering AI component: `{comp_type}` with params `{params}`. Reason: {e}")


# --- UI RENDERING FUNCTIONS ---
def render_kpi_summary(df: pd.DataFrame, metric: str, year: int):
    """Displays a grid of KPI metric cards for a given metric and year."""
    df_metric = df[(df['Metric'] == metric) & (df['Year'] == year)]
    if df_metric.empty:
        st.warning(f"No data available for '{metric}' in {year}.")
        return

    st.subheader(f"Executive Summary: {metric} ({year})")

    channels = sorted(df_metric['Channel'].unique())
    num_columns = min(len(channels), 4)
    cols = st.columns(num_columns) if channels else [st.container()]

    is_percent = '%' in metric
    is_currency = '$' in metric
    lower_is_better = 'rate' in metric.lower() or 'cost' in metric.lower()

    for i, channel in enumerate(channels):
        col = cols[i % num_columns]
        latest_data = df_metric[df_metric['Channel'] == channel].sort_values('Date').iloc[-1]

        value_format = "{:,.2%}" if is_percent else ("${:,.2f}" if is_currency else "{:,.0f}")
        val_display = value_format.format(latest_data['Value'])

        yoy_change = latest_data['YoY Change']
        delta_display = "No prior year data"
        if pd.notna(yoy_change):
            delta_format = "{:+.2f} pts" if is_percent else ("${:,.2f}" if is_currency else "{:+.0f}")
            delta_display = f"{delta_format.format(yoy_change)} vs. prior year"

        col.metric(
            label=f"{channel} ({latest_data['Date'].strftime('%b %Y')})",
            value=val_display,
            delta=delta_display,
            delta_color="inverse" if lower_is_better else "normal"
        )

def render_line_chart(df: pd.DataFrame, title: str, metric: str, year: int):
    """Displays a line chart comparing performance for a metric between years."""
    df_metric = df[df['Metric'] == metric]
    df_current_year = df_metric[df_metric['Year'] == year]
    df_previous_year = df_metric[df_metric['Year'] == year - 1]

    if df_current_year.empty:
        st.warning(f"No chart data available for '{metric}' in {year}.")
        return

    st.subheader(title)
    fig = go.Figure()

    channels = sorted(df_current_year['Channel'].unique())
    colors = px.colors.qualitative.Plotly

    for i, channel in enumerate(channels):
        color = colors[i % len(colors)]
        df_ch = df_current_year[df_current_year['Channel'] == channel].sort_values('Date')
        if not df_ch.empty:
            fig.add_trace(go.Scatter(x=df_ch['Date'].dt.month, y=df_ch['Value'], name=f'{channel} ({year})',
                                     mode='lines+markers', line=dict(color=color, width=3)))
            latest_pt = df_ch.iloc[-1]
            fig.add_annotation(x=latest_pt['Date'].month, y=latest_pt['Value'], text=f"{latest_pt['Value']:.2%}" if '%' in metric else f"${latest_pt['Value']:.2f}",
                               showarrow=True, arrowhead=2, ax=0, ay=-40, bordercolor="#c7c7c7", borderwidth=2, bgcolor="rgba(255,255,255,0.8)")

        df_ch_prev = df_previous_year[df_previous_year['Channel'] == channel].sort_values('Date')
        if not df_ch_prev.empty:
            fig.add_trace(go.Scatter(x=df_ch_prev['Date'].dt.month, y=df_ch_prev['Value'], name=f'{channel} ({year - 1})',
                                     mode='lines', line=dict(color=color, width=2, dash='dash')))

    is_percent = '%' in metric
    is_currency = '$' in metric
    yaxis_tickformat = '.1%' if is_percent else ('$,.2f' if is_currency else ',.0f')

    fig.update_layout(
        template="plotly_dark",
        yaxis_tickformat=yaxis_tickformat,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode='array', tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN APP LOGIC ---
st.title("📊 Quality Department KPI Dashboard")

df = load_and_transform_data()

if df.empty:
    st.warning("Data could not be loaded. Please check the data source and format.")
    st.stop()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Dashboard Controls")
    dashboard_mode = st.radio("Choose Dashboard Type", ["Curated View", "AI-Generated View"], help="Choose between a standard view or let an AI build a layout.")
    st.markdown("---")

    st.header("Filters")
    available_metrics = sorted(df['Metric'].unique())
    selected_metric = st.selectbox("Select KPI", available_metrics)

    available_years = sorted(df['Year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Year", available_years)

    st.markdown("---")

    if dashboard_mode == "AI-Generated View":
        st.header("AI Configuration")
        available_models = AIDashboardGenerator.get_available_models()
        if not available_models:
            st.warning("No AI models configured. Add API keys to secrets to enable.")
            model_choice = None
        else:
            model_choice = st.selectbox("Choose AI Model", available_models)

    if st.button("🔄 Refresh Data"):
        with st.spinner("Clearing cache and refreshing data..."):
            st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='footer'><p>For questions, contact the <a href='mailto:your.email@example.com'>Quality Dept</a>.</p></div>", unsafe_allow_html=True)


# --- MAIN PANEL DISPLAY ---
if dashboard_mode == "Curated View":
    render_kpi_summary(df, selected_metric, selected_year)
    st.markdown("---")
    render_line_chart(df, f"Performance Trends: {selected_year} vs. {selected_year - 1}", selected_metric, selected_year)

elif dashboard_mode == "AI-Generated View":
    if not model_choice:
        st.info("Please select a configured AI model from the sidebar to generate a dashboard.")
    else:
        with st.spinner(f"🤖 Asking {model_choice} to design the dashboard..."):
            data_summary = f"Available Metrics: {df['Metric'].unique().tolist()}. Available Years: {df['Year'].unique().tolist()}."
            ai_layout = AIDashboardGenerator.get_ai_layout(data_summary, model_choice)
            AIDashboardGenerator.render_dashboard(ai_layout, df)
