# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Dict, Optional
import numpy as np

# --- Dependency Checks ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("Please install openai: pip install openai")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-Powered Quality Dashboard",
    page_icon="🧠",
    layout="wide",
)

# --- STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 2rem; }
    h1 { text-align: center; font-weight: 700; color: #1f2937; }
    h2, h3 { font-weight: 600; color: #374151; }
    .metric-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- AI PROCESSOR ---
class AIQualityAnalyzer:
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        self.client = openai.OpenAI(api_key=api_key)
    
    def parse_quality_data(self, file_content: str) -> pd.DataFrame:
        """Parse CSV content and return structured DataFrame"""
        try:
            # For CSV files, we can parse directly
            import io
            df = pd.read_csv(io.StringIO(file_content))
            
            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]
            
            return df
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            return None
    
    def generate_insights(self, metrics_summary: Dict) -> List[str]:
        """Generate AI insights based on metrics"""
        prompt = f"""
        As a Quality Manager expert, analyze these quality metrics and provide 3-4 key insights:
        
        {json.dumps(metrics_summary, indent=2)}
        
        Focus on:
        1. Most concerning trends
        2. Improvements or deteriorations
        3. Actionable recommendations
        
        Be direct and specific. Format as a JSON object with key "insights" containing a list of strings.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("insights", ["Unable to generate insights"])
        except Exception as e:
            return [f"AI insight generation failed: {str(e)}"]
    
    def categorize_metrics(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Categorize metrics into logical groups"""
        categories = {
            "Quality & Returns": [],
            "Operations": [],
            "Financial": [],
            "Customer Service": []
        }
        
        for metric in metric_names:
            metric_lower = metric.lower()
            if any(term in metric_lower for term in ['return', 'defect', 'quality', 'rework']):
                categories["Quality & Returns"].append(metric)
            elif any(term in metric_lower for term in ['order', 'inspect', 'percent']):
                categories["Operations"].append(metric)
            elif any(term in metric_lower for term in ['cost', '$', 'price']):
                categories["Financial"].append(metric)
            else:
                categories["Customer Service"].append(metric)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

# --- DATA PROCESSING ---
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with YoY and MoM calculations"""
    # Convert data types
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Create proper date column
    df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'], 
                                format='%Y-%b', errors='coerce')
    
    # Remove invalid dates
    df = df[df['Date'].notna()]
    
    # Sort data
    df = df.sort_values(['Metric', 'Date'])
    
    # Calculate changes
    df['MoM_Change'] = df.groupby('Metric')['Value'].pct_change()
    df['YoY_Change'] = df.groupby('Metric')['Value'].pct_change(12)
    
    # Add trend indicator
    df['Trend'] = df.groupby('Metric')['Value'].transform(
        lambda x: 'Improving' if x.iloc[-1] < x.iloc[0] else 'Worsening'
    )
    
    return df

def calculate_summary_metrics(df: pd.DataFrame) -> Dict:
    """Calculate summary metrics for AI insights"""
    summary = {}
    
    latest_date = df['Date'].max()
    latest_year = latest_date.year
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        latest = metric_data[metric_data['Date'] == latest_date]
        
        if not latest.empty:
            latest_row = latest.iloc[0]
            summary[metric] = {
                'current_value': float(latest_row['Value']),
                'yoy_change': float(latest_row['YoY_Change']) if pd.notna(latest_row['YoY_Change']) else None,
                'mom_change': float(latest_row['MoM_Change']) if pd.notna(latest_row['MoM_Change']) else None,
                'trend': latest_row['Trend']
            }
    
    return summary

# --- VISUALIZATION FUNCTIONS ---
def create_trend_sparkline(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a small sparkline chart"""
    metric_data = df[df['Metric'] == metric].sort_values('Date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metric_data['Date'],
        y=metric_data['Value'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create YoY comparison chart"""
    metric_data = df[df['Metric'] == metric].sort_values('Date')
    current_year = metric_data['Year'].max()
    
    fig = go.Figure()
    
    # Add traces for each year
    for year in sorted(metric_data['Year'].unique()):
        year_data = metric_data[metric_data['Year'] == year]
        
        fig.add_trace(go.Scatter(
            x=year_data['Month'],
            y=year_data['Value'],
            name=str(int(year)),
            mode='lines+markers',
            line=dict(width=3 if year == current_year else 2),
            marker=dict(size=8 if year == current_year else 6)
        ))
    
    # Format based on metric type
    is_percent = any(term in metric.lower() for term in ['rate', 'percent', '%'])
    is_currency = any(term in metric.lower() for term in ['cost', '$'])
    
    yaxis_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
    
    fig.update_layout(
        title=f"{metric} - Year over Year Comparison",
        xaxis_title="Month",
        yaxis_title="Value",
        yaxis_tickformat=yaxis_format,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_metric_card(metric: str, current_value: float, change: float, change_type: str = "YoY") -> None:
    """Create a metric card with value and change"""
    # Determine formatting
    is_percent = any(term in metric.lower() for term in ['rate', 'percent', '%'])
    is_currency = any(term in metric.lower() for term in ['cost', '$'])
    
    if is_percent:
        value_str = f"{current_value:.1%}"
        change_str = f"{change:+.1%}" if change else "N/A"
    elif is_currency:
        value_str = f"${current_value:,.2f}"
        change_str = f"{change:+.1%}" if change else "N/A"
    else:
        value_str = f"{current_value:,.0f}"
        change_str = f"{change:+.1%}" if change else "N/A"
    
    # Determine if improvement
    lower_is_better = any(term in metric.lower() for term in ['cost', 'return', 'defect', 'rework'])
    is_good = (change < 0 and lower_is_better) or (change > 0 and not lower_is_better) if change else None
    
    # Color coding
    if is_good is None:
        delta_color = "gray"
        icon = "➖"
    elif is_good:
        delta_color = "green"
        icon = "✅"
    else:
        delta_color = "red"
        icon = "⚠️"
    
    st.metric(
        label=metric,
        value=value_str,
        delta=f"{icon} {change_str} {change_type}",
        delta_color="off"
    )

# --- MAIN APP ---
def main():
    st.title("🧠 AI-Powered Quality Dashboard")
    st.markdown("Intelligent analysis of quality metrics with automatic insights")
    
    # Check for API key
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Please add your OpenAI API key to Streamlit secrets")
        st.stop()
    
    # Initialize AI analyzer
    analyzer = AIQualityAnalyzer(api_key)
    
    # Load data
    try:
        with open('quality_data_clean.csv', 'r') as f:
            csv_content = f.read()
        
        with st.spinner("🤖 AI is analyzing your quality data..."):
            df = analyzer.parse_quality_data(csv_content)
            
            if df is None:
                st.stop()
            
            # Prepare data
            df = prepare_data(df)
            
            # Get metrics summary
            metrics_summary = calculate_summary_metrics(df)
            
            # Categorize metrics
            categories = analyzer.categorize_metrics(list(df['Metric'].unique()))
    
    except FileNotFoundError:
        st.error("quality_data_clean.csv not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Display AI insights at the top
    st.markdown("## 💡 AI-Generated Insights")
    
    with st.spinner("Generating insights..."):
        insights = analyzer.generate_insights(metrics_summary)
    
    # Display insights in a nice format
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
            <strong>Insight {i}:</strong> {insight}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics Overview
    st.markdown("## 📊 Key Metrics Overview")
    
    # Get latest values
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    # Display metrics by category
    for category, metrics in categories.items():
        if metrics:
            st.markdown(f"### {category}")
            
            cols = st.columns(min(len(metrics), 4))
            
            for i, metric in enumerate(metrics):
                if metric in metrics_summary:
                    data = metrics_summary[metric]
                    with cols[i % len(cols)]:
                        create_metric_card(
                            metric,
                            data['current_value'],
                            data['yoy_change'],
                            "YoY"
                        )
                        
                        # Add sparkline
                        fig = create_trend_sparkline(df, metric)
                        st.plotly_chart(fig, use_container_width=True, key=f"spark_{metric}")
    
    st.markdown("---")
    
    # Detailed Analysis Section
    st.markdown("## 🔍 Detailed Analysis")
    
    # Metric selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric for detailed view:",
            options=sorted(df['Metric'].unique())
        )
    
    with col2:
        view_type = st.radio(
            "View:",
            ["Year over Year", "Trend Analysis"],
            horizontal=True
        )
    
    # Display detailed chart
    if view_type == "Year over Year":
        fig = create_comparison_chart(df, selected_metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Trend analysis with MoM changes
        metric_data = df[df['Metric'] == selected_metric].sort_values('Date')
        
        fig = go.Figure()
        
        # Value line
        fig.add_trace(go.Scatter(
            x=metric_data['Date'],
            y=metric_data['Value'],
            name='Value',
            yaxis='y',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # MoM change bars
        fig.add_trace(go.Bar(
            x=metric_data['Date'],
            y=metric_data['MoM_Change'],
            name='MoM Change %',
            yaxis='y2',
            marker_color=np.where(metric_data['MoM_Change'] > 0, '#ef4444', '#10b981'),
            opacity=0.7
        ))
        
        is_percent = any(term in selected_metric.lower() for term in ['rate', 'percent', '%'])
        is_currency = any(term in selected_metric.lower() for term in ['cost', '$'])
        y1_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
        
        fig.update_layout(
            title=f"{selected_metric} - Trend Analysis with Month-over-Month Changes",
            xaxis_title="Date",
            yaxis=dict(title="Value", side="left", tickformat=y1_format),
            yaxis2=dict(title="MoM Change %", side="right", tickformat='.1%', overlaying='y'),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        metric_display = df[df['Metric'] == selected_metric][
            ['Date', 'Value', 'MoM_Change', 'YoY_Change']
        ].sort_values('Date', ascending=False)
        
        # Format the display
        metric_display['MoM_Change'] = metric_display['MoM_Change'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        metric_display['YoY_Change'] = metric_display['YoY_Change'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        st.dataframe(metric_display, use_container_width=True)
    
    # Performance Summary
    st.markdown("---")
    st.markdown("## 🎯 Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate overall metrics
    return_rate = df[(df['Metric'] == 'Overall Return Rate') & (df['Date'] == latest_date)]
    inspection_rate = df[(df['Metric'] == 'Percent Order Inspected') & (df['Date'] == latest_date)]
    avg_cost = df[(df['Metric'] == 'Average Cost per Inspection') & (df['Date'] == latest_date)]
    
    with col1:
        if not return_rate.empty:
            val = return_rate.iloc[0]['Value']
            yoy = return_rate.iloc[0]['YoY_Change']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Overall Quality Score**")
            quality_score = max(0, 100 - (val * 100))  # Convert return rate to quality score
            st.metric("Quality Score", f"{quality_score:.0f}/100", 
                     f"{-yoy*100:.1f} pts YoY" if pd.notna(yoy) else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if not inspection_rate.empty:
            val = inspection_rate.iloc[0]['Value']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Inspection Coverage**")
            st.metric("Coverage Rate", f"{val:.1%}", 
                     "✅ Above 85%" if val > 0.85 else "⚠️ Below target")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if not avg_cost.empty:
            val = avg_cost.iloc[0]['Value']
            yoy = avg_cost.iloc[0]['YoY_Change']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Cost Efficiency**")
            st.metric("Cost per Inspection", f"${val:.2f}", 
                     f"{yoy:.1%} YoY" if pd.notna(yoy) else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
