# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import io
import time

# --- Dependency Checks ---
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
    page_title="Quality Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    h1 { text-align: center; font-weight: 700; margin-bottom: 0.5rem; }
    h2 { font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; }
    h3 { font-weight: 600; margin-bottom: 0.5rem; }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-label"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-value"] {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-delta"] {
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Insight cards */
    .insight-card {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .action-card {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .warning-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Trend indicators */
    .trend-up { color: #10b981; font-size: 1.2rem; margin: 0; }
    .trend-down { color: #ef4444; font-size: 1.2rem; margin: 0; }
    .trend-neutral { color: #6b7280; font-size: 1.2rem; margin: 0; }
    
    /* Improve column spacing */
    .stColumn > div { padding: 0 0.5rem; }
    
    /* Summary cards */
    .summary-metric {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .summary-metric h4 {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .summary-metric .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- AI PROCESSOR ---
class AIQualityAnalyzer:
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.openai_client = None
        self.anthropic_client = None
        self.openai_available = False
        self.anthropic_available = False
        
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                self.openai_available = True
            except Exception as e:
                st.warning(f"OpenAI initialization failed: {str(e)}")
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                self.anthropic_available = True
            except Exception as e:
                st.warning(f"Anthropic initialization failed: {str(e)}")
    
    def parse_quality_data(self, file_content: str) -> pd.DataFrame:
        """Parse CSV content and return structured DataFrame"""
        try:
            df = pd.read_csv(io.StringIO(file_content))
            df.columns = [col.capitalize() for col in df.columns]
            return df
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            return None
    
    def generate_insights_with_ai(self, metrics_summary: Dict) -> Dict[str, List[str]]:
        """Generate insights using available AI"""
        insights = {
            "key_findings": [],
            "actions": [],
            "warnings": []
        }
        
        # Analyze metrics locally first
        local_insights = self._analyze_metrics_locally(metrics_summary)
        
        # Try AI enhancement if available
        if self.anthropic_available:
            ai_insights = self._get_claude_insights(metrics_summary)
            if ai_insights:
                insights = ai_insights
        elif self.openai_available:
            ai_insights = self._get_openai_insights(metrics_summary)
            if ai_insights:
                insights = ai_insights
        else:
            # Use local analysis
            insights = local_insights
        
        return insights
    
    def _analyze_metrics_locally(self, metrics_summary: Dict) -> Dict[str, List[str]]:
        """Analyze metrics without AI"""
        insights = {
            "key_findings": [],
            "actions": [],
            "warnings": []
        }
        
        # Analyze return rates
        if 'Overall Return Rate' in metrics_summary:
            return_data = metrics_summary['Overall Return Rate']
            current_rate = return_data['current_value']
            yoy_change = return_data.get('yoy_change', 0)
            
            if current_rate > 0.10:  # Above 10%
                insights["warnings"].append(f"Return rate is high at {current_rate:.1%} - investigate quality issues")
            elif yoy_change and yoy_change < -0.1:  # Improved by more than 10%
                insights["key_findings"].append(f"Return rate improved by {abs(yoy_change):.1%} YoY - quality initiatives working")
        
        # Analyze inspection coverage
        if 'Percent Order Inspected' in metrics_summary:
            inspection_data = metrics_summary['Percent Order Inspected']
            coverage = inspection_data['current_value']
            
            if coverage < 0.80:
                insights["actions"].append(f"Increase inspection coverage from {coverage:.1%} to at least 85%")
            
        # Analyze costs
        if 'Average Cost per Inspection' in metrics_summary:
            cost_data = metrics_summary['Average Cost per Inspection']
            current_cost = cost_data['current_value']
            yoy_change = cost_data.get('yoy_change', 0)
            
            if yoy_change and yoy_change > 0.15:  # Increased by more than 15%
                insights["warnings"].append(f"Inspection costs up {yoy_change:.1%} YoY - review efficiency")
        
        # Add general insights if none found
        if not any(insights.values()):
            insights["key_findings"].append("Quality metrics are stable - maintain current processes")
            insights["actions"].append("Continue monitoring for trend changes")
        
        return insights
    
    def _get_claude_insights(self, metrics_summary: Dict) -> Optional[Dict[str, List[str]]]:
        """Get insights from Claude"""
        if not self.anthropic_client:
            return None
        
        prompt = f"""
        Analyze these quality metrics as a quality manager and provide insights.
        
        Metrics: {json.dumps(metrics_summary, indent=2)}
        
        Provide:
        1. 2-3 key findings about quality performance
        2. 2-3 specific actions to improve
        3. 1-2 warnings about risks or concerning trends
        
        Format as JSON with keys: "key_findings", "actions", "warnings" (each containing a list of strings)
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            st.warning(f"Claude analysis error: {str(e)}")
            return None
    
    def _get_openai_insights(self, metrics_summary: Dict) -> Optional[Dict[str, List[str]]]:
        """Get insights from OpenAI"""
        if not self.openai_client:
            return None
        
        prompt = f"""
        Analyze these quality metrics and provide insights.
        
        Metrics: {json.dumps(metrics_summary, indent=2)}
        
        Provide:
        1. 2-3 key findings
        2. 2-3 actions to improve
        3. 1-2 warnings about risks
        
        Return JSON with keys: "key_findings", "actions", "warnings"
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.warning(f"OpenAI analysis error: {str(e)}")
            return None

# --- DATA PROCESSING ---
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with calculations"""
    # Convert data types
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Handle Channel column if exists
    if 'Channel' in df.columns:
        # Keep channel-specific data
        pass
    
    # Create date column
    df['Date'] = pd.to_datetime(
        df['Year'].astype(int).astype(str) + '-' + df['Month'], 
        format='%Y-%b', 
        errors='coerce'
    )
    
    # Remove invalid dates
    df = df[df['Date'].notna()]
    
    # Sort data
    df = df.sort_values(['Metric', 'Date'])
    
    # Calculate changes
    df['MoM_Change'] = df.groupby('Metric')['Value'].pct_change()
    df['YoY_Change'] = df.groupby('Metric')['Value'].pct_change(12)
    
    # Calculate 3-month moving average
    df['MA3'] = df.groupby('Metric')['Value'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    return df

def calculate_summary_metrics(df: pd.DataFrame) -> Dict:
    """Calculate summary metrics"""
    summary = {}
    latest_date = df['Date'].max()
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        
        # Get latest data point
        latest = metric_data[metric_data['Date'] == latest_date]
        
        if not latest.empty:
            latest_row = latest.iloc[0]
            
            # Get previous year same month for true YoY
            prev_year = metric_data[
                metric_data['Date'] == latest_date - pd.DateOffset(years=1)
            ]
            
            yoy_change = None
            if not prev_year.empty:
                yoy_change = (latest_row['Value'] - prev_year.iloc[0]['Value']) / prev_year.iloc[0]['Value']
            
            # Calculate trend
            recent_data = metric_data.nlargest(6, 'Date')
            if len(recent_data) >= 2:
                trend = 'up' if recent_data.iloc[0]['Value'] > recent_data.iloc[-1]['Value'] else 'down'
            else:
                trend = 'neutral'
            
            summary[metric] = {
                'current_value': float(latest_row['Value']),
                'yoy_change': float(yoy_change) if yoy_change is not None else None,
                'mom_change': float(latest_row['MoM_Change']) if pd.notna(latest_row['MoM_Change']) else None,
                'trend': trend,
                '3_month_avg': float(metric_data.nlargest(3, 'Date')['Value'].mean())
            }
    
    return summary

# --- VISUALIZATION FUNCTIONS ---
def create_metric_dashboard(df: pd.DataFrame, metrics_summary: Dict) -> None:
    """Create a comprehensive metrics dashboard"""
    
    # Group metrics by type
    categories = {
        "📦 Quality Metrics": {
            'metrics': ['Overall Return Rate'],
            'description': 'Product quality and customer satisfaction'
        },
        "🔍 Operations": {
            'metrics': ['Percent Order Inspected', 'Orders Inspected', 'Total Orders'],
            'description': 'Operational efficiency and throughput'
        },
        "💰 Financial": {
            'metrics': ['Total Cost of Inspection', 'Average Cost per Inspection'],
            'description': 'Cost management and efficiency'
        }
    }
    
    for category, info in categories.items():
        metric_list = info['metrics']
        available_metrics = [m for m in metric_list if m in metrics_summary]
        
        if available_metrics:
            # Section header
            st.markdown(f"""
                <div class="section-header">
                    <h3 style="margin: 0;">{category}</h3>
                    <span style="color: #64748b; font-size: 0.9rem;">{info['description']}</span>
                </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(len(available_metrics))
            
            for idx, metric in enumerate(available_metrics):
                with cols[idx]:
                    data = metrics_summary[metric]
                    
                    # Format value based on type
                    is_percent = any(term in metric.lower() for term in ['rate', 'percent', '%'])
                    is_currency = any(term in metric.lower() for term in ['cost', '$'])
                    
                    if is_percent:
                        value_str = f"{data['current_value']:.1%}"
                    elif is_currency:
                        value_str = f"${data['current_value']:,.2f}"
                    else:
                        value_str = f"{data['current_value']:,.0f}"
                    
                    # Determine delta color
                    yoy = data.get('yoy_change')
                    lower_is_better = any(term in metric.lower() for term in ['cost', 'return', 'defect', 'rework', 'rate'])
                    
                    if yoy is not None:
                        is_good = (yoy < 0 and lower_is_better) or (yoy > 0 and not lower_is_better)
                        delta = f"{'↑' if yoy > 0 else '↓'} {abs(yoy):.1%} YoY"
                        delta_color = "normal" if is_good else "inverse"
                    else:
                        delta = "No YoY data"
                        delta_color = "off"
                    
                    # Create metric
                    st.metric(
                        label=metric,
                        value=value_str,
                        delta=delta,
                        delta_color=delta_color
                    )
                    
                    # Add mini chart
                    metric_data = df[df['Metric'] == metric].tail(12)
                    
                    if len(metric_data) > 1:
                        fig = go.Figure()
                        
                        # Determine if metric should decrease (lower is better)
                        lower_is_better = any(term in metric.lower() for term in ['cost', 'return', 'defect', 'rework', 'rate'])
                        
                        # Set color based on trend
                        if len(metric_data) >= 2:
                            if lower_is_better:
                                color = '#ef4444' if metric_data.iloc[-1]['Value'] > metric_data.iloc[0]['Value'] else '#10b981'
                            else:
                                color = '#10b981' if metric_data.iloc[-1]['Value'] > metric_data.iloc[0]['Value'] else '#ef4444'
                        else:
                            color = '#3b82f6'
                        
                        fig.add_trace(go.Scatter(
                            x=metric_data['Date'],
                            y=metric_data['Value'],
                            mode='lines',
                            line=dict(color=color, width=3),
                            fill='tozeroy',
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
                            showlegend=False
                        ))
                        
                        # Add min/max points
                        min_point = metric_data.loc[metric_data['Value'].idxmin()]
                        max_point = metric_data.loc[metric_data['Value'].idxmax()]
                        
                        fig.add_trace(go.Scatter(
                            x=[min_point['Date'], max_point['Date']],
                            y=[min_point['Value'], max_point['Value']],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=['#ef4444', '#10b981'] if lower_is_better else ['#10b981', '#ef4444']
                            ),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            height=120,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(visible=False, fixedrange=True),
                            yaxis=dict(visible=False, fixedrange=True),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            dragmode=False,
                            hovermode=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_main_chart(df: pd.DataFrame, metric: str, chart_type: str = "trend") -> go.Figure:
    """Create main analysis chart"""
    metric_data = df[df['Metric'] == metric].sort_values('Date')
    
    if chart_type == "comparison":
        # Year over year comparison
        fig = go.Figure()
        
        for year in sorted(metric_data['Year'].unique(), reverse=True):
            year_data = metric_data[metric_data['Year'] == year]
            
            fig.add_trace(go.Scatter(
                x=year_data['Month'],
                y=year_data['Value'],
                name=str(int(year)),
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Year-over-Year Comparison",
            xaxis_title="Month",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
    else:  # trend analysis
        fig = go.Figure()
        
        # Main value line
        fig.add_trace(go.Scatter(
            x=metric_data['Date'],
            y=metric_data['Value'],
            name='Actual',
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=metric_data['Date'],
            y=metric_data['MA3'],
            name='3-Month Avg',
            mode='lines',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        # Add annotations for significant changes
        significant_changes = []
        for idx, row in metric_data.iterrows():
            if pd.notna(row['MoM_Change']) and abs(row['MoM_Change']) > 0.15:
                significant_changes.append({
                    'x': row['Date'],
                    'y': row['Value'],
                    'change': row['MoM_Change']
                })
        
        # Add only the most significant changes to avoid clutter
        if significant_changes:
            significant_changes.sort(key=lambda x: abs(x['change']), reverse=True)
            for change in significant_changes[:3]:  # Show top 3 changes
                fig.add_annotation(
                    x=change['x'],
                    y=change['y'],
                    text=f"{change['change']:+.0%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#6b7280",
                    bgcolor="white",
                    bordercolor="#6b7280",
                    borderwidth=1
                )
        
        fig.update_layout(
            title="Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
    
    # Format y-axis based on metric type
    is_percent = any(term in metric.lower() for term in ['rate', 'percent', '%'])
    is_currency = any(term in metric.lower() for term in ['cost', '$'])
    
    yaxis_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
    
    fig.update_layout(
        yaxis=dict(tickformat=yaxis_format),
        template='plotly_white',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# --- MAIN APP ---
def main():
    # Custom header
    st.markdown("""
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 3rem;
                   text-align: center;
                   margin-bottom: 0;">
            Quality Intelligence Dashboard
        </h1>
        <p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-top: 0;">
            Comprehensive quality metrics analysis with AI-powered insights
        </p>
    """, unsafe_allow_html=True)
    
    # Check for API keys
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <h3 style="margin-bottom: 1rem;">🤖 AI Analysis</h3>
        """, unsafe_allow_html=True)
        
        # API Status
        col1, col2 = st.columns(2)
        with col1:
            if openai_key:
                st.success("OpenAI ✓")
            else:
                st.info("OpenAI ✗")
                
        with col2:
            if anthropic_key:
                st.success("Claude ✓")
            else:
                st.info("Claude ✗")
        
        if not openai_key and not anthropic_key:
            st.warning("Local mode active")
        
        st.markdown("---")
        
        # Dashboard info
        st.markdown("""
            <h4 style="margin-bottom: 0.5rem;">📊 Metrics Tracked</h4>
            <ul style="font-size: 0.9rem; color: #64748b;">
                <li>Return rates & quality</li>
                <li>Inspection coverage</li>
                <li>Operational efficiency</li>
                <li>Cost management</li>
            </ul>
            
            <h4 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">🎯 Key Features</h4>
            <ul style="font-size: 0.9rem; color: #64748b;">
                <li>AI-powered insights</li>
                <li>Year-over-year analysis</li>
                <li>Trend detection</li>
                <li>Executive summary</li>
            </ul>
        """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AIQualityAnalyzer(openai_key, anthropic_key)
    
    # Load data
    progress_placeholder = st.empty()
    
    try:
        # Initial load
        with open('quality_data_clean.csv', 'r') as f:
            csv_content = f.read()
        
        # Show progress
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Parsing data structure...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            df = analyzer.parse_quality_data(csv_content)
            if df is None:
                st.stop()
            
            status_text.text("📈 Calculating metrics...")
            progress_bar.progress(50)
            time.sleep(0.3)
            
            # Prepare data
            df = prepare_data(df)
            
            status_text.text("🧮 Computing summaries...")
            progress_bar.progress(75)
            time.sleep(0.3)
            
            metrics_summary = calculate_summary_metrics(df)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
        
        # Clear progress
        progress_placeholder.empty()
        
    except FileNotFoundError:
        st.error("❌ quality_data_clean.csv not found!")
        st.info("Please ensure the CSV file is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Display insights
    st.markdown("## 💡 Intelligent Insights")
    
    with st.spinner("🤖 Analyzing quality metrics..."):
        insights = analyzer.generate_insights_with_ai(metrics_summary)
    
    # Create a container for insights
    insights_container = st.container()
    
    with insights_container:
        # Display insights in an elegant grid
        col1, col2, col3 = st.columns([1.1, 1.1, 0.8])
        
        with col1:
            st.markdown("""
                <h4 style="color: #3b82f6; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🎯</span> Key Findings
                </h4>
            """, unsafe_allow_html=True)
            
            for finding in insights.get("key_findings", ["Analyzing data..."]):
                st.markdown(f"""
                <div class="insight-card">
                    {finding}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <h4 style="color: #10b981; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">✅</span> Recommended Actions
                </h4>
            """, unsafe_allow_html=True)
            
            for action in insights.get("actions", ["Processing recommendations..."]):
                st.markdown(f"""
                <div class="action-card">
                    {action}
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <h4 style="color: #f59e0b; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">⚠️</span> Alerts
                </h4>
            """, unsafe_allow_html=True)
            
            warnings = insights.get("warnings", [])
            if warnings:
                for warning in warnings:
                    st.markdown(f"""
                    <div class="warning-card">
                        {warning}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color: #64748b; font-size: 0.9rem; padding: 1rem;">
                    No critical issues detected
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Dashboard
    st.markdown("## 📊 Metrics Overview")
    latest_date = df['Date'].max()
    st.caption(f"Latest data: {latest_date.strftime('%B %Y')}")
    
    # Update sidebar with data freshness
    with st.sidebar:
        st.markdown(f"""
            <div style="margin-top: 2rem; padding: 1rem; background: #f1f5f9; border-radius: 8px;">
                <small style="color: #64748b;">Data freshness:</small><br>
                <strong>{latest_date.strftime('%B %Y')}</strong>
            </div>
        """, unsafe_allow_html=True)
    
    create_metric_dashboard(df, metrics_summary)
    
    st.markdown("---")
    
    # Detailed Analysis
    st.markdown("## 🔍 Detailed Analysis")
    
    # Create a more elegant selector
    analysis_container = st.container()
    
    with analysis_container:
        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        
        with col1:
            selected_metric = st.selectbox(
                "Select metric to analyze:",
                options=sorted(df['Metric'].unique()),
                index=0,
                help="Choose a metric to see detailed trends and analysis"
            )
        
        with col2:
            chart_type = st.radio(
                "Visualization:",
                ["📈 Trend", "📊 YoY Compare"],
                horizontal=True,
                help="Toggle between trend analysis and year-over-year comparison"
            )
        
        with col3:
            # Show current status
            if selected_metric in metrics_summary:
                data = metrics_summary[selected_metric]
                trend = data.get('trend', 'neutral')
                
                # Determine if metric is good or bad
                lower_is_better = any(term in selected_metric.lower() for term in ['cost', 'return', 'defect', 'rework', 'rate'])
                current_val = data.get('current_value', 0)
                
                if trend == 'up':
                    if lower_is_better:
                        st.markdown('<div style="text-align: center; padding: 1rem;"><h3 class="trend-down">↑ Worsening</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align: center; padding: 1rem;"><h3 class="trend-up">↑ Improving</h3></div>', unsafe_allow_html=True)
                elif trend == 'down':
                    if lower_is_better:
                        st.markdown('<div style="text-align: center; padding: 1rem;"><h3 class="trend-up">↓ Improving</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align: center; padding: 1rem;"><h3 class="trend-down">↓ Worsening</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align: center; padding: 1rem;"><h3 class="trend-neutral">→ Stable</h3></div>', unsafe_allow_html=True)
    
    # Display chart
    fig = create_main_chart(
        df, 
        selected_metric, 
        "comparison" if "YoY" in chart_type else "trend"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Detailed Data", expanded=False):
        st.markdown(f"**{selected_metric} - Last 12 Months**")
        
        display_df = df[df['Metric'] == selected_metric][
            ['Date', 'Value', 'MoM_Change', 'YoY_Change']
        ].sort_values('Date', ascending=False).head(12)
        
        # Create a copy for display
        display_df = display_df.copy()
        
        # Format date
        display_df['Date'] = display_df['Date'].dt.strftime('%b %Y')
        
        # Format values based on metric type
        is_percent = any(term in selected_metric.lower() for term in ['rate', 'percent', '%'])
        is_currency = any(term in selected_metric.lower() for term in ['cost', '$'])
        
        if is_percent:
            display_df['Value'] = display_df['Value'].map(lambda x: f"{x:.1%}")
        elif is_currency:
            display_df['Value'] = display_df['Value'].map(lambda x: f"${x:,.2f}")
        else:
            display_df['Value'] = display_df['Value'].map(lambda x: f"{x:,.0f}")
        
        # Format changes with % symbol
        display_df['MoM_Change'] = display_df['MoM_Change'].map(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "-"
        )
        display_df['YoY_Change'] = display_df['YoY_Change'].map(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "-"
        )
        
        # Rename columns for clarity
        display_df = display_df.rename(columns={
            'Date': 'Month',
            'MoM_Change': 'Month-over-Month',
            'YoY_Change': 'Year-over-Year'
        })
        
        # Display with custom styling
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Footer summary
    st.markdown("---")
    st.markdown("""
        <h3 style="text-align: center; margin-bottom: 2rem;">
            🎯 Executive Summary
        </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    return_rate = metrics_summary.get('Overall Return Rate', {}).get('current_value', 0)
    quality_score = max(0, 100 - (return_rate * 100)) if return_rate else 0
    
    inspection_rate = metrics_summary.get('Percent Order Inspected', {}).get('current_value', 0)
    
    total_orders = metrics_summary.get('Total Orders', {}).get('current_value', 0)
    
    avg_cost = metrics_summary.get('Average Cost per Inspection', {}).get('current_value', 0)
    
    with col1:
        st.markdown("""
        <div class="summary-metric">
            <h4>Quality Score</h4>
            <div class="value" style="color: #10b981;">""" + f"{quality_score:.0f}/100" + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#10b981" if inspection_rate > 0.85 else "#f59e0b"
        st.markdown(f"""
        <div class="summary-metric">
            <h4>Inspection Coverage</h4>
            <div class="value" style="color: {color};">""" + f"{inspection_rate:.0%}" + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="summary-metric">
            <h4>Monthly Orders</h4>
            <div class="value">""" + f"{total_orders:,.0f}" + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="summary-metric">
            <h4>Cost per Inspection</h4>
            <div class="value">""" + f"${avg_cost:.2f}" + """</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
