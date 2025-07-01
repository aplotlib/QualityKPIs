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
    .main .block-container { padding: 2rem; }
    h1 { text-align: center; font-weight: 700; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        height: 100%;
    }
    .insight-card {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .action-card {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }
    .trend-neutral { color: #6b7280; }
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
                model="claude-3-haiku-20240307",  # Using Haiku which is reliable
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
                model="gpt-3.5-turbo",  # Using 3.5 for reliability
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
        "📦 Quality Metrics": ['Overall Return Rate'],
        "🔍 Operations": ['Percent Order Inspected', 'Orders Inspected', 'Total Orders'],
        "💰 Financial": ['Total Cost of Inspection', 'Average Cost per Inspection']
    }
    
    for category, metric_list in categories.items():
        st.markdown(f"### {category}")
        
        cols = st.columns(len([m for m in metric_list if m in metrics_summary]))
        col_idx = 0
        
        for metric in metric_list:
            if metric in metrics_summary:
                with cols[col_idx]:
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
                    
                    # Create metric
                    yoy = data.get('yoy_change')
                    if yoy is not None:
                        delta = f"{yoy:+.1%} YoY"
                    else:
                        delta = "No YoY data"
                    
                    st.metric(
                        label=metric,
                        value=value_str,
                        delta=delta
                    )
                    
                    # Add mini chart
                    metric_data = df[df['Metric'] == metric].tail(12)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metric_data['Date'],
                        y=metric_data['Value'],
                        mode='lines',
                        line=dict(color='#3b82f6', width=2),
                        showlegend=False
                    ))
                    fig.update_layout(
                        height=100,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                col_idx += 1

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
        for idx, row in metric_data.iterrows():
            if pd.notna(row['MoM_Change']) and abs(row['MoM_Change']) > 0.15:
                fig.add_annotation(
                    x=row['Date'],
                    y=row['Value'],
                    text=f"{row['MoM_Change']:+.0%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#6b7280"
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
    
    if is_percent:
        fig.update_yaxis(tickformat='.1%')
    elif is_currency:
        fig.update_yaxis(tickformat='$,.0f')
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

# --- MAIN APP ---
def main():
    st.title("📊 Quality Intelligence Dashboard")
    st.markdown("Comprehensive quality metrics analysis with AI-powered insights")
    
    # Check for API keys
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Status")
        
        if openai_key:
            st.success("✅ OpenAI Connected")
        else:
            st.info("ℹ️ OpenAI Not Configured")
            
        if anthropic_key:
            st.success("✅ Claude Connected")
        else:
            st.info("ℹ️ Claude Not Configured")
        
        if not openai_key and not anthropic_key:
            st.warning("Running in local analysis mode")
        
        st.markdown("---")
        st.markdown("### 📈 Dashboard Info")
        st.info(
            "This dashboard analyzes quality metrics including:\n"
            "• Return rates\n"
            "• Inspection coverage\n"
            "• Cost efficiency\n"
            "• Order volumes"
        )
    
    # Initialize analyzer
    analyzer = AIQualityAnalyzer(openai_key, anthropic_key)
    
    # Load data
    try:
        with open('quality_data_clean.csv', 'r') as f:
            csv_content = f.read()
        
        df = analyzer.parse_quality_data(csv_content)
        if df is None:
            st.stop()
        
        # Prepare data
        df = prepare_data(df)
        metrics_summary = calculate_summary_metrics(df)
        
    except FileNotFoundError:
        st.error("❌ quality_data_clean.csv not found!")
        st.info("Please ensure the CSV file is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Display insights
    st.markdown("## 💡 Intelligent Insights")
    
    with st.spinner("Analyzing quality metrics..."):
        insights = analyzer.generate_insights_with_ai(metrics_summary)
    
    # Display insights in cards
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### 🎯 Key Findings")
        for finding in insights.get("key_findings", ["No findings available"]):
            st.markdown(f"""
            <div class="insight-card">
                {finding}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ✅ Recommended Actions")
        for action in insights.get("actions", ["No actions available"]):
            st.markdown(f"""
            <div class="action-card">
                {action}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### ⚠️ Warnings")
        for warning in insights.get("warnings", ["No warnings"]):
            st.markdown(f"""
            <div class="warning-card">
                {warning}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Dashboard
    st.markdown("## 📊 Metrics Overview")
    latest_date = df['Date'].max()
    st.caption(f"Latest data: {latest_date.strftime('%B %Y')}")
    
    create_metric_dashboard(df, metrics_summary)
    
    st.markdown("---")
    
    # Detailed Analysis
    st.markdown("## 🔍 Detailed Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric to analyze:",
            options=sorted(df['Metric'].unique()),
            index=0
        )
    
    with col2:
        chart_type = st.radio(
            "Chart type:",
            ["Trend", "YoY Comparison"],
            horizontal=True
        )
    
    with col3:
        # Show metric info
        if selected_metric in metrics_summary:
            data = metrics_summary[selected_metric]
            trend = data.get('trend', 'neutral')
            if trend == 'up':
                st.markdown('<h3 class="trend-up">↑ Trending Up</h3>', unsafe_allow_html=True)
            elif trend == 'down':
                st.markdown('<h3 class="trend-down">↓ Trending Down</h3>', unsafe_allow_html=True)
            else:
                st.markdown('<h3 class="trend-neutral">→ Stable</h3>', unsafe_allow_html=True)
    
    # Display chart
    fig = create_main_chart(
        df, 
        selected_metric, 
        "comparison" if chart_type == "YoY Comparison" else "trend"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Data Table"):
        display_df = df[df['Metric'] == selected_metric][
            ['Date', 'Value', 'MoM_Change', 'YoY_Change']
        ].sort_values('Date', ascending=False).head(12)
        
        # Format percentages
        display_df['MoM_Change'] = display_df['MoM_Change'].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
        display_df['YoY_Change'] = display_df['YoY_Change'].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Footer summary
    st.markdown("---")
    st.markdown("### 🎯 Quick Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    return_rate = metrics_summary.get('Overall Return Rate', {}).get('current_value', 0)
    quality_score = max(0, 100 - (return_rate * 100)) if return_rate else 0
    
    inspection_rate = metrics_summary.get('Percent Order Inspected', {}).get('current_value', 0)
    
    total_orders = metrics_summary.get('Total Orders', {}).get('current_value', 0)
    
    avg_cost = metrics_summary.get('Average Cost per Inspection', {}).get('current_value', 0)
    
    with col1:
        st.metric("Quality Score", f"{quality_score:.0f}/100")
    
    with col2:
        st.metric("Inspection Coverage", f"{inspection_rate:.0%}")
    
    with col3:
        st.metric("Monthly Orders", f"{total_orders:,.0f}")
    
    with col4:
        st.metric("Avg Inspection Cost", f"${avg_cost:.2f}")

if __name__ == "__main__":
    main()
