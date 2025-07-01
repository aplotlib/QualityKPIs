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
    .ai-insight-box {
        background: #e0e7ff;
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .synthesis-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1.2rem;
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
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.openai_client = None
        self.anthropic_client = None
        
        if openai_key and OPENAI_AVAILABLE:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        
        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
    
    def parse_quality_data(self, file_content: str) -> pd.DataFrame:
        """Parse CSV content and return structured DataFrame"""
        try:
            df = pd.read_csv(io.StringIO(file_content))
            
            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]
            
            return df
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            return None
    
    def get_openai_insights(self, metrics_summary: Dict) -> List[str]:
        """Get insights from OpenAI"""
        if not self.openai_client:
            return []
        
        prompt = f"""
        As a Quality Manager expert, analyze these quality metrics and provide 3-4 key insights:
        
        {json.dumps(metrics_summary, indent=2)}
        
        Focus on:
        1. Most concerning trends
        2. Improvements or deteriorations
        3. Specific metrics that need attention
        4. Actionable recommendations
        
        Be direct, specific, and data-driven. Format as a JSON object with key "insights" containing a list of strings.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("insights", [])
        except Exception as e:
            st.warning(f"OpenAI analysis failed: {str(e)}")
            return []
    
    def get_anthropic_insights(self, metrics_summary: Dict) -> List[str]:
        """Get insights from Anthropic Claude"""
        if not self.anthropic_client:
            return []
        
        prompt = f"""
        As a Quality Manager expert, analyze these quality metrics and provide 3-4 key insights:
        
        {json.dumps(metrics_summary, indent=2)}
        
        Focus on:
        1. Systemic quality issues
        2. Cost-benefit analysis of current inspection rates
        3. Seasonal patterns or anomalies
        4. Process improvement opportunities
        
        Be analytical and focus on root causes. Return a JSON object with key "insights" containing a list of strings.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            content = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("insights", [])
            return []
        except Exception as e:
            st.warning(f"Claude analysis failed: {str(e)}")
            return []
    
    def synthesize_insights(self, openai_insights: List[str], anthropic_insights: List[str], 
                          metrics_summary: Dict) -> Dict[str, List[str]]:
        """Use Claude to synthesize insights from both AI analyses"""
        if not self.anthropic_client:
            return {"synthesis": ["Unable to synthesize insights"], "actions": []}
        
        prompt = f"""
        You are a senior Quality Manager reviewing analyses from two AI systems about quality metrics.
        
        Current Metrics Summary:
        {json.dumps(metrics_summary, indent=2)}
        
        Analysis from AI System 1 (GPT-4):
        {json.dumps(openai_insights, indent=2)}
        
        Analysis from AI System 2 (Claude):
        {json.dumps(anthropic_insights, indent=2)}
        
        Please:
        1. Synthesize the key findings from both analyses into 2-3 comprehensive insights
        2. Identify any conflicting viewpoints and reconcile them
        3. Provide 2-3 specific action items based on the combined analysis
        
        Return a JSON object with two keys:
        - "synthesis": list of synthesized insights
        - "actions": list of specific action items
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",  # Using Sonnet for better synthesis
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.5
            )
            content = response.content[0].text
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            return {"synthesis": ["Unable to parse synthesis"], "actions": []}
        except Exception as e:
            return {"synthesis": [f"Synthesis failed: {str(e)}"], "actions": []}
    
    def categorize_metrics(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Categorize metrics into logical groups"""
        categories = {
            "Quality & Returns": [],
            "Operations": [],
            "Financial": [],
            "Volume & Throughput": []
        }
        
        for metric in metric_names:
            metric_lower = metric.lower()
            if any(term in metric_lower for term in ['return', 'defect', 'quality', 'rework']):
                categories["Quality & Returns"].append(metric)
            elif any(term in metric_lower for term in ['inspect', 'percent']):
                categories["Operations"].append(metric)
            elif any(term in metric_lower for term in ['cost', '$', 'price']):
                categories["Financial"].append(metric)
            elif any(term in metric_lower for term in ['order', 'total', 'volume']):
                categories["Volume & Throughput"].append(metric)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

# --- DATA PROCESSING ---
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with YoY and MoM calculations"""
    # Convert data types
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Handle Channel column if it exists
    if 'Channel' in df.columns:
        # For aggregated metrics, combine channels
        df_agg = df.groupby(['Metric', 'Year', 'Month'])['Value'].sum().reset_index()
        df_agg['Channel'] = 'Total'
        df = pd.concat([df, df_agg], ignore_index=True)
    
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
    
    # Add trend indicator (based on 3-month moving average)
    df['MA3'] = df.groupby('Metric')['Value'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # Determine trend based on metric type
    def calculate_trend(x, metric_name):
        if len(x) < 2:
            return 'Insufficient Data'
        
        # Determine if lower is better
        lower_is_better = any(term in metric_name.lower() for term in ['cost', 'return', 'defect', 'rework', 'rate'])
        
        # Compare recent vs older values
        if len(x) >= 3:
            recent = x.iloc[-1]
            older = x.iloc[-3]
        else:
            recent = x.iloc[-1]
            older = x.iloc[0]
        
        if lower_is_better:
            return 'Improving' if recent < older else 'Worsening'
        else:
            return 'Improving' if recent > older else 'Worsening'
    
    # Apply trend calculation
    for metric in df['Metric'].unique():
        mask = df['Metric'] == metric
        df.loc[mask, 'Trend'] = calculate_trend(df.loc[mask, 'MA3'], metric)
    
    return df

def calculate_summary_metrics(df: pd.DataFrame) -> Dict:
    """Calculate summary metrics for AI insights"""
    summary = {}
    
    latest_date = df['Date'].max()
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        latest = metric_data[metric_data['Date'] == latest_date]
        
        if not latest.empty:
            latest_row = latest.iloc[0]
            
            # Calculate 3-month average
            last_3_months = metric_data.nlargest(3, 'Date')['Value'].mean()
            
            # Calculate year-to-date average
            ytd = metric_data[metric_data['Year'] == latest_date.year]['Value'].mean()
            
            summary[metric] = {
                'current_value': float(latest_row['Value']),
                'yoy_change': float(latest_row['YoY_Change']) if pd.notna(latest_row['YoY_Change']) else None,
                'mom_change': float(latest_row['MoM_Change']) if pd.notna(latest_row['MoM_Change']) else None,
                'trend': latest_row['Trend'],
                '3_month_avg': float(last_3_months),
                'ytd_avg': float(ytd)
            }
    
    return summary

# --- VISUALIZATION FUNCTIONS ---
def create_trend_sparkline(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a small sparkline chart"""
    metric_data = df[df['Metric'] == metric].sort_values('Date').tail(12)  # Last 12 months
    
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
    for year in sorted(metric_data['Year'].unique(), reverse=True):
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
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_metric_card(metric: str, data: Dict) -> None:
    """Create a metric card with value and change"""
    current_value = data['current_value']
    change = data['yoy_change']
    
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
        icon = "➖"
    elif is_good:
        icon = "✅"
    else:
        icon = "⚠️"
    
    st.metric(
        label=metric,
        value=value_str,
        delta=f"{icon} {change_str} YoY",
        delta_color="off"
    )

# --- MAIN APP ---
def main():
    st.title("🧠 AI-Powered Quality Dashboard")
    st.markdown("Dual AI analysis with synthesized insights for comprehensive quality management")
    
    # Check for API keys
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if not openai_key and not anthropic_key:
        st.error("No AI API keys found in Streamlit secrets. Please add OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        st.stop()
    
    # Show available AI providers in sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Analysis Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if openai_key and OPENAI_AVAILABLE:
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        with col2:
            if anthropic_key and ANTHROPIC_AVAILABLE:
                st.success("✅ Claude")
            else:
                st.error("❌ Claude")
        
        st.markdown("---")
        st.info("This dashboard uses both AI systems to provide comprehensive insights, then synthesizes them for actionable recommendations.")
    
    # Initialize AI analyzer
    analyzer = AIQualityAnalyzer(openai_key, anthropic_key)
    
    # Load data
    try:
        with open('quality_data_clean.csv', 'r') as f:
            csv_content = f.read()
        
        with st.spinner("🤖 Loading and preparing quality data..."):
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
    st.markdown("## 🔍 Multi-AI Analysis & Synthesis")
    
    # Create columns for dual AI analysis
    col1, col2 = st.columns(2)
    
    with st.spinner("🤖 Running dual AI analysis..."):
        # Get insights from both AIs
        openai_insights = analyzer.get_openai_insights(metrics_summary)
        anthropic_insights = analyzer.get_anthropic_insights(metrics_summary)
    
    # Display individual AI insights
    with col1:
        st.markdown("#### 🟦 GPT-4 Analysis")
        if openai_insights:
            for insight in openai_insights:
                st.markdown(f"""
                <div class="ai-insight-box">
                    • {insight}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("GPT-4 analysis not available")
    
    with col2:
        st.markdown("#### 🟣 Claude Analysis")
        if anthropic_insights:
            for insight in anthropic_insights:
                st.markdown(f"""
                <div class="ai-insight-box">
                    • {insight}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Claude analysis not available")
    
    # Synthesized insights
    if openai_insights or anthropic_insights:
        st.markdown("### 🎯 Synthesized Insights & Action Items")
        
        with st.spinner("🧠 Synthesizing insights from both AI analyses..."):
            synthesis_result = analyzer.synthesize_insights(
                openai_insights, 
                anthropic_insights,
                metrics_summary
            )
        
        # Display synthesis
        if synthesis_result.get("synthesis"):
            for i, insight in enumerate(synthesis_result["synthesis"], 1):
                st.markdown(f"""
                <div class="synthesis-box">
                    <strong>Key Finding {i}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Display action items
        if synthesis_result.get("actions"):
            st.markdown("#### 📋 Recommended Actions")
            for i, action in enumerate(synthesis_result["actions"], 1):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Action {i}:</strong> {action}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics Overview
    st.markdown("## 📊 Key Metrics Overview")
    
    # Get latest values
    latest_date = df['Date'].max()
    st.caption(f"Latest data: {latest_date.strftime('%B %Y')}")
    
    # Display metrics by category
    for category, metrics in categories.items():
        if metrics:
            st.markdown(f"### {category}")
            
            cols = st.columns(min(len(metrics), 4))
            
            for i, metric in enumerate(metrics):
                if metric in metrics_summary:
                    with cols[i % len(cols)]:
                        create_metric_card(metric, metrics_summary[metric])
                        
                        # Add sparkline
                        fig = create_trend_sparkline(df, metric)
                        st.plotly_chart(fig, use_container_width=True, key=f"spark_{metric}")
    
    st.markdown("---")
    
    # Detailed Analysis Section
    st.markdown("## 📈 Detailed Analysis")
    
    # Metric selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric for detailed view:",
            options=sorted(df['Metric'].unique()),
            index=0
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
            marker_color=np.where(metric_data['MoM_Change'] > 0, '#10b981', '#ef4444'),
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
        metric_display['MoM_Change'] = metric_display['MoM_Change'].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        metric_display['YoY_Change'] = metric_display['YoY_Change'].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        
        st.dataframe(metric_display, use_container_width=True, hide_index=True)
    
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
            st.markdown("**🏆 Overall Quality Score**")
            quality_score = max(0, 100 - (val * 100))  # Convert return rate to quality score
            st.metric("Quality Score", f"{quality_score:.0f}/100", 
                     f"{-yoy*100:.1f} pts YoY" if pd.notna(yoy) else "N/A")
    
    with col2:
        if not inspection_rate.empty:
            val = inspection_rate.iloc[0]['Value']
            st.markdown("**🔍 Inspection Coverage**")
            st.metric("Coverage Rate", f"{val:.1%}", 
                     "✅ Above 85%" if val > 0.85 else "⚠️ Below target")
    
    with col3:
        if not avg_cost.empty:
            val = avg_cost.iloc[0]['Value']
            yoy = avg_cost.iloc[0]['YoY_Change']
            st.markdown("**💰 Cost Efficiency**")
            st.metric("Cost per Inspection", f"${val:.2f}", 
                     f"{yoy:.1%} YoY" if pd.notna(yoy) else "N/A")

if __name__ == "__main__":
    main()
