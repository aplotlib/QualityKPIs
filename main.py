# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import io
import time
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quality Executive Dashboard",
    page_icon="🎯",
    layout="wide",
)

# --- EXECUTIVE DASHBOARD STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Reset and global styles */
    html, body, [class*="st-"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .main .block-container { 
        padding: 1rem 2rem;
        max-width: 1800px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header */
    .dashboard-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    /* Critical metric card - HUGE and BOLD */
    .critical-metric-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        position: relative;
        overflow: hidden;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .critical-metric-card.red {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 3px solid #dc2626;
    }
    
    .critical-metric-card.yellow {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 3px solid #f59e0b;
    }
    
    .critical-metric-card.green {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 3px solid #10b981;
    }
    
    .critical-metric-value {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
    }
    
    .critical-metric-label {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .critical-metric-change {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    
    .metric-arrow {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    
    /* Status banner - IMPOSSIBLE TO MISS */
    .status-banner {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        animation: pulse 2s infinite;
    }
    
    .status-banner.critical {
        background: #dc2626;
        color: white;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.3);
    }
    
    .status-banner.warning {
        background: #f59e0b;
        color: white;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
    }
    
    .status-banner.good {
        background: #10b981;
        color: white;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Action cards - CLEAR AND ACTIONABLE */
    .action-section {
        background: #f8fafc;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .action-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .action-card {
        background: white;
        border-left: 5px solid;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .action-card:hover {
        transform: translateX(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .action-card.urgent {
        border-color: #dc2626;
    }
    
    .action-card.important {
        border-color: #f59e0b;
    }
    
    .action-card.routine {
        border-color: #3b82f6;
    }
    
    .action-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .action-description {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Trend indicator - BIG AND CLEAR */
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.125rem;
        gap: 0.5rem;
    }
    
    .trend-indicator.improving {
        background: #d1fae5;
        color: #064e3b;
    }
    
    .trend-indicator.worsening {
        background: #fee2e2;
        color: #7f1d1d;
    }
    
    .trend-indicator.stable {
        background: #e0e7ff;
        color: #312e81;
    }
    
    /* Big number displays */
    .big-number {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
    }
    
    .big-number-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    /* Simple progress bars */
    .progress-container {
        margin: 1.5rem 0;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .progress-bar {
        height: 20px;
        background: #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        transition: width 1s ease;
    }
    
    .progress-fill.good {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    }
    
    .progress-fill.warning {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    }
    
    .progress-fill.critical {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Executive summary box */
    .executive-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .executive-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    }
    
    .executive-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .executive-content {
        font-size: 1.125rem;
        line-height: 1.8;
        position: relative;
        z-index: 1;
    }
    
    /* Quality score display - MASSIVE */
    .quality-score-display {
        text-align: center;
        padding: 3rem;
        background: white;
        border-radius: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .quality-score-number {
        font-size: 8rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .quality-score-label {
        font-size: 1.5rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 1rem;
    }
    
    /* Alert box */
    .alert-box {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-weight: 600;
        font-size: 1.125rem;
    }
    
    .alert-box.red {
        background: #fee2e2;
        border: 2px solid #dc2626;
        color: #7f1d1d;
    }
    
    .alert-box.yellow {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        color: #78350f;
    }
    
    .alert-box.green {
        background: #d1fae5;
        border: 2px solid #10b981;
        color: #064e3b;
    }
    
    .alert-icon {
        font-size: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .critical-metric-value { font-size: 3rem; }
        .quality-score-number { font-size: 5rem; }
        .big-number { font-size: 2.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING (Simplified) ---
def load_and_process_data():
    """Load and process quality data"""
    try:
        df = pd.read_csv('quality_data_clean.csv')
        
        # Clean column names
        df.columns = [col.strip().title() for col in df.columns]
        
        # Process data
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
        
        # Sort by date
        df = df.sort_values(['Metric', 'Date'])
        
        # Calculate changes
        df['MoM_Change'] = df.groupby('Metric')['Value'].pct_change()
        df['YoY_Change'] = df.groupby('Metric')['Value'].pct_change(12)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_current_metrics(df):
    """Get current values and changes for all metrics"""
    latest_date = df['Date'].max()
    metrics = {}
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        current = metric_data[metric_data['Date'] == latest_date]
        
        if not current.empty:
            current_val = current.iloc[0]['Value']
            
            # Get YoY value
            prev_year = metric_data[metric_data['Date'] == latest_date - pd.DateOffset(years=1)]
            yoy_change = None
            if not prev_year.empty:
                yoy_change = (current_val - prev_year.iloc[0]['Value']) / prev_year.iloc[0]['Value']
            
            # Get trend (last 6 months)
            recent = metric_data.nlargest(6, 'Date')
            if len(recent) >= 2:
                first_avg = recent.iloc[:3]['Value'].mean()
                last_avg = recent.iloc[-3:]['Value'].mean()
                
                if abs(last_avg - first_avg) < (first_avg * 0.02):  # Less than 2% change
                    trend = 'stable'
                elif last_avg > first_avg:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'stable'
            
            metrics[metric] = {
                'value': current_val,
                'yoy_change': yoy_change,
                'trend': trend,
                'history': metric_data[['Date', 'Value']].tail(12)
            }
    
    return metrics

def calculate_quality_score(metrics):
    """Calculate overall quality score (0-100)"""
    score = 100
    
    # Deduct points for high return rate
    if 'Overall Return Rate' in metrics:
        return_rate = metrics['Overall Return Rate']['value']
        if return_rate > 0.10:
            score -= 40  # Critical
        elif return_rate > 0.08:
            score -= 25  # Warning
        elif return_rate > 0.05:
            score -= 10  # Acceptable
        # Bonus for excellent return rate
        elif return_rate < 0.03:
            score += 5
    
    # Deduct points for low inspection coverage
    if 'Percent Order Inspected' in metrics:
        inspection_rate = metrics['Percent Order Inspected']['value']
        if inspection_rate < 0.80:
            score -= 30  # Critical
        elif inspection_rate < 0.85:
            score -= 15  # Warning
        # Bonus for excellent coverage
        elif inspection_rate > 0.95:
            score += 5
    
    # Deduct for increasing costs
    if 'Average Cost per Inspection' in metrics:
        cost_trend = metrics['Average Cost per Inspection']['trend']
        if cost_trend == 'increasing':
            score -= 5
    
    return max(0, min(100, score))

def get_status_and_actions(metrics):
    """Determine overall status and required actions"""
    critical_issues = []
    warnings = []
    actions = []
    
    # Check return rate
    if 'Overall Return Rate' in metrics:
        return_rate = metrics['Overall Return Rate']['value']
        return_trend = metrics['Overall Return Rate']['trend']
        
        if return_rate > 0.10:
            critical_issues.append("Return rate exceeds 10% threshold")
            actions.append(("URGENT: Launch immediate root cause analysis", "urgent"))
            actions.append(("URGENT: 100% inspection for next 3 batches", "urgent"))
        elif return_rate > 0.08:
            warnings.append("Return rate approaching critical level")
            actions.append(("Increase QC sampling to 50%", "important"))
            actions.append(("Review top 3 return reasons", "important"))
        
        if return_trend == 'increasing':
            warnings.append("Return rate trending upward")
            actions.append(("Analyze monthly return data patterns", "routine"))
    
    # Check inspection coverage
    if 'Percent Order Inspected' in metrics:
        inspection_rate = metrics['Percent Order Inspected']['value']
        
        if inspection_rate < 0.80:
            critical_issues.append("Inspection coverage below 80% minimum")
            actions.append(("URGENT: Hire additional QC staff", "urgent"))
        elif inspection_rate < 0.85:
            warnings.append("Inspection coverage below target")
            actions.append(("Optimize inspection workflow", "important"))
    
    # Determine overall status
    if critical_issues:
        status = "CRITICAL - IMMEDIATE ACTION REQUIRED"
        status_type = "critical"
    elif warnings:
        status = "WARNING - ATTENTION NEEDED"
        status_type = "warning"
    else:
        status = "GOOD - ALL SYSTEMS NORMAL"
        status_type = "good"
    
    return status, status_type, critical_issues, warnings, actions

# --- VISUALIZATION FUNCTIONS ---
def create_big_metric_chart(metric_data, metric_name, is_percent=False, is_currency=False):
    """Create a large, clear trend chart for a metric"""
    fig = go.Figure()
    
    # Determine if lower is better
    lower_is_better = any(term in metric_name.lower() for term in ['cost', 'return', 'defect', 'rate'])
    
    # Main line
    fig.add_trace(go.Scatter(
        x=metric_data['Date'],
        y=metric_data['Value'],
        mode='lines+markers',
        name=metric_name,
        line=dict(width=4, color='#3b82f6'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add threshold lines for key metrics
    if 'return' in metric_name.lower():
        fig.add_hline(y=0.10, line_dash="dash", line_color="red", line_width=3,
                     annotation_text="CRITICAL (10%)", annotation_position="right")
        fig.add_hline(y=0.08, line_dash="dash", line_color="orange", line_width=3,
                     annotation_text="WARNING (8%)", annotation_position="right")
    elif 'inspect' in metric_name.lower() and 'percent' in metric_name.lower():
        fig.add_hline(y=0.80, line_dash="dash", line_color="red", line_width=3,
                     annotation_text="MINIMUM (80%)", annotation_position="right")
        fig.add_hline(y=0.95, line_dash="dash", line_color="green", line_width=3,
                     annotation_text="TARGET (95%)", annotation_position="right")
    
    # Format
    yaxis_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
    
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(tickformat=yaxis_format, showgrid=True, gridcolor='#f0f0f0'),
        hovermode='x',
        plot_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig

def create_mini_sparkline(history, color='#3b82f6'):
    """Create a simple sparkline"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history['Date'],
        y=history['Value'],
        mode='lines',
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
        showlegend=False
    ))
    
    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# --- MAIN APPLICATION ---
def main():
    # Header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">Quality Executive Dashboard</h1>
            <p class="dashboard-subtitle">Real-time quality metrics at a glance</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_and_process_data()
    if df is None:
        st.stop()
    
    # Get current metrics
    metrics = get_current_metrics(df)
    
    # Calculate quality score and status
    quality_score = calculate_quality_score(metrics)
    status, status_type, critical_issues, warnings, actions = get_status_and_actions(metrics)
    
    # Display date
    latest_date = df['Date'].max()
    st.caption(f"📅 Data as of {latest_date.strftime('%B %Y')}")
    
    # STATUS BANNER - Can't miss this!
    st.markdown(f"""
        <div class="status-banner {status_type}">
            {status}
        </div>
    """, unsafe_allow_html=True)
    
    # Top row - CRITICAL METRICS (Big and Bold)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # QUALITY SCORE
    with col1:
        score_color = "#10b981" if quality_score >= 80 else "#f59e0b" if quality_score >= 60 else "#dc2626"
        score_status = "green" if quality_score >= 80 else "yellow" if quality_score >= 60 else "red"
        
        st.markdown(f"""
            <div class="critical-metric-card {score_status}">
                <div class="critical-metric-value" style="color: {score_color};">
                    {quality_score}
                </div>
                <div class="critical-metric-label">Quality Score</div>
                <div class="critical-metric-change">
                    Target: 85+
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # RETURN RATE
    with col2:
        if 'Overall Return Rate' in metrics:
            return_data = metrics['Overall Return Rate']
            return_rate = return_data['value']
            yoy_change = return_data['yoy_change']
            
            # Determine status
            if return_rate > 0.10:
                card_status = "red"
                value_color = "#dc2626"
            elif return_rate > 0.08:
                card_status = "yellow"
                value_color = "#f59e0b"
            else:
                card_status = "green"
                value_color = "#10b981"
            
            # Format change
            if yoy_change is not None:
                if yoy_change > 0:
                    change_str = f'<span class="metric-arrow" style="color: #dc2626;">↑</span>{yoy_change:.1%} vs last year'
                else:
                    change_str = f'<span class="metric-arrow" style="color: #10b981;">↓</span>{abs(yoy_change):.1%} vs last year'
            else:
                change_str = "No YoY data"
            
            st.markdown(f"""
                <div class="critical-metric-card {card_status}">
                    <div class="critical-metric-value" style="color: {value_color};">
                        {return_rate:.1%}
                    </div>
                    <div class="critical-metric-label">Return Rate</div>
                    <div class="critical-metric-change">
                        {change_str}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # INSPECTION COVERAGE
    with col3:
        if 'Percent Order Inspected' in metrics:
            inspection_data = metrics['Percent Order Inspected']
            inspection_rate = inspection_data['value']
            
            # Determine status
            if inspection_rate < 0.80:
                card_status = "red"
                value_color = "#dc2626"
            elif inspection_rate < 0.85:
                card_status = "yellow"
                value_color = "#f59e0b"
            else:
                card_status = "green"
                value_color = "#10b981"
            
            st.markdown(f"""
                <div class="critical-metric-card {card_status}">
                    <div class="critical-metric-value" style="color: {value_color};">
                        {inspection_rate:.0%}
                    </div>
                    <div class="critical-metric-label">Inspection Coverage</div>
                    <div class="critical-metric-change">
                        Minimum: 80%
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # ALERTS AND ACTIONS
    if critical_issues or warnings:
        st.markdown("---")
        st.markdown("## 🚨 Alerts & Required Actions")
        
        col1, col2 = st.columns([1, 2])
        
        # Alerts column
        with col1:
            if critical_issues:
                for issue in critical_issues:
                    st.markdown(f"""
                        <div class="alert-box red">
                            <span class="alert-icon">🚨</span>
                            <span>{issue}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            if warnings:
                for warning in warnings:
                    st.markdown(f"""
                        <div class="alert-box yellow">
                            <span class="alert-icon">⚠️</span>
                            <span>{warning}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            if not critical_issues and not warnings:
                st.markdown(f"""
                    <div class="alert-box green">
                        <span class="alert-icon">✅</span>
                        <span>All metrics within acceptable ranges</span>
                    </div>
                """, unsafe_allow_html=True)
        
        # Actions column
        with col2:
            st.markdown('<div class="action-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="action-header">Required Actions</h3>', unsafe_allow_html=True)
            
            if actions:
                for action, priority in actions:
                    st.markdown(f"""
                        <div class="action-card {priority}">
                            <div class="action-title">{action}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="action-card routine">
                        <div class="action-title">Continue standard monitoring procedures</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # KEY METRICS TRENDS
    st.markdown("---")
    st.markdown("## 📊 Key Metric Trends")
    
    # Return Rate Trend
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if 'Overall Return Rate' in metrics:
            st.markdown("### Return Rate Trend")
            fig = create_big_metric_chart(
                metrics['Overall Return Rate']['history'],
                'Overall Return Rate',
                is_percent=True
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Overall Return Rate' in metrics:
            trend = metrics['Overall Return Rate']['trend']
            if trend == 'increasing':
                trend_class = 'worsening'
                trend_text = '📈 INCREASING'
            elif trend == 'decreasing':
                trend_class = 'improving'
                trend_text = '📉 DECREASING'
            else:
                trend_class = 'stable'
                trend_text = '➡️ STABLE'
            
            st.markdown(f"""
                <div style="margin-top: 4rem; text-align: center;">
                    <div class="trend-indicator {trend_class}">
                        {trend_text}
                    </div>
                    <p style="margin-top: 1rem; color: #64748b;">6-month trend</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Secondary Metrics
    st.markdown("### Other Metrics")
    
    metric_cols = st.columns(3)
    
    # Define secondary metrics
    secondary_metrics = [
        ('Total Orders', False, False),
        ('Orders Inspected', False, False),
        ('Average Cost per Inspection', False, True)
    ]
    
    for idx, (metric_name, is_percent, is_currency) in enumerate(secondary_metrics):
        if metric_name in metrics:
            with metric_cols[idx % 3]:
                data = metrics[metric_name]
                
                # Format value
                if is_percent:
                    value_str = f"{data['value']:.1%}"
                elif is_currency:
                    value_str = f"${data['value']:,.2f}"
                else:
                    value_str = f"{data['value']:,.0f}"
                
                # Determine color based on trend
                lower_is_better = 'cost' in metric_name.lower()
                if data['trend'] == 'increasing':
                    trend_color = "#dc2626" if lower_is_better else "#10b981"
                elif data['trend'] == 'decreasing':
                    trend_color = "#10b981" if lower_is_better else "#dc2626"
                else:
                    trend_color = "#3b82f6"
                
                st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <h4 style="margin: 0; color: #64748b; font-size: 0.875rem;">{metric_name}</h4>
                        <p class="big-number" style="color: {trend_color}; font-size: 2.5rem; margin: 0.5rem 0;">
                            {value_str}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Mini sparkline
                fig = create_mini_sparkline(data['history'], trend_color)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # EXECUTIVE SUMMARY
    st.markdown("---")
    st.markdown("""
        <div class="executive-box">
            <h2 class="executive-title">📋 Executive Summary</h2>
            <div class="executive-content">
    """, unsafe_allow_html=True)
    
    # Generate summary text
    summary_points = []
    
    if quality_score >= 80:
        summary_points.append(f"• Quality score is GOOD at {quality_score}/100")
    else:
        summary_points.append(f"• Quality score needs improvement at {quality_score}/100")
    
    if 'Overall Return Rate' in metrics:
        return_rate = metrics['Overall Return Rate']['value']
        if return_rate > 0.10:
            summary_points.append(f"• Return rate is CRITICAL at {return_rate:.1%} - immediate intervention required")
        elif return_rate > 0.08:
            summary_points.append(f"• Return rate is elevated at {return_rate:.1%} - close monitoring needed")
        else:
            summary_points.append(f"• Return rate is acceptable at {return_rate:.1%}")
    
    if 'Percent Order Inspected' in metrics:
        inspection_rate = metrics['Percent Order Inspected']['value']
        if inspection_rate < 0.80:
            summary_points.append(f"• Inspection coverage is TOO LOW at {inspection_rate:.0%} - increase immediately")
        else:
            summary_points.append(f"• Inspection coverage is adequate at {inspection_rate:.0%}")
    
    # Add trend summary
    improving_metrics = sum(1 for m in metrics.values() if m['trend'] == 'decreasing' and any(term in str(m).lower() for term in ['cost', 'return']))
    worsening_metrics = sum(1 for m in metrics.values() if m['trend'] == 'increasing' and any(term in str(m).lower() for term in ['cost', 'return']))
    
    if worsening_metrics > improving_metrics:
        summary_points.append(f"• Overall trend is NEGATIVE with {worsening_metrics} metrics worsening")
    else:
        summary_points.append(f"• Overall trend is POSITIVE with {improving_metrics} metrics improving")
    
    for point in summary_points:
        st.markdown(f"<p style='margin: 0.5rem 0; font-size: 1.125rem;'>{point}</p>", unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer with data freshness
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; color: #94a3b8; font-size: 0.875rem;">
            Last updated: {latest_date.strftime('%B %d, %Y')} | 
            Next update: {(latest_date + pd.DateOffset(months=1)).strftime('%B %d, %Y')}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
