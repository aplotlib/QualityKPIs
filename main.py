# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION FOR TV DISPLAY ---
st.set_page_config(
    page_title="Quality TV Dashboard",
    page_icon="📺",
    layout="wide",
)

# --- TV DISPLAY OPTIMIZED STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    /* TV Display Optimization */
    html, body, [class*="st-"] { 
        font-family: 'Inter', sans-serif;
        background: #0f172a;
    }
    
    .stApp {
        background: #0f172a;
    }
    
    .main .block-container { 
        padding: 2rem;
        max-width: 100%;
        background: #0f172a;
    }
    
    /* Hide all Streamlit chrome for TV */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container with dark theme */
    .tv-container {
        background: #0f172a;
        min-height: 100vh;
        color: white;
    }
    
    /* Header for TV */
    .tv-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e293b 0%, #334155 50%, #1e293b 100%);
        border-bottom: 3px solid #3b82f6;
    }
    
    .tv-title {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tv-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Status indicator - HUGE for TV */
    .tv-status {
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-radius: 20px;
        animation: pulse 2s infinite;
    }
    
    .tv-status.critical {
        background: #dc2626;
        color: white;
        box-shadow: 0 0 40px rgba(220, 38, 38, 0.5);
        animation: critical-pulse 1s infinite;
    }
    
    .tv-status.warning {
        background: #f59e0b;
        color: #1f2937;
        box-shadow: 0 0 40px rgba(245, 158, 11, 0.5);
    }
    
    .tv-status.good {
        background: #10b981;
        color: white;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
    }
    
    @keyframes critical-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    /* Metric cards for TV - MASSIVE */
    .tv-metric-card {
        background: #1e293b;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        border: 3px solid;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .tv-metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .tv-metric-card.red {
        border-color: #dc2626;
        background: linear-gradient(135deg, #1e293b 0%, #450a0a 100%);
    }
    
    .tv-metric-card.yellow {
        border-color: #f59e0b;
        background: linear-gradient(135deg, #1e293b 0%, #451a03 100%);
    }
    
    .tv-metric-card.green {
        border-color: #10b981;
        background: linear-gradient(135deg, #1e293b 0%, #052e16 100%);
    }
    
    .tv-metric-value {
        font-size: 7rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .tv-metric-label {
        font-size: 2rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .tv-metric-change {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .tv-arrow {
        font-size: 3rem;
    }
    
    /* Alert section for TV */
    .tv-alert-container {
        background: #1e293b;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .tv-alert {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .tv-alert.critical {
        background: rgba(220, 38, 38, 0.2);
        border: 2px solid #dc2626;
        color: #fca5a5;
    }
    
    .tv-alert.warning {
        background: rgba(245, 158, 11, 0.2);
        border: 2px solid #f59e0b;
        color: #fcd34d;
    }
    
    .tv-alert-icon {
        font-size: 3rem;
    }
    
    /* Action items for TV */
    .tv-action {
        background: rgba(59, 130, 246, 0.1);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.75rem;
        font-weight: 600;
        color: #93bbfc;
    }
    
    /* Chart container for TV */
    .tv-chart-container {
        background: #1e293b;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .tv-chart-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Quality score gauge for TV */
    .tv-quality-score {
        font-size: 10rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 0 40px currentColor;
    }
    
    .tv-quality-label {
        font-size: 2.5rem;
        text-align: center;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Live indicator */
    .live-indicator {
        position: fixed;
        top: 2rem;
        right: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        background: rgba(30, 41, 59, 0.9);
        padding: 1rem 2rem;
        border-radius: 50px;
        border: 2px solid #10b981;
    }
    
    .live-dot {
        width: 20px;
        height: 20px;
        background: #10b981;
        border-radius: 50%;
        animation: live-pulse 2s infinite;
    }
    
    @keyframes live-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .live-text {
        color: #10b981;
        font-weight: 700;
        font-size: 1.25rem;
        text-transform: uppercase;
    }
    
    /* Hide scrollbars for TV */
    ::-webkit-scrollbar {
        display: none;
    }
    
    /* Auto-refresh notice */
    .refresh-notice {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        color: #64748b;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- AUTO REFRESH SETUP ---
# Auto-refresh every 60 seconds for TV display
st_autorefresh = st.empty()

# --- DATA PROCESSING ---
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_and_process_data():
    """Load and process quality data with caching"""
    try:
        df = pd.read_csv('quality_data_clean.csv')
        
        # Clean column names
        df.columns = [col.strip().title() for col in df.columns]
        
        # Process data
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Create proper date column
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        # Convert month names to numbers if needed
        if df['Month'].dtype == 'object':
            df['Month_Num'] = df['Month'].map(month_map)
            df['Date'] = pd.to_datetime(
                df[['Year', 'Month_Num']].rename(columns={'Year': 'year', 'Month_Num': 'month'})
            )
        else:
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
            
            # Get last 12 months of data
            last_12_months = metric_data.nlargest(12, 'Date').sort_values('Date')
            
            # Calculate trend
            if len(last_12_months) >= 6:
                recent_6 = last_12_months.tail(6)
                first_val = recent_6.iloc[0]['Value']
                last_val = recent_6.iloc[-1]['Value']
                
                if abs(last_val - first_val) < (first_val * 0.02):
                    trend = 'stable'
                elif last_val > first_val:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'stable'
            
            metrics[metric] = {
                'value': current_val,
                'yoy_change': yoy_change,
                'trend': trend,
                'history': last_12_months[['Date', 'Value']],
                'full_history': metric_data[['Date', 'Value']]
            }
    
    return metrics

def calculate_quality_score(metrics):
    """Calculate overall quality score (0-100)"""
    score = 100
    
    # Return rate impact (most critical)
    if 'Overall Return Rate' in metrics:
        return_rate = metrics['Overall Return Rate']['value']
        if return_rate > 0.10:
            score -= 50  # Critical
        elif return_rate > 0.08:
            score -= 30  # Warning
        elif return_rate > 0.05:
            score -= 15  # Acceptable
    
    # Inspection coverage impact
    if 'Percent Order Inspected' in metrics:
        inspection_rate = metrics['Percent Order Inspected']['value']
        if inspection_rate < 0.80:
            score -= 40  # Critical
        elif inspection_rate < 0.85:
            score -= 20  # Warning
    
    return max(0, min(100, score))

def get_status_and_actions(metrics):
    """Determine overall status and required actions"""
    critical_issues = []
    warnings = []
    urgent_actions = []
    
    # Check return rate
    if 'Overall Return Rate' in metrics:
        return_rate = metrics['Overall Return Rate']['value']
        return_trend = metrics['Overall Return Rate']['trend']
        
        if return_rate > 0.10:
            critical_issues.append(f"RETURN RATE {return_rate:.1%} - EXCEEDS 10%")
            urgent_actions.append("STOP SHIPMENTS - QUALITY REVIEW")
        elif return_rate > 0.08:
            warnings.append(f"Return rate {return_rate:.1%} approaching limit")
            urgent_actions.append("Increase QC to 100% sampling")
        
        if return_trend == 'increasing' and return_rate > 0.05:
            warnings.append("Return rate trending UP")
    
    # Check inspection coverage
    if 'Percent Order Inspected' in metrics:
        inspection_rate = metrics['Percent Order Inspected']['value']
        
        if inspection_rate < 0.80:
            critical_issues.append(f"INSPECTION {inspection_rate:.0%} - BELOW MINIMUM")
            urgent_actions.append("ADD QC STAFF IMMEDIATELY")
        elif inspection_rate < 0.85:
            warnings.append(f"Inspection {inspection_rate:.0%} below target")
    
    # Determine overall status
    if critical_issues:
        status = "🚨 CRITICAL - ACTION REQUIRED"
        status_type = "critical"
    elif warnings:
        status = "⚠️ WARNING - MONITOR CLOSELY"
        status_type = "warning"
    else:
        status = "✅ ALL SYSTEMS NORMAL"
        status_type = "good"
    
    return status, status_type, critical_issues, warnings, urgent_actions

# --- VISUALIZATION FUNCTIONS ---
def create_tv_return_rate_chart(metrics):
    """Create a large, clear return rate chart for TV display"""
    if 'Overall Return Rate' not in metrics:
        return None
    
    data = metrics['Overall Return Rate']['history']
    
    fig = go.Figure()
    
    # Main line with markers
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Value'],
        mode='lines+markers',
        name='Return Rate',
        line=dict(width=6, color='#3b82f6'),
        marker=dict(size=15, color='#3b82f6', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    # Critical threshold
    fig.add_hline(
        y=0.10, 
        line_dash="dash", 
        line_color="#dc2626", 
        line_width=4,
        annotation_text="CRITICAL 10%", 
        annotation_position="left",
        annotation_font_size=20,
        annotation_font_color="#dc2626"
    )
    
    # Warning threshold
    fig.add_hline(
        y=0.08, 
        line_dash="dash", 
        line_color="#f59e0b", 
        line_width=4,
        annotation_text="WARNING 8%", 
        annotation_position="left",
        annotation_font_size=20,
        annotation_font_color="#f59e0b"
    )
    
    # Target line
    fig.add_hline(
        y=0.05, 
        line_dash="dot", 
        line_color="#10b981", 
        line_width=3,
        annotation_text="TARGET 5%", 
        annotation_position="left",
        annotation_font_size=20,
        annotation_font_color="#10b981"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=100, r=50, t=50, b=50),
        xaxis=dict(
            showgrid=True,
            gridcolor='#334155',
            tickfont=dict(size=18, color='#94a3b8'),
            tickformat='%b %Y'
        ),
        yaxis=dict(
            tickformat='.1%',
            showgrid=True,
            gridcolor='#334155',
            tickfont=dict(size=20, color='#94a3b8'),
            title=dict(text="Return Rate %", font=dict(size=24, color='#94a3b8'))
        ),
        hovermode='x unified',
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font=dict(size=18, color='white'),
        showlegend=False
    )
    
    return fig

def create_tv_mini_charts(metrics):
    """Create mini charts for secondary metrics"""
    charts = []
    
    secondary_metrics = [
        ('Total Orders', False, False, '#3b82f6'),
        ('Orders Inspected', False, False, '#10b981'),
        ('Average Cost per Inspection', False, True, '#f59e0b')
    ]
    
    for metric_name, is_percent, is_currency, color in secondary_metrics:
        if metric_name in metrics:
            data = metrics[metric_name]['history'].tail(6)  # Last 6 months
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data['Date'],
                y=data['Value'],
                marker_color=color,
                marker_line_color='white',
                marker_line_width=2
            ))
            
            yaxis_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
            
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=14, color='#94a3b8'),
                    tickformat='%b'
                ),
                yaxis=dict(
                    tickformat=yaxis_format,
                    showgrid=False,
                    tickfont=dict(size=16, color='#94a3b8'),
                    visible=True
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                title=dict(
                    text=metric_name,
                    font=dict(size=20, color='white'),
                    x=0.5,
                    xanchor='center'
                )
            )
            
            charts.append((metric_name, fig, metrics[metric_name]))
    
    return charts

# --- MAIN TV DASHBOARD ---
def main():
    # Container for dark theme
    st.markdown('<div class="tv-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="tv-header">
            <h1 class="tv-title">QUALITY CONTROL CENTER</h1>
            <p class="tv-subtitle">Real-Time Quality Metrics Monitoring</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Live indicator
    st.markdown("""
        <div class="live-indicator">
            <div class="live-dot"></div>
            <span class="live-text">LIVE</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_and_process_data()
    if df is None:
        st.error("❌ DATA CONNECTION LOST")
        time.sleep(10)
        st.rerun()
    
    # Get metrics
    metrics = get_current_metrics(df)
    quality_score = calculate_quality_score(metrics)
    status, status_type, critical_issues, warnings, urgent_actions = get_status_and_actions(metrics)
    
    # Display update time
    latest_date = df['Date'].max()
    current_time = datetime.now()
    st.markdown(f"""
        <div style="text-align: center; color: #64748b; font-size: 1.5rem; margin-bottom: 2rem;">
            Last Data: {latest_date.strftime('%B %Y')} | Updated: {current_time.strftime('%I:%M %p')}
        </div>
    """, unsafe_allow_html=True)
    
    # STATUS BANNER
    st.markdown(f"""
        <div class="tv-status {status_type}">
            {status}
        </div>
    """, unsafe_allow_html=True)
    
    # MAIN METRICS ROW
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # Quality Score
    with col1:
        score_color = "#10b981" if quality_score >= 80 else "#f59e0b" if quality_score >= 60 else "#dc2626"
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="tv-quality-score" style="color: {score_color};">
                    {quality_score}
                </div>
                <div class="tv-quality-label">Quality Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Return Rate
    with col2:
        if 'Overall Return Rate' in metrics:
            return_data = metrics['Overall Return Rate']
            return_rate = return_data['value']
            
            if return_rate > 0.10:
                card_class = "red"
                value_color = "#fca5a5"
            elif return_rate > 0.08:
                card_class = "yellow"
                value_color = "#fcd34d"
            else:
                card_class = "green"
                value_color = "#86efac"
            
            yoy = return_data['yoy_change']
            if yoy is not None:
                if yoy > 0:
                    change_html = f'<span class="tv-arrow" style="color: #dc2626;">↑</span><span style="color: #fca5a5;">{yoy:.0%}</span>'
                else:
                    change_html = f'<span class="tv-arrow" style="color: #10b981;">↓</span><span style="color: #86efac;">{abs(yoy):.0%}</span>'
            else:
                change_html = '<span style="color: #64748b;">No YoY</span>'
            
            st.markdown(f"""
                <div class="tv-metric-card {card_class}">
                    <div class="tv-metric-value" style="color: {value_color};">
                        {return_rate:.1%}
                    </div>
                    <div class="tv-metric-label">Return Rate</div>
                    <div class="tv-metric-change">
                        {change_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Inspection Coverage
    with col3:
        if 'Percent Order Inspected' in metrics:
            inspection_data = metrics['Percent Order Inspected']
            inspection_rate = inspection_data['value']
            
            if inspection_rate < 0.80:
                card_class = "red"
                value_color = "#fca5a5"
            elif inspection_rate < 0.85:
                card_class = "yellow"
                value_color = "#fcd34d"
            else:
                card_class = "green"
                value_color = "#86efac"
            
            st.markdown(f"""
                <div class="tv-metric-card {card_class}">
                    <div class="tv-metric-value" style="color: {value_color};">
                        {inspection_rate:.0%}
                    </div>
                    <div class="tv-metric-label">QC Coverage</div>
                    <div class="tv-metric-change">
                        <span style="color: #64748b;">Min: 80%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Total Orders
    with col4:
        if 'Total Orders' in metrics:
            orders_data = metrics['Total Orders']
            total_orders = orders_data['value']
            
            trend = orders_data['trend']
            if trend == 'increasing':
                trend_html = '<span class="tv-arrow" style="color: #10b981;">↑</span><span style="color: #86efac;">Growing</span>'
            elif trend == 'decreasing':
                trend_html = '<span class="tv-arrow" style="color: #f59e0b;">↓</span><span style="color: #fcd34d;">Declining</span>'
            else:
                trend_html = '<span style="color: #93bbfc;">→ Stable</span>'
            
            st.markdown(f"""
                <div class="tv-metric-card green">
                    <div class="tv-metric-value" style="color: #93bbfc;">
                        {total_orders:,.0f}
                    </div>
                    <div class="tv-metric-label">Total Orders</div>
                    <div class="tv-metric-change">
                        {trend_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # ALERTS AND ACTIONS (if any)
    if critical_issues or warnings or urgent_actions:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="tv-alert-container">', unsafe_allow_html=True)
            st.markdown('<h2 style="color: white; font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">⚠️ ALERTS</h2>', unsafe_allow_html=True)
            
            for issue in critical_issues[:2]:  # Show max 2
                st.markdown(f"""
                    <div class="tv-alert critical">
                        <span class="tv-alert-icon">🚨</span>
                        <span>{issue}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            for warning in warnings[:2]:  # Show max 2
                st.markdown(f"""
                    <div class="tv-alert warning">
                        <span class="tv-alert-icon">⚠️</span>
                        <span>{warning}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="tv-alert-container">', unsafe_allow_html=True)
            st.markdown('<h2 style="color: white; font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📋 REQUIRED ACTIONS</h2>', unsafe_allow_html=True)
            
            for action in urgent_actions[:3]:  # Show max 3
                st.markdown(f"""
                    <div class="tv-action">
                        → {action}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # RETURN RATE TREND CHART
    st.markdown('<div class="tv-chart-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="tv-chart-title">📈 RETURN RATE TREND - LAST 12 MONTHS</h2>', unsafe_allow_html=True)
    
    return_chart = create_tv_return_rate_chart(metrics)
    if return_chart:
        st.plotly_chart(return_chart, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECONDARY METRICS
    st.markdown('<div class="tv-chart-container">', unsafe_allow_html=True)
    
    mini_charts = create_tv_mini_charts(metrics)
    if mini_charts:
        cols = st.columns(len(mini_charts))
        for idx, (name, chart, data) in enumerate(mini_charts):
            with cols[idx]:
                # Show current value above chart
                value = data['value']
                is_currency = 'cost' in name.lower()
                if is_currency:
                    value_str = f"${value:,.0f}"
                else:
                    value_str = f"{value:,.0f}"
                
                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <span style="font-size: 3rem; font-weight: 800; color: white;">{value_str}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh notice
    st.markdown("""
        <div class="refresh-notice">
            Auto-refresh: 60 seconds
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(60)
    st.rerun()

if __name__ == "__main__":
    main()
