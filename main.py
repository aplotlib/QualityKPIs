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
import os

# --- PAGE CONFIGURATION FOR TV DISPLAY ---
st.set_page_config(
    page_title="Quality TV Dashboard",
    page_icon="📺",
    layout="wide",
)

# --- BRIGHT TV DISPLAY STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    /* Bright Theme for TV Display */
    html, body, [class*="st-"] { 
        font-family: 'Inter', sans-serif;
        background: #ffffff;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    .main .block-container { 
        padding: 1.5rem;
        max-width: 100%;
        background: #ffffff;
    }
    
    /* Hide all Streamlit chrome for TV */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    
    /* Header for TV */
    .tv-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 50%, #1e40af 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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
        color: rgba(255,255,255,0.9);
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
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .tv-status.critical {
        background: #dc2626;
        color: white;
        animation: critical-flash 2s infinite;
    }
    
    .tv-status.warning {
        background: #f59e0b;
        color: white;
    }
    
    .tv-status.good {
        background: #10b981;
        color: white;
    }
    
    @keyframes critical-flash {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Metric cards for TV - MASSIVE and BRIGHT */
    .tv-metric-card {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        border: 4px solid;
        height: 100%;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .tv-metric-card.red {
        border-color: #dc2626;
        background: #fee2e2;
    }
    
    .tv-metric-card.yellow {
        border-color: #f59e0b;
        background: #fef3c7;
    }
    
    .tv-metric-card.green {
        border-color: #10b981;
        background: #d1fae5;
    }
    
    .tv-metric-card.blue {
        border-color: #3b82f6;
        background: #dbeafe;
    }
    
    .tv-metric-value {
        font-size: 6rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
    }
    
    .tv-metric-label {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tv-metric-change {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .tv-arrow {
        font-size: 2.5rem;
    }
    
    /* Alert section for TV */
    .tv-alert-container {
        background: #f3f4f6;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .tv-alert {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        font-size: 1.75rem;
        font-weight: 700;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .tv-alert.critical {
        background: #dc2626;
        color: white;
    }
    
    .tv-alert.warning {
        background: #f59e0b;
        color: white;
    }
    
    .tv-alert-icon {
        font-size: 2.5rem;
    }
    
    /* Action items for TV */
    .tv-action {
        background: #3b82f6;
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Chart container for TV */
    .tv-chart-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .tv-chart-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Quality score gauge for TV */
    .tv-quality-score {
        font-size: 8rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
    }
    
    .tv-quality-label {
        font-size: 2rem;
        text-align: center;
        color: #4b5563;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    /* Time display */
    .time-display {
        text-align: center;
        color: #6b7280;
        font-size: 1.75rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Hide scrollbars for TV */
    ::-webkit-scrollbar {
        display: none;
    }
    
    /* Mini chart styling */
    .mini-chart-value {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .mini-chart-number {
        font-size: 3rem;
        font-weight: 800;
        color: #1f2937;
    }
    
    .mini-chart-label {
        font-size: 1.25rem;
        color: #6b7280;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING ---
def load_and_process_data():
    """Load and process quality data"""
    try:
        # Check if file exists
        if not os.path.exists('quality_data_clean.csv'):
            st.error("❌ Cannot find quality_data_clean.csv file")
            return None
            
        df = pd.read_csv('quality_data_clean.csv')
        
        # Clean column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Process data
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Create proper date column - handle month names
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        # Convert month to lowercase for mapping
        df['month_lower'] = df['month'].str.lower()
        df['month_num'] = df['month_lower'].map(month_map)
        
        # Create date
        df['date'] = pd.to_datetime(
            df[['year', 'month_num']].rename(columns={'year': 'year', 'month_num': 'month'})
        )
        
        # Clean up metric names
        df['metric'] = df['metric'].str.strip()
        
        # Sort by date
        df = df.sort_values(['metric', 'date'])
        
        # Calculate changes
        df['mom_change'] = df.groupby('metric')['value'].pct_change()
        df['yoy_change'] = df.groupby('metric')['value'].pct_change(12)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_current_metrics(df):
    """Get current values and changes for all metrics"""
    if df is None or df.empty:
        return {}
        
    latest_date = df['date'].max()
    metrics = {}
    
    for metric in df['metric'].unique():
        metric_data = df[df['metric'] == metric]
        current = metric_data[metric_data['date'] == latest_date]
        
        if not current.empty:
            current_val = current.iloc[0]['value']
            
            # Get YoY value
            prev_year = metric_data[metric_data['date'] == latest_date - pd.DateOffset(years=1)]
            yoy_change = None
            if not prev_year.empty:
                yoy_change = (current_val - prev_year.iloc[0]['value']) / prev_year.iloc[0]['value']
            
            # Get last 12 months of data
            last_12_months = metric_data.nlargest(12, 'date').sort_values('date')
            
            # Calculate trend
            if len(last_12_months) >= 6:
                recent_6 = last_12_months.tail(6)
                first_val = recent_6.iloc[0]['value']
                last_val = recent_6.iloc[-1]['value']
                
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
                'history': last_12_months[['date', 'value']],
                'full_history': metric_data[['date', 'value']]
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
        x=data['date'],
        y=data['value'],
        mode='lines+markers',
        name='Return Rate',
        line=dict(width=6, color='#2563eb'),
        marker=dict(size=20, color='#2563eb', line=dict(width=3, color='white')),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)'
    ))
    
    # Add value labels on points
    for _, row in data.iterrows():
        fig.add_annotation(
            x=row['date'],
            y=row['value'],
            text=f"{row['value']:.1%}",
            showarrow=False,
            yshift=30,
            font=dict(size=18, color='#1f2937', weight=700)
        )
    
    # Critical threshold
    fig.add_hline(
        y=0.10, 
        line_dash="dash", 
        line_color="#dc2626", 
        line_width=5,
        annotation_text="CRITICAL 10%", 
        annotation_position="right",
        annotation_font_size=24,
        annotation_font_color="#dc2626",
        annotation_font_weight=700
    )
    
    # Warning threshold
    fig.add_hline(
        y=0.08, 
        line_dash="dash", 
        line_color="#f59e0b", 
        line_width=5,
        annotation_text="WARNING 8%", 
        annotation_position="right",
        annotation_font_size=24,
        annotation_font_color="#f59e0b",
        annotation_font_weight=700
    )
    
    # Target line
    fig.add_hline(
        y=0.05, 
        line_dash="dot", 
        line_color="#10b981", 
        line_width=4,
        annotation_text="TARGET 5%", 
        annotation_position="right",
        annotation_font_size=22,
        annotation_font_color="#10b981",
        annotation_font_weight=700
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=120, r=120, t=50, b=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickfont=dict(size=20, color='#374151', weight=600),
            tickformat='%b<br>%Y',
            dtick='M1'  # Show every month
        ),
        yaxis=dict(
            tickformat='.0%',
            showgrid=True,
            gridcolor='#e5e7eb',
            tickfont=dict(size=24, color='#374151', weight=700),
            title=dict(text="Return Rate %", font=dict(size=28, color='#1f2937', weight=700)),
            range=[0, max(0.12, data['value'].max() * 1.1)]  # Always show up to at least 12%
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=20, color='#1f2937'),
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
                x=data['date'],
                y=data['value'],
                marker_color=color,
                marker_line_color='white',
                marker_line_width=3,
                text=data['value'].map(lambda x: f"${x:.0f}" if is_currency else f"{x:.0f}"),
                textposition='outside',
                textfont=dict(size=18, color='#1f2937', weight=700)
            ))
            
            yaxis_format = '.1%' if is_percent else ('$,.0f' if is_currency else ',.0f')
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=60, b=40),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=16, color='#374151', weight=600),
                    tickformat='%b'
                ),
                yaxis=dict(
                    tickformat=yaxis_format,
                    showgrid=True,
                    gridcolor='#f3f4f6',
                    tickfont=dict(size=18, color='#374151', weight=600),
                    visible=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#1f2937'),
                showlegend=False,
                title=dict(
                    text=metric_name,
                    font=dict(size=24, color='#1f2937', weight=700),
                    x=0.5,
                    xanchor='center'
                )
            )
            
            charts.append((metric_name, fig, metrics[metric_name]))
    
    return charts

# --- MAIN TV DASHBOARD ---
def main():
    # Header
    st.markdown("""
        <div class="tv-header">
            <h1 class="tv-title">QUALITY CONTROL CENTER</h1>
            <p class="tv-subtitle">Real-Time Quality Metrics Monitoring</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_and_process_data()
    if df is None:
        st.stop()
    
    # Get metrics
    metrics = get_current_metrics(df)
    
    if not metrics:
        st.error("❌ No metrics data available")
        st.stop()
    
    quality_score = calculate_quality_score(metrics)
    status, status_type, critical_issues, warnings, urgent_actions = get_status_and_actions(metrics)
    
    # Display update time
    latest_date = df['date'].max()
    current_time = datetime.now()
    st.markdown(f"""
        <div class="time-display">
            Data Period: {latest_date.strftime('%B %Y')} | Dashboard Updated: {current_time.strftime('%I:%M %p - %B %d, %Y')}
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
                value_color = "#dc2626"
            elif return_rate > 0.08:
                card_class = "yellow"
                value_color = "#f59e0b"
            else:
                card_class = "green"
                value_color = "#10b981"
            
            yoy = return_data['yoy_change']
            if yoy is not None:
                if yoy > 0:
                    change_html = f'<span class="tv-arrow" style="color: #dc2626;">↑</span><span style="color: #dc2626;">{yoy:.0%} YoY</span>'
                else:
                    change_html = f'<span class="tv-arrow" style="color: #10b981;">↓</span><span style="color: #10b981;">{abs(yoy):.0%} YoY</span>'
            else:
                change_html = '<span style="color: #6b7280;">No YoY Data</span>'
            
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
                value_color = "#dc2626"
            elif inspection_rate < 0.85:
                card_class = "yellow"
                value_color = "#f59e0b"
            else:
                card_class = "green"
                value_color = "#10b981"
            
            st.markdown(f"""
                <div class="tv-metric-card {card_class}">
                    <div class="tv-metric-value" style="color: {value_color};">
                        {inspection_rate:.0%}
                    </div>
                    <div class="tv-metric-label">QC Coverage</div>
                    <div class="tv-metric-change">
                        <span style="color: #6b7280;">Min Required: 80%</span>
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
                trend_html = '<span class="tv-arrow" style="color: #10b981;">↑</span><span style="color: #10b981;">Growing</span>'
            elif trend == 'decreasing':
                trend_html = '<span class="tv-arrow" style="color: #f59e0b;">↓</span><span style="color: #f59e0b;">Declining</span>'
            else:
                trend_html = '<span style="color: #3b82f6;">→ Stable</span>'
            
            st.markdown(f"""
                <div class="tv-metric-card blue">
                    <div class="tv-metric-value" style="color: #3b82f6;">
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
            st.markdown('<h2 style="color: #1f2937; font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">⚠️ ALERTS</h2>', unsafe_allow_html=True)
            
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
            st.markdown('<h2 style="color: #1f2937; font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📋 REQUIRED ACTIONS</h2>', unsafe_allow_html=True)
            
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
                    <div class="mini-chart-value">
                        <div class="mini-chart-number">{value_str}</div>
                        <div class="mini-chart-label">Current</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh every 5 minutes
    time.sleep(300)
    st.rerun()

if __name__ == "__main__":
    main()
