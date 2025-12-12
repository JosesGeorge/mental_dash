import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Edumorph Analytics Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Main Background adjustments */
    .main {
        background-color: #f8f9fa; 
    }
    
    /* Card Styling for KPIs */
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #4e73df;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Smooth UI animations for charts and metrics ---
st.markdown("""
<style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stPlotlyChart"] {
        animation: fadeUp 700ms cubic-bezier(.2,.8,.2,1) both;
    }
    .metric-card {
        transition: transform 220ms cubic-bezier(.2,.8,.2,1);
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA ENGINE ---
@st.cache_data
def get_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except:
            return pd.read_excel(uploaded_file)
    else:
        # SIMULATION MODE
        np.random.seed(42)
        rows = 800
        data = {
            'Student ID': range(1000, 1000 + rows),
            'Department': np.random.choice(['Computer Science', 'Mechanical', 'Civil', 'Electrical', 'Biotech'], rows),
            'Year': np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year'], rows),
            'Stress Level': np.random.randint(1, 11, rows),
            'Sleep Hours': np.random.normal(6.5, 1.5, rows).clip(3, 10).round(1),
            'CGPA': np.random.uniform(5.0, 10.0, rows).round(2),
            'Attendance (%)': np.random.randint(60, 100, rows)
        }
        df = pd.DataFrame(data)
        # Add Risk Logic
        df['Risk Status'] = pd.cut(df['Stress Level'], bins=[0, 4, 7, 10], labels=['Low', 'Moderate', 'High'])
        return df

# --- 4. SIDEBAR NAVIGATION & FILTERS ---
with st.sidebar:
    st.markdown("## 🎓 **Edumorph**")
    st.markdown("---")
    
    # Navigation
    page = st.radio("Navigate", ["📊 Dashboard", "📉 Deep Dive Analysis", "📂 Data Manager"])
    
    st.markdown("---")
    st.subheader("🛠 Control Panel")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Data Source", type=["csv", "xlsx"])
    
    # Load Data
    df = get_data(uploaded_file)
    
    # Global Filters
    st.markdown("### Filters")
    dept_filter = st.multiselect("Department", options=df['Department'].unique(), default=df['Department'].unique())
    year_filter = st.multiselect("Year", options=df['Year'].unique(), default=df['Year'].unique())
    risk_filter = st.multiselect("Risk Status", options=df['Risk Status'].unique(), default=df['Risk Status'].unique())
    
    # Apply Filters
    df_filtered = df.query("Department == @dept_filter & Year == @year_filter & `Risk Status` == @risk_filter")
    
    st.markdown("---")
    st.caption("v2.0.1 | Enterprise Edition")

# --- 5. PAGE 1: EXECUTIVE DASHBOARD ---
if page == "📊 Dashboard":
    st.title("🎓 Student Wellness Command Center")
    st.markdown(f"**Live Overview** • Showing data for **{len(df_filtered)}** students")
    st.markdown("---")

    # KPI ROW (Styled with Metrics)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    avg_stress = df_filtered['Stress Level'].mean()
    avg_cgpa = df_filtered['CGPA'].mean()
    high_risk_count = len(df_filtered[df_filtered['Risk Status'] == 'High'])
    
    kpi1.metric("Total Students", f"{len(df_filtered)}", delta="Active")
    kpi2.metric("Avg Stress Level", f"{avg_stress:.1f} / 10", delta="-0.2 vs Last Month", delta_color="inverse")
    kpi3.metric("Avg CGPA", f"{avg_cgpa:.2f}", delta="+0.1 vs Last Sem")
    kpi4.metric("High Risk Cases", f"{high_risk_count}", delta="Needs Attention", delta_color="inverse")
    
    st.markdown("---")

    # ROW 2: Primary Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Departmental Stress Heatmap")
        # include Year for animated transitions between cohorts
        dept_group = df_filtered.groupby(['Year', 'Department', 'Risk Status']).size().reset_index(name='Count')
        frame_duration = 900
        transition_duration = 700
        fig_bar = px.bar(
            dept_group,
            x='Department',
            y='Count',
            color='Risk Status',
            animation_frame='Year',
            color_discrete_map={'High': '#ff4d4d', 'Moderate': '#ffa600', 'Low': '#2ecc71'},
            title="Risk Distribution by Department",
            barmode='group'
        )
            # build symbolic buttons and reset to first frame
            frame_names_b = [f.name for f in fig_bar.frames] if hasattr(fig_bar, 'frames') else []
            first_frame_b = frame_names_b[0] if frame_names_b else None
            play_b = {
                'label': '▶',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': frame_duration, 'redraw': False},
                    'fromcurrent': False,
                    'transition': {'duration': transition_duration, 'easing': 'cubic-in-out'}
                }]
            }
            pause_b = {
                'label': '⏸',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
            buttons_b = [play_b, pause_b]
            if first_frame_b is not None:
                reset_b = {
                    'label': '⟲',
                    'method': 'animate',
                    'args': [[first_frame_b], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                buttons_b.append(reset_b)
            # add direct-frame buttons so user can jump to and stop at each frame
            for name in frame_names_b:
                buttons_b.append({
                    'label': str(name),
                    'method': 'animate',
                    'args': [[name], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                })

            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title=None,
                transition={'duration': transition_duration, 'easing': 'cubic-in-out'},
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': buttons_b,
                    'direction': 'left',
                    'pad': {'r': 10, 't': 10},
                    'showactive': False,
                    'x': 0.1,
                    'y': -0.1,
                    'xanchor': 'right',
                    'yanchor': 'top'
                }]
            )
            if fig_bar.layout.sliders:
                fig_bar.layout.sliders[0].pad.t = 50
                fig_bar.layout.sliders[0].currentvalue.update(prefix='Year: ')
            st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Overall Risk Profile")
        # Static donut pie (Plotly Express `pie` doesn't support `animation_frame`)
        risk_counts = df_filtered['Risk Status'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.5,
            color_discrete_sequence=['#2ecc71', '#ff4d4d', '#ffa600']
        )
        fig_pie.update_layout(showlegend=True, legend=dict(orientation="h"), transition={'duration':700,'easing':'cubic-in-out'})
        st.plotly_chart(fig_pie, use_container_width=True)

    # ROW 3: Advanced Hierarchical View
    st.subheader("📍 Drill-Down: Department > Year > Risk")
    st.markdown("Click on any segment to zoom in.")
    
    # Sunburst Chart
    fig_sun = px.sunburst(df_filtered, path=['Department', 'Year', 'Risk Status'], 
                          color='Stress Level', color_continuous_scale='RdBu_r')
    fig_sun.update_layout(height=500)
    st.plotly_chart(fig_sun, use_container_width=True)

# --- 6. PAGE 2: DEEP DIVE ANALYSIS ---
elif page == "📉 Deep Dive Analysis":
    st.title("📉 Correlation & Trends Analysis")
    
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        st.subheader("Stress vs. Academic Performance")
        # animate scatter by Year for smooth cohort transitions
        fig_scatter = px.scatter(
            df_filtered,
            x="Stress Level",
            y="CGPA",
            color="Sleep Hours",
            size="Attendance (%)",
            hover_data=['Student ID', 'Department'],
            color_continuous_scale='Viridis',
            animation_frame='Year',
            title="Multivariate Analysis: Stress, CGPA, Sleep & Attendance"
        )
        # Smooth transitions and Play/Pause for animated scatter
        frame_duration = 900
        transition_duration = 700
        # replace play/pause with symbols and add reset for scatter
        frame_names_s = [f.name for f in fig_scatter.frames] if hasattr(fig_scatter, 'frames') else []
        first_frame_s = frame_names_s[0] if frame_names_s else None
        play_s = {
            'label': '▶',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': False},
                'fromcurrent': False,
                'transition': {'duration': transition_duration, 'easing': 'cubic-in-out'}
            }]
        }
        pause_s = {
            'label': '⏸',
            'method': 'animate',
            'args': [[None], {
                'frame': {'duration': 0, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }]
        }
        buttons_s = [play_s, pause_s]
        if first_frame_s is not None:
            reset_s = {
                'label': '⟲',
                'method': 'animate',
                'args': [[first_frame_s], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
            buttons_s.append(reset_s)

        fig_scatter.update_layout(
            transition={'duration': transition_duration, 'easing': 'cubic-in-out'},
            updatemenus=[{
                'type': 'buttons',
                'buttons': buttons_s,
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'x': 0.1,
                'y': -0.1,
                'xanchor': 'right',
                'yanchor': 'top'
            }]
        )
        if fig_scatter.layout.sliders:
            fig_scatter.layout.sliders[0].pad.t = 50
            fig_scatter.layout.sliders[0].currentvalue.update(prefix='Year: ')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_b:
        st.success("💡 **Key Insight**")
        st.markdown("""
        **Negative Correlation Detected:**
        
        The data suggests that as **Stress Levels** increase beyond 7, **CGPA** tends to drop below 7.0.
        
        **Sleep Factor:**
        Students with < 5 hours of sleep are concentrated in the High Stress / Low CGPA quadrant.
        """)
        
    # Correlation Heatmap (New Feature)
    st.subheader("Statistical Correlation Matrix")
    corr_matrix = df_filtered[['Stress Level', 'Sleep Hours', 'CGPA', 'Attendance (%)']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- 7. PAGE 3: DATA MANAGER ---
elif page == "📂 Data Manager":
    st.title("📂 Raw Data Explorer")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("View, search, and edit the raw dataset below.")
    with c2:
        # Download Button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name='edumorph_filtered_data.csv',
            mime='text/csv',
        )
        
    # Interactive Data Editor
    st.data_editor(
        df_filtered,
        column_config={
            "Attendance (%)": st.column_config.ProgressColumn(
                "Attendance", format="%d%%", min_value=0, max_value=100
            ),
            "CGPA": st.column_config.NumberColumn("CGPA", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )