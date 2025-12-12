import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# --- 1. APP CONFIGURATION (Dark Mode Enabled) ---
st.set_page_config(
    page_title="Edumorph Analytics Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIGH-END CUSTOM CSS (The Futuristic Dark Theme) ---
st.markdown("""
<style>
    /* --- GLOBAL DARK THEME OVERRIDES --- */
    .stApp {
        background-color: #0E1117; /* Deep Space Blue/Black */
        color: #FAFAFA;
    }
    
    /* --- TYPOGRAPHY --- */
    h1, h2, h3, .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    /* --- METRIC CARDS (KPIs) --- */
    [data-testid="stMetric"] {
        background-color: #161B22; /* Dark Card Background */
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #8B949E;
    }
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #58A6FF; /* Cyber Blue Text */
    }
    
    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363D;
    }
    
    /* --- TABS STYLING --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161B22;
        border-radius: 6px;
        color: #8B949E;
        border: 1px solid #30363D;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important; /* Success Green for active tab */
        color: #FFFFFF !important;
        border: 1px solid #238636;
    }

    /* --- SMOOTH UI & CHART ENTRY ANIMATIONS --- */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Animate Plotly charts as they mount */
    div[data-testid="stPlotlyChart"] {
        animation: fadeUp 700ms cubic-bezier(.2,.8,.2,1) both;
    }

    /* Subtle lift on KPI hover */
    [data-testid="stMetric"] {
        transition: transform 220ms cubic-bezier(.2,.8,.2,1);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
    }

    /* Buttons get a nice micro-interaction */
    .stButton>button {
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA ENGINE (Your Original Logic) ---
@st.cache_data
def get_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_excel(uploaded_file)

        # Ensure `Risk Status` exists. If missing, derive from `Stress Level` when possible.
        if 'Risk Status' not in df.columns:
            if 'Stress Level' in df.columns:
                df['Risk Status'] = pd.cut(df['Stress Level'], bins=[0, 4, 7, 10], labels=['Low', 'Moderate', 'High'])
            else:
                df['Risk Status'] = 'Unknown'

        return df
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
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942813.png", width=60)
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

    # KPI ROW
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    avg_stress = df_filtered['Stress Level'].mean()
    avg_cgpa = df_filtered['CGPA'].mean()
    high_risk_count = len(df_filtered[df_filtered['Risk Status'] == 'High'])
    
    kpi1.metric("Total Students", f"{len(df_filtered)}", delta="Active")
    kpi2.metric("Avg Stress Level", f"{avg_stress:.1f} / 10", delta="-0.2 vs Last Month", delta_color="inverse")
    kpi3.metric("Avg CGPA", f"{avg_cgpa:.2f}", delta="+0.1 vs Last Sem")
    kpi4.metric("High Risk Cases", f"{high_risk_count}", delta="Needs Attention", delta_color="inverse")
    
    st.markdown("---")

    # Custom Tabs: Analytics Overview and Drill-Down
    tab1, tab2 = st.tabs(["📊 Analytics Overview", "📍 Drill-Down"])

    with tab1:
        # ROW 2: Primary Charts
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Departmental Stress Heatmap")
            # include Year as an animation frame so the chart smoothly transitions across years
            dept_group = df_filtered.groupby(['Year', 'Department', 'Risk Status']).size().reset_index(name='Count')

            # Animated Bar Chart across Year
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

            # Apply Dark Theme, transitions and transparent background
            # Smooth transitions and controlled frame timing
            # build symbolic buttons and a reset button to jump back to first frame
            frame_names = [f.name for f in fig_bar.frames] if hasattr(fig_bar, 'frames') else []
            first_frame = frame_names[0] if frame_names else None
            play_btn = {
                'label': '▶',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': frame_duration, 'redraw': False},
                    'fromcurrent': False,
                    'transition': {'duration': transition_duration, 'easing': 'cubic-in-out'}
                }]
            }
            pause_btn = {
                'label': '⏸',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
            buttons = [play_btn, pause_btn]
            if first_frame is not None:
                reset_btn = {
                    'label': '⟲',
                    'method': 'animate',
                    'args': [[first_frame], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                buttons.append(reset_btn)

            fig_bar.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                transition={'duration': transition_duration, 'easing': 'cubic-in-out'},
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': buttons,
                    'direction': 'left',
                    'pad': {'r': 10, 't': 10},
                    'showactive': False,
                    'x': 0.1,
                    'y': -0.1,
                    'xanchor': 'right',
                    'yanchor': 'top'
                }]
            )
            # Tidy slider appearance if present
            if fig_bar.layout.sliders:
                fig_bar.layout.sliders[0].pad.t = 50
                fig_bar.layout.sliders[0].currentvalue.update(prefix='Year: ')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("Overall Risk Profile")
            # Use a static donut pie (Plotly Express `pie` doesn't accept `animation_frame`)
            risk_counts = df_filtered['Risk Status'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.6,
                color_discrete_sequence=['#2ecc71', '#ff4d4d', '#ffa600']
            )
            fig_pie.update_layout(
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                transition={'duration': 700, 'easing': 'cubic-in-out'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # -- moved drill-down to its own tab for clarity --

    with tab2:
        st.markdown(f"### 📍 Hierarchical Drill-Down")
        
        # Select Drill Path
        cols_available = [c for c in ['Year', 'Department', 'Employment', 'Gender', 'Country'] if c in df.columns]
        path_sel = st.multiselect("Select Hierarchy Path", cols_available, default=cols_available[:2])
        path = path_sel + ['Risk Status']
        
        chart_type = st.selectbox("Chart Type", ["Sunburst", "Treemap"])
        
        if not df.empty and path:
            # --- FIX: ROBUST AGGREGATION ---
            # 1. Calculate Counts using size() (Counts rows regardless of missing values)
            df_counts = df.groupby(path).size().reset_index(name='Count')
            
            # 2. Calculate Average Stress (ignores NaNs automatically)
            df_stress = df.groupby(path)['Stress Level'].mean().reset_index(name='Avg_Stress')
            
            # 3. Merge them safely
            df_agg = pd.merge(df_counts, df_stress, on=path)
            
            # 4. Fill missing stress averages (e.g., if all were NaN) with 0 or global mean to prevent errors
            df_agg['Avg_Stress'] = df_agg['Avg_Stress'].fillna(0)
            
            # 5. Filter out zero-count groups (just in case)
            df_agg = df_agg[df_agg['Count'] > 0]
            
            if chart_type == "Sunburst":
                fig_drill = px.sunburst(df_agg, path=path, values='Count', color='Avg_Stress', 
                                        color_continuous_scale='RdBu_r')
            else:
                fig_drill = px.treemap(df_agg, path=path, values='Count', color='Avg_Stress', 
                                       color_continuous_scale='RdBu_r')
            
            fig_drill.update_layout(template="plotly_dark", height=600, paper_bgcolor="#262626")
            st.plotly_chart(fig_drill, use_container_width=True)
        else:
            st.warning("Select at least one hierarchy level.")

    # --- PAGE 2 LOGIC MOVED INSIDE TABS IF DESIRED, OR KEPT SEPARATE ---
    # Since your original code had Page 2 as a Sidebar option, let's keep it that way for logic consistency.

# --- 6. PAGE 2: DEEP DIVE ANALYSIS ---
elif page == "📉 Deep Dive Analysis":
    st.title("📉 Correlation & Trends Analysis")
    
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        st.subheader("Stress vs. Academic Performance")
        # Add per-year animation for smooth transitions between cohorts
        # Smooth animated scatter with controlled timing and Play/Pause
        frame_duration = 900
        transition_duration = 700
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
        # replace play/pause labels with symbols and add reset
        frame_names_s = [f.name for f in fig_scatter.frames] if hasattr(fig_scatter, 'frames') else []
        first_frame_s = frame_names_s[0] if frame_names_s else None
        play_btn_s = {
            'label': '▶',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': False},
                'fromcurrent': False,
                'transition': {'duration': transition_duration, 'easing': 'cubic-in-out'}
            }]
        }
        pause_btn_s = {
            'label': '⏸',
            'method': 'animate',
            'args': [[None], {
                'frame': {'duration': 0, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }]
        }
        buttons_s = [play_btn_s, pause_btn_s]
        if first_frame_s is not None:
            reset_btn_s = {
                'label': '⟲',
                'method': 'animate',
                'args': [[first_frame_s], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
            buttons_s.append(reset_btn_s)

        fig_scatter.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
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
        st.info("💡 **Key Insight**")
        st.markdown("""
        **Negative Correlation Detected:**
        
        The data suggests that as **Stress Levels** increase beyond 7, **CGPA** tends to drop below 7.0.
        
        **Sleep Factor:**
        Students with < 5 hours of sleep are concentrated in the High Stress / Low CGPA quadrant.
        """)
        
    # Correlation Heatmap
    st.subheader("Statistical Correlation Matrix")
    corr_matrix = df_filtered[['Stress Level', 'Sleep Hours', 'CGPA', 'Attendance (%)']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    fig_corr.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
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