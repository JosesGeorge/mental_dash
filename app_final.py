import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

# --- 1. APP CONFIGURATION (IBM Carbon Dark Mode) ---
st.set_page_config(
    page_title="Mental Analytics Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIGH-END CUSTOM CSS ---
st.markdown("""
<style>
    /* --- GLOBAL DARK THEME --- */
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
    
    /* --- METRIC CARDS --- */
    [data-testid="stMetric"] {
        background-color: #161B22;
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
        color: #58A6FF;
    }
    
    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363D;
    }
    
    /* --- TABS --- */
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
        background-color: #238636 !important;
        color: #FFFFFF !important;
        border: 1px solid #238636;
    }
    
    /* --- ANIMATIONS --- */
    div[data-testid="stPlotlyChart"] {
        animation: fadeUp 0.7s ease-in-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def clean_text_to_number(val):
    """Converts text numbers like 'twenty-five' to integers."""
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    mapping = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'twenty': 20, 'twenty-five': 25, 'thirty': 30
    }
    if val in mapping: return mapping[val]
    try: return float(re.sub(r'[^\d\.-]', '', val))
    except: return np.nan

def clean_stress_level(val):
    """Normalizes messy stress inputs."""
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    if 'high' in val: return 9
    if 'medium' in val: return 5
    if 'low' in val: return 2
    try:
        score = float(re.sub(r'[^\d\.]', '', val))
        return 10 if score > 10 else score
    except: return np.nan

# --- 4. ROBUST DATA ENGINE ---
@st.cache_data
def process_data(uploaded_file):
    """Universal Data Processor"""
    if uploaded_file is None: return None, None

    try:
        raw_df = pd.read_csv(uploaded_file)
    except:
        raw_df = pd.read_excel(uploaded_file)

    df = raw_df.copy()

    # 1. UNIVERSAL COLUMN MAPPING
    col_map = {
        'stress_level': 'Stress Level', 'stress': 'Stress Level',
        'sleep_hours': 'Sleep Hours', 'sleep': 'Sleep Hours',
        'cgpa': 'CGPA', 'grade': 'CGPA',
        'attendance': 'Attendance (%)', 'attendance_pct': 'Attendance (%)',
        'dept': 'Department', 'department': 'Department',
        'year': 'Year',
        'employment_status': 'Employment', 'employment': 'Employment',
        'age': 'Age', 'gender': 'Gender', 'country': 'Country'
    }
    
    # Normalize and Rename
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    # 2. CLEANING ROUTINES
    
    # Clean Stress
    if 'Stress Level' in df.columns:
        df['Stress Level'] = df['Stress Level'].apply(clean_stress_level)
        df['Stress Level'] = df['Stress Level'].fillna(df['Stress Level'].median())

    # Clean Sleep
    if 'Sleep Hours' in df.columns:
        df['Sleep Hours'] = pd.to_numeric(df['Sleep Hours'], errors='coerce').abs()
        df['Sleep Hours'] = df['Sleep Hours'].fillna(df['Sleep Hours'].mean())

    # Clean Attendance
    if 'Attendance (%)' in df.columns:
        df['Attendance (%)'] = pd.to_numeric(df['Attendance (%)'], errors='coerce')
        df['Attendance (%)'] = df['Attendance (%)'].abs() 
        df['Attendance (%)'] = df['Attendance (%)'].fillna(df['Attendance (%)'].mean())

    # Clean Age
    if 'Age' in df.columns:
        df['Age'] = df['Age'].apply(clean_text_to_number)
        df.loc[(df['Age'] > 100) | (df['Age'] < 10), 'Age'] = np.nan
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # Standardize Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
        df['Gender'] = df['Gender'].replace({'f': 'Female', 'woman': 'Female', 'm': 'Male', 'man': 'Male'})
        df.loc[~df['Gender'].isin(['Female', 'Male']), 'Gender'] = 'Other'

    # 3. FEATURE ENGINEERING (Risk Status)
    if 'Stress Level' in df.columns:
        sleep_col = df['Sleep Hours'] if 'Sleep Hours' in df.columns else 10
        conditions = [
            (df['Stress Level'] >= 8) | (sleep_col < 4),
            (df['Stress Level'] >= 5),
            (df['Stress Level'] < 5)
        ]
        choices = ['High', 'Moderate', 'Low']
        df['Risk Status'] = np.select(conditions, choices, default='Moderate')

    # 4. Final Cleanup
    df = df.drop_duplicates()
    
    if 'Year' not in df.columns:
        df['Year'] = 'All' 

    return raw_df, df

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942813.png", width=60)
    st.markdown("## 🎓 **Edumorph**")
    st.caption("Universal Analytics Pipeline")
    st.markdown("---")
    
    page = st.radio("Navigate", ["📊 Dashboard", "📉 Deep Dive Analysis", "📂 Data Manager"])
    
    st.markdown("---")
    st.subheader("🛠 Data Loader")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        raw_df, df = process_data(uploaded_file)
        
        # DYNAMIC FILTERS
        st.markdown("### Filters")
        
        group_cols = [c for c in ['Department', 'Employment', 'Country'] if c in df.columns]
        main_group = group_cols[0] if group_cols else 'Risk Status'
        
        if main_group in df.columns:
            opts = sorted(df[main_group].astype(str).unique())
            sel = st.multiselect(f"Filter {main_group}", opts, default=opts)
            if sel: df = df[df[main_group].isin(sel)]

        if 'Year' in df.columns and df['Year'].nunique() > 1:
            y_opts = sorted(df['Year'].astype(str).unique())
            y_sel = st.multiselect("Filter Year", y_opts, default=y_opts)
            if y_sel: df = df[df['Year'].isin(y_sel)]
            
        if 'Risk Status' in df.columns:
            r_opts = df['Risk Status'].unique()
            r_sel = st.multiselect("Risk Category", r_opts, default=r_opts)
            if r_sel: df = df[df['Risk Status'].isin(r_sel)]
            
        df_filtered = df
            
    else:
        st.info("👆 Upload data to begin.")
        st.stop()

# --- 6. DASHBOARD PAGE ---
if page == "📊 Dashboard":
    st.title("🎓 Student Wellness Command Center")
    st.markdown(f"**Live Overview** • Analysis of **{len(df_filtered)}** records")
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    
    avg_stress = df_filtered['Stress Level'].mean() if 'Stress Level' in df_filtered.columns else 0
    high_risk = len(df_filtered[df_filtered['Risk Status'] == 'High']) if 'Risk Status' in df_filtered.columns else 0
    
    k1.metric("Total Records", f"{len(df_filtered)}")
    k2.metric("Avg Stress Index", f"{avg_stress:.1f} / 10", delta="Live", delta_color="inverse")
    
    if 'Sleep Hours' in df_filtered.columns:
        avg_sleep = df_filtered['Sleep Hours'].mean()
        k3.metric("Avg Sleep", f"{avg_sleep:.1f} hrs", delta="-Target")
    else:
        k3.metric("Data Quality", "Active", delta="OK")
        
    k4.metric("High Risk Cases", f"{high_risk}", delta="Action Needed", delta_color="inverse")

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Analytics Overview", "📍 Drill-Down"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            x_axis = main_group
            st.subheader(f"Stress Distribution by {x_axis}")
            
            if 'Year' in df_filtered.columns and df_filtered['Year'].nunique() > 1:
                df_grp = df_filtered.groupby(['Year', x_axis, 'Risk Status']).size().reset_index(name='Count')
                anim_frame = 'Year'
            else:
                df_grp = df_filtered.groupby([x_axis, 'Risk Status']).size().reset_index(name='Count')
                anim_frame = None
            
            colors = {'High': '#DA1E28', 'Moderate': '#F1C21B', 'Low': '#24A148'}
            
            fig_bar = px.bar(df_grp, x=x_axis, y='Count', color='Risk Status',
                             animation_frame=anim_frame,
                             color_discrete_map=colors, barmode='group')
            
            if anim_frame and fig_bar.layout.updatemenus:
                try: fig_bar.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
                except: pass
            
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#262626")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("Risk Composition")
            if 'Risk Status' in df_filtered.columns:
                risk_counts = df_filtered['Risk Status'].value_counts()
                fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.6,
                                 color_discrete_sequence=['#24A148', '#DA1E28', '#F1C21B'])
                fig_pie.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#262626",
                                      legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown(f"### 📍 Hierarchical Drill-Down")
        
        target_df = df_filtered 
        
        possible_cols = ['Year', 'Department', 'Employment', 'Gender', 'Country']
        cols_available = [c for c in possible_cols if c in target_df.columns]
        default_path = cols_available[:2] if len(cols_available) >= 2 else cols_available
        
        path_sel = st.multiselect("Select Hierarchy Path", cols_available, default=default_path)
        path = path_sel + ['Risk Status']
        chart_type = st.selectbox("Chart Type", ["Sunburst", "Treemap"])
        
        if not target_df.empty and path:
            df_counts = target_df.groupby(path).size().reset_index(name='Count')
            df_stress = target_df.groupby(path)['Stress Level'].mean().reset_index(name='Avg_Stress')
            df_agg = pd.merge(df_counts, df_stress, on=path)
            df_agg['Avg_Stress'] = df_agg['Avg_Stress'].fillna(0) 
            df_agg = df_agg[df_agg['Count'] > 0]
            
            if chart_type == "Sunburst":
                fig_drill = px.sunburst(df_agg, path=path, values='Count', color='Avg_Stress', 
                                        color_continuous_scale='RdBu_r')
            else:
                fig_drill = px.treemap(df_agg, path=path, values='Count', color='Avg_Stress', 
                                       color_continuous_scale='RdBu_r')
            
            fig_drill.update_layout(template="plotly_dark", height=600, paper_bgcolor="#262626")
            st.plotly_chart(fig_drill, use_container_width=True)
            
            with st.expander("View Aggregated Data"):
                st.dataframe(df_agg)
        else:
            st.warning("Please select at least one hierarchy level.")

# --- 7. DEEP DIVE ---
# --- 7. DEEP DIVE ---
elif page == "📉 Deep Dive Analysis":
    st.title("Correlation Analysis")
    
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        st.subheader("Stress vs. Sleep Analysis")
        if 'Sleep Hours' in df_filtered.columns and 'Stress Level' in df_filtered.columns:
            
            # Setup Animation Frame
            anim_frame = 'Year' if ('Year' in df_filtered.columns and df_filtered['Year'].nunique() > 1) else None
            
            # Handle Attendance Size
            size_col = 'Attendance (%)' if 'Attendance (%)' in df_filtered.columns else None
            
            # --- MODIFIED SCATTER PLOT ---
            fig_scat = px.scatter(df_filtered, 
                                  x="Sleep Hours", 
                                  y="Stress Level", 
                                  color="Stress Level",       # <--- Continuous variable for Gradient
                                  size=size_col,
                                  size_max=35,                # <--- Makes bubbles much bigger
                                  animation_frame=anim_frame,
                                  color_continuous_scale="RdBu_r", # <--- Cool-to-Hot Gradient
                                  hover_data=[main_group])
            
            # Animation Safety
            if anim_frame and fig_scat.layout.updatemenus:
                try: fig_scat.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
                except: pass

            fig_scat.update_layout(template="plotly_dark", paper_bgcolor="#262626", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Sleep/Stress data missing.")
            
    with col_b:
        st.info("💡 **Insight:** Lower sleep hours (< 5) typically show strong clustering with High Stress levels.")

    st.subheader("Statistical Matrix")
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="#262626")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- 8. DATA MANAGER (Updated with Metrics & Split Layout) ---
elif page == "📂 Data Manager":
    st.title("📂 Data Pipeline Explorer")
    st.markdown("Inspect the transformation from **Raw Uploaded Data** to **Cleaned Analytics Data**.")
    
    # Alias df for consistency with user snippet
    df_cleaned = df 
    
    tab_raw, tab_clean = st.tabs(["📄 Raw Unprocessed Data", "✨ Cleaned & Processed Data"])
    
    with tab_raw:
        st.subheader("Raw Data Source")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(raw_df, use_container_width=True)
        with c2:
            st.info(f"**Rows:** {raw_df.shape[0]}\n\n**Columns:** {raw_df.shape[1]}")

    with tab_clean:
        st.subheader("Cleaned Analysis-Ready Data")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(df_cleaned, use_container_width=True)
        with c2:
            st.success(f"**Rows:** {df_cleaned.shape[0]}\n\n**Columns:** {df_cleaned.shape[1]}")
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cleaned Data", csv, 'edumorph_cleaned_data.csv', 'text/csv')

    st.divider()
    st.subheader("Pipeline Metrics")
    m1, m2, m3 = st.columns(3)
    rows_dropped = raw_df.shape[0] - df_cleaned.shape[0]
    cols_added = df_cleaned.shape[1] - raw_df.shape[1]
    m1.metric("Rows Dropped (Duplicates)", rows_dropped)
    m2.metric("New Features Engineered", cols_added)
    m3.metric("Data Health", "100%", "Ready for AI")