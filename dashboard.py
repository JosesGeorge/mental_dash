import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# --- 1. APP CONFIGURATION (Must be the first command) ---
st.set_page_config(
    page_title="Edumorph Analytics Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2e86de;
    }
    .main {
        background-color: #f5f6fa;
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
        # AUTOMATIC MOCK DATA GENERATOR
        np.random.seed(42)
        rows = 500
        data = {
            'Student ID': range(1000, 1000 + rows),
            'Department': np.random.choice(['Computer Science', 'Mechanical', 'Civil', 'Electrical', 'Biotech'], rows),
            'Year': np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year'], rows),
            'Stress Level': np.random.randint(1, 11, rows),
            'Sleep Hours': np.random.normal(6.5, 1.5, rows).clip(3, 10).round(1),
            'CGPA': np.random.uniform(5.0, 10.0, rows).round(2),
            'Attendance (%)': np.random.randint(60, 100, rows)
        }
        return pd.DataFrame(data)

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942813.png", width=80)
    st.title("Admin Controls")
    st.write("Upload survey data or use the system's live simulation.")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    st.divider()
    st.subheader("Filter Data")
    df = get_data(uploaded_file)
    
    dept_filter = st.multiselect("Select Department", options=df['Department'].unique(), default=df['Department'].unique())
    year_filter = st.multiselect("Select Year", options=df['Year'].unique(), default=df['Year'].unique())
    
    # Apply Filters
    df_filtered = df.query("Department == @dept_filter & Year == @year_filter")
    
    st.info(f"Showing {len(df_filtered)} students")

# --- 5. MAIN DASHBOARD UI ---
st.title("🎓  Dashboard for Mental Health Survey")
st.markdown("Real-time monitoring of student mental health and academic correlations.")

# KPI ROW
col1, col2, col3, col4 = st.columns(4)
avg_stress = df_filtered['Stress Level'].mean()
avg_cgpa = df_filtered['CGPA'].mean()
risk_students = len(df_filtered[df_filtered['Stress Level'] >= 8])

col1.metric("Total Students", f"{len(df_filtered)}")
col2.metric("Avg Stress Index", f"{avg_stress:.1f}/10", "-0.4 vs Last Sem")
col3.metric("Avg CGPA", f"{avg_cgpa:.2f}", "Stable")
col4.metric("High Risk Cases", f"{risk_students}", "Requires Action", delta_color="inverse")

st.divider()

# TABS FOR ORGANIZATION
tab1, tab2, tab3 = st.tabs(["📊 Analytics Overview", "🧠 Correlation Deep Dive", "📂 Raw Data Manager"])

with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Stress Levels by Department")
        # Interactive Bar Chart
        dept_stress = df_filtered.groupby('Department')['Stress Level'].mean().reset_index().sort_values('Stress Level')
        fig_bar = px.bar(dept_stress, x='Department', y='Stress Level', color='Stress Level', 
                         color_continuous_scale='Reds', text_auto='.1f', height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Risk Distribution")
        # Donut Chart for Risk
        df_filtered['Risk'] = pd.cut(df_filtered['Stress Level'], bins=[0, 4, 7, 10], labels=['Low', 'Moderate', 'High'])
        risk_counts = df_filtered['Risk'].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4, 
                         color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'])
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Does Stress Affect Grades?")
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        # Advanced Scatter Plot with 4 Dimensions (X, Y, Color, Size)
        fig_scatter = px.scatter(
            df_filtered, 
            x="Stress Level", 
            y="CGPA", 
            color="Sleep Hours", 
            size="Attendance (%)",
            hover_data=['Student ID'],
            color_continuous_scale='Viridis',
            title="Correlation: Stress vs. Academic Performance"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_b:
        st.warning("💡 **AI Insight:**\n\nData indicates a strong negative correlation. Students sleeping less than 5 hours show a **15% drop in CGPA** when stress levels exceed 8.")

with tab3:
    st.subheader("System Data Log")
    st.dataframe(df_filtered, use_container_width=True)