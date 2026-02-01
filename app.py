import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from prophet import Prophet
from scipy import stats
import warnings
import configparser
import subprocess

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ==================== MODERN CSS STYLING ====================
st.markdown("""
<style>
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.main {
    padding: 1rem;
}

.card {
    border-radius: 12px;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 10px 0;
}

.metric-box {
    border-radius: 10px;
    padding: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.header-text {
    color: #1f3a93;
    font-weight: 700;
    font-size: 1.3rem;
    border-bottom: 3px solid #667eea;
    padding-bottom: 10px;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s ease;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

.stExpander {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

.footer {
    text-align: center;
    color: #666;
    font-size: 0.85rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "column_types" not in st.session_state:
    st.session_state.column_types = {}

# ==================== HELPER FUNCTIONS ====================

def detect_column_types(df):
    """Detect column types: numeric, categorical, date"""
    col_types = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            col_types[col] = 'numeric'
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='coerce')
                if pd.to_datetime(df[col], errors='coerce').notna().sum() / len(df) > 0.8:
                    col_types[col] = 'date'
                else:
                    col_types[col] = 'categorical'
            except:
                col_types[col] = 'categorical'
        else:
            col_types[col] = 'other'
    return col_types

def preprocess_data(df):
    """Preprocess data: handle missing values, convert dates"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        # Handle missing values
        if df_clean[col].isna().sum() > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col].fillna(mode_val[0], inplace=True)
                else:
                    df_clean[col].fillna('Unknown', inplace=True)
        
        # Try to convert date columns
        if df_clean[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df_clean[col], errors='coerce')
                if converted.notna().sum() / len(df_clean) > 0.8:
                    df_clean[col] = converted
            except:
                pass
    
    return df_clean

def get_numeric_columns(df):
    """Get numeric columns"""
    return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_columns(df):
    """Get categorical columns"""
    return df.select_dtypes(include=['object']).columns.tolist()

def get_date_columns(df):
    """Get date columns"""
    return df.select_dtypes(include=['datetime64']).columns.tolist()

def get_github_config():
    """Read GitHub token and repo from config.ini"""
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        token = config.get('GitHub', 'GitHubToken', fallback=None)
        repo = config.get('GitHub', 'GitHubRepo', fallback=None)
        return token, repo
    except:
        return None, None

def deploy_to_github():
    """Push to GitHub and trigger Netlify deployment"""
    try:
        token, repo = get_github_config()
        if not token or not repo:
            st.error("❌ GitHub config not found in config.ini")
            return False
        
        with st.spinner("🚀 Pushing to GitHub..."):
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Update dashboard - auto deploy"], 
                         check=True, capture_output=True)
            push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
            subprocess.run(["git", "push", push_url, "main", "--force"], 
                         check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Git error: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False

# ==================== TAB 1: PREVIEW & EDA ====================

def tab_preview_eda(df):
    """Data Preview & EDA Tab"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", len(df))
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("❌ Missing", df.isna().sum().sum())
    with col4:
        st.metric("🔄 Duplicates", df.duplicated().sum())
    
    st.divider()
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="header-text">📋 Data Preview (First 10 Rows)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        with st.expander("ℹ️ Dataset Info", expanded=True):
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            st.write("**Data Types:**")
            for col, dtype in df.dtypes.items():
                st.write(f"• {col}: {dtype}")
    
    st.divider()
    
    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="header-text">🔍 Missing Values</div>', unsafe_allow_html=True)
        missing = df.isnull().astype(int)
        if missing.sum().sum() > 0:
            fig = px.imshow(missing.T, color_continuous_scale="Reds", aspect="auto", height=400)
            fig.update_xaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values!")
    
    with col2:
        st.markdown('<div class="header-text">📈 Correlation Matrix</div>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr(numeric_only=True)
            fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ numeric columns for correlation")

# ==================== TAB 2: DESCRIPTIVE VISUALIZATION ====================

def tab_descriptive_viz(df):
    """Descriptive Visualization Tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox("📊 Chart Type", ["Histogram", "Boxplot", "Bar Chart", "Line Chart"])
    
    with col2:
        cols = st.multiselect("📍 Columns", df.columns, default=[df.columns[0]] if len(df.columns) > 0 else [])
    
    if not cols:
        st.warning("Select at least one column!")
        return
    
    if chart_type == "Histogram":
        st.markdown('<div class="header-text">📊 Histogram</div>', unsafe_allow_html=True)
        for col in cols:
            if df[col].dtype in ['float64', 'int64']:
                fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{col} is not numeric")
    
    elif chart_type == "Boxplot":
        st.markdown('<div class="header-text">📦 Boxplot</div>', unsafe_allow_html=True)
        for col in cols:
            if df[col].dtype in ['float64', 'int64']:
                fig = px.box(df, y=col, color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{col} is not numeric")
    
    elif chart_type == "Bar Chart":
        st.markdown('<div class="header-text">📊 Bar Chart</div>', unsafe_allow_html=True)
        for col in cols:
            if df[col].dtype == 'object':
                fig = px.bar(df[col].value_counts().head(20), color_discrete_sequence=['#06a77d'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{col} is numeric")
    
    elif chart_type == "Line Chart":
        st.markdown('<div class="header-text">📈 Line Chart</div>', unsafe_allow_html=True)
        numeric_cols = get_numeric_columns(df)
        for col in cols:
            if col in numeric_cols:
                fig = px.line(df, y=col, color_discrete_sequence=['#f77f00'])
                st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: REGRESSION ====================

def tab_regression(df):
    """Linear Regression Tab"""
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox("🎯 Target Variable", numeric_cols)
    
    with col2:
        features = st.multiselect("📍 Features", 
                                 [c for c in numeric_cols if c != target],
                                 default=[c for c in numeric_cols if c != target][:min(2, len(numeric_cols)-1)])
    
    if not features:
        st.warning("Select at least one feature!")
        return
    
    X = df[features].copy()
    y = df[target].copy()
    
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.error("❌ No valid data!")
        return
    
    with st.spinner("🔄 Training model..."):
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
    with col3:
        st.metric("RMSE", f"{rmse:.4f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📋 Coefficients", expanded=True):
            coef_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            st.write(f"**Intercept:** {model.intercept_:.4f}")
    
    with col2:
        st.markdown('<div class="header-text">📈 Actual vs Predicted</div>', unsafe_allow_html=True)
        fig = px.scatter(x=y, y=y_pred, trendline="ols",
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: CLUSTERING ====================

def tab_clustering(df):
    """K-Means Clustering Tab"""
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        features = st.multiselect("📍 Features", numeric_cols, 
                                 default=numeric_cols[:min(2, len(numeric_cols))])
    
    with col2:
        k = st.slider("🎯 Clusters", 2, 10, 3)
    
    if not features:
        st.warning("Select at least one feature!")
        return
    
    X = df[features].copy().dropna()
    
    if len(X) == 0:
        st.error("❌ No valid data!")
        return
    
    with st.spinner("🔄 Computing clusters..."):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow method
        inertias = []
        silhouettes = []
        for i in range(2, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            labels = km.fit_predict(X_scaled)
            silhouettes.append(silhouette_score(X_scaled, labels))
        
        # Final model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        X['Cluster'] = clusters
        silhouette_final = silhouette_score(X_scaled, clusters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="header-text">📊 Elbow Method</div>', unsafe_allow_html=True)
        fig = px.line(x=list(range(2, 11)), y=inertias, markers=True,
                     color_discrete_sequence=['#667eea'])
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="header-text">⭐ Silhouette Score</div>', unsafe_allow_html=True)
        fig = px.line(x=list(range(2, 11)), y=silhouettes, markers=True,
                     color_discrete_sequence=['#764ba2'])
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⭐ Silhouette", f"{silhouette_final:.4f}")
    with col2:
        st.metric("📊 Inertia", f"{kmeans.inertia_:.2f}")
    
    st.divider()
    
    st.markdown('<div class="header-text">🎨 Cluster Visualization</div>', unsafe_allow_html=True)
    
    if len(features) == 2:
        fig = px.scatter(x=X[features[0]], y=X[features[1]], 
                        color=X['Cluster'].astype(str),
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                        color=X['Cluster'].astype(str),
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: TIME SERIES ====================

def tab_time_series(df):
    """Time Series Forecasting Tab"""
    date_cols = get_date_columns(df)
    numeric_cols = get_numeric_columns(df)
    
    if len(date_cols) == 0:
        st.warning("⚠️ No date columns found!")
        return
    
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_col = st.selectbox("📅 Date Column", date_cols)
    
    with col2:
        target_col = st.selectbox("🎯 Target", numeric_cols)
    
    with col3:
        days = st.slider("⏱️ Forecast Days", 7, 365, 90, 7)
    
    ts_data = df[[date_col, target_col]].copy()
    ts_data.columns = ['ds', 'y']
    ts_data = ts_data.dropna().sort_values('ds')
    
    if len(ts_data) < 2:
        st.error("❌ Not enough data!")
        return
    
    try:
        with st.spinner("🔄 Training Prophet..."):
            model = Prophet(yearly_seasonality=True, monthly_seasonality=True, daily_seasonality=False)
            model.fit(ts_data)
            
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
        
        st.markdown('<div class="header-text">📈 Forecast</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Historical'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', 
                                name='95% CI', fillcolor='rgba(0,100,150,0.2)'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        with st.expander("📊 Components"):
            st.markdown("Trend & Seasonal Components:")
            fig_comp = model.plot_components(forecast)
            st.pyplot(fig_comp, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==================== TAB 6: COHORT ANALYSIS ====================

def tab_cohort_analysis(df):
    """Cohort Analysis Tab"""
    date_cols = get_date_columns(df)
    other_cols = [c for c in df.columns if c not in date_cols]
    
    if len(date_cols) == 0:
        st.warning("⚠️ No date columns!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_col = st.selectbox("👤 User/ID Column", other_cols)
    
    with col2:
        date_col = st.selectbox("📅 Date Column", date_cols)
    
    try:
        with st.spinner("🔄 Computing cohorts..."):
            cohort_data = df[[user_col, date_col]].copy()
            cohort_data = cohort_data.dropna()
            cohort_data.columns = ['user_id', 'date']
            
            cohort_data['cohort'] = cohort_data['date'].dt.to_period('M')
            cohort_data['user_date'] = cohort_data.groupby('user_id')['date'].transform('min')
            cohort_data['user_cohort'] = cohort_data['user_date'].dt.to_period('M')
            cohort_data['cohort_age'] = (cohort_data['cohort'] - cohort_data['user_cohort']).apply(
                lambda x: x.n if pd.notna(x) else None)
            
            cohort_pivot = cohort_data.groupby(['user_cohort', 'cohort_age']).agg(
                {'user_id': 'nunique'}).reset_index()
            cohort_table = cohort_pivot.pivot_table(index='user_cohort', columns='cohort_age', values='user_id')
            
            cohort_size = cohort_table.iloc[:, 0]
            retention_table = (cohort_table.divide(cohort_size, axis=0) * 100).round(1)
        
        st.markdown('<div class="header-text">👥 Retention Rate (%)</div>', unsafe_allow_html=True)
        
        fig = px.imshow(retention_table, color_continuous_scale="RdYlGn", 
                       text_auto=".1f", aspect="auto", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        with st.expander("📊 Retention Table"):
            st.dataframe(retention_table, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==================== TAB 7: HYPOTHESIS TESTING ====================

def tab_hypothesis_testing(df):
    """Hypothesis Testing Tab"""
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_type = st.selectbox("🔬 Test Type", ["T-Test", "Chi-Square"])
    
    with col2:
        alpha = st.slider("📊 Alpha (α)", 0.01, 0.10, 0.05, 0.01)
    
    try:
        if test_type == "T-Test":
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                st.error("❌ Need grouping and numeric columns!")
                return
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("📍 Group Column", categorical_cols)
            with col2:
                value_col = st.selectbox("📍 Value Column", numeric_cols)
            
            groups = df[group_col].unique()
            groups = groups[pd.notna(groups)]
            
            if len(groups) < 2:
                st.error("❌ Need at least 2 groups!")
                return
            
            col1, col2 = st.columns(2)
            with col1:
                g1 = st.selectbox("Group 1", groups)
            with col2:
                g2 = st.selectbox("Group 2", [g for g in groups if g != g1])
            
            data1 = df[df[group_col] == g1][value_col].dropna()
            data2 = df[df[group_col] == g2][value_col].dropna()
            
            if len(data1) > 0 and len(data2) > 0:
                with st.spinner("Running t-test..."):
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("T-Statistic", f"{t_stat:.4f}")
                with col2:
                    st.metric("P-Value", f"{p_value:.6f}")
                with col3:
                    result = "✅ Significant" if p_value < alpha else "❌ Not Sig."
                    st.metric("Result", result)
                
                st.divider()
                
                if p_value < alpha:
                    st.success(f"✅ **SIGNIFICANT** - p-value {p_value:.6f} < {alpha}")
                else:
                    st.warning(f"❌ **NOT SIGNIFICANT** - p-value {p_value:.6f} ≥ {alpha}")
        
        else:  # Chi-Square
            if len(categorical_cols) < 2:
                st.error("❌ Need at least 2 categorical columns!")
                return
            
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("📍 Variable 1", categorical_cols)
            with col2:
                var2 = st.selectbox("📍 Variable 2", [c for c in categorical_cols if c != var1])
            
            with st.spinner("Running chi-square test..."):
                contingency = pd.crosstab(df[var1], df[var2])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chi² Statistic", f"{chi2:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.6f}")
            with col3:
                result = "✅ Significant" if p_value < alpha else "❌ Not Sig."
                st.metric("Result", result)
            
            st.divider()
            
            if p_value < alpha:
                st.success(f"✅ **SIGNIFICANT** - p-value {p_value:.6f} < {alpha}")
            else:
                st.warning(f"❌ **NOT SIGNIFICANT** - p-value {p_value:.6f} ≥ {alpha}")
            
            st.divider()
            
            with st.expander("📊 Contingency Table"):
                st.dataframe(contingency, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("📊 Dashboard")
    
    with st.expander("⚙️ Settings", expanded=True):
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.original_df = pd.read_csv(uploaded_file)
            else:
                st.session_state.original_df = pd.read_excel(uploaded_file)
            st.success(f"✅ {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ {e}")
    
    st.divider()
    
    if st.session_state.original_df is not None:
        if st.button("🔄 Load & Preprocess", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                st.session_state.df = preprocess_data(st.session_state.original_df)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success("✅ Ready!")
        
        st.divider()
        
        if st.session_state.df is not None:
            with st.expander("📊 Data Info"):
                st.metric("Rows", st.session_state.df.shape[0])
                st.metric("Columns", st.session_state.df.shape[1])
            
            st.divider()
            
            csv = st.session_state.df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "data.csv", "text/csv", use_container_width=True)
    
    st.divider()
    
    # GitHub Deploy
    token, repo = get_github_config()
    if token and repo:
        st.caption(f"🔗 {repo}")
        if st.button("🚀 Push & Deploy", use_container_width=True, type="primary"):
            if deploy_to_github():
                st.success("✅ Pushed! Netlify deploying...")
                st.info("dataanalis.netlify.app")
            else:
                st.error("Deploy failed")

# ==================== MAIN AREA ====================

if st.session_state.df is None:
    st.title("📊 Data Analysis Dashboard")
    st.markdown("""
    **Getting Started:**
    1. Upload CSV or Excel file in sidebar
    2. Click "Load & Preprocess"
    3. Explore data using 7 tabs
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Preview & EDA**\nData overview, stats, heatmaps")
    with col2:
        st.info("**Visualize**\nCustom charts & plots")
    with col3:
        st.info("**Analyze**\nRegression, clustering, forecast")

else:
    st.title("📊 Data Analysis Dashboard")
    st.write(f"**{st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns**")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Preview & EDA",
        "📈 Descriptive Viz",
        "📊 Regression",
        "🎯 Clustering",
        "⏱️ Time Series",
        "👥 Cohort",
        "🔬 Tests"
    ])
    
    with tab1:
        tab_preview_eda(st.session_state.df)
    with tab2:
        tab_descriptive_viz(st.session_state.df)
    with tab3:
        tab_regression(st.session_state.df)
    with tab4:
        tab_clustering(st.session_state.df)
    with tab5:
        tab_time_series(st.session_state.df)
    with tab6:
        tab_cohort_analysis(st.session_state.df)
    with tab7:
        tab_hypothesis_testing(st.session_state.df)

# ==================== FOOTER ====================

st.markdown("""
<div class="footer">
<p><strong>📊 Data Analysis Dashboard</strong></p>
<p>Streamlit • Pandas • Plotly • Scikit-learn • Prophet • SciPy</p>
<p style="font-size: 0.75rem; margin-top: 10px;">© 2026 • Deploy-friendly • No external APIs</p>
</div>
""", unsafe_allow_html=True)

# ==================== DEPLOYMENT NOTES ====================
# To deploy: 
# 1. Set GitHub token & repo in config.ini
# 2. Click "Push & Deploy" button in sidebar
# 3. Or manually: git add . && git commit -m "msg" && git push
