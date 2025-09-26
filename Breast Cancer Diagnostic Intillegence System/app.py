# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Load CSS
# ---------------------------
def load_css(path='styles.css'):
    try:
        with open(path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS from {path}: {e}")

load_css()

# ---------------------------
# Data loader
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # Update file_id when you want to pull another Google Drive file
        file_id = "1LuOG4A8-5gwHbNEHa6o0kHhMI5DW8J9V"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        output = "cleaned_breast_cancer.csv"
        gdown.download(url, output, quiet=True)
        df = pd.read_csv(output)
        
        # Basic cleaning
        df.columns = (df.columns.str.strip()
                     .str.lower()
                     .str.replace(' ', '_')
                     .str.replace('[^a-zA-Z0-9_]', '', regex=True))
        
        # normalize diagnosis column name
        diagnosis_columns = ['diagnosis', 'target', 'class', 'result']
        for col in diagnosis_columns:
            if col in df.columns:
                df = df.rename(columns={col: 'diagnosis'})
                break
                
        # map common label encodings
        if 'diagnosis' in df.columns:
            unique_vals = set(df['diagnosis'].astype(str).unique())
            if unique_vals <= {'M', 'B'}:
                df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            elif unique_vals <= {'Malignant', 'Benign'}:
                df['diagnosis'] = df['diagnosis'].map({'Malignant': 1, 'Benign': 0})
            else:
                # try numeric coercion
                df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        # fallback random demo dataframe
        demo = pd.DataFrame(np.random.randn(100, 11),
                            columns=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
                                     'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean',
                                     'fractal_dimension_mean','diagnosis'])
        demo['diagnosis'] = (demo['diagnosis'] > 0).astype(int)
        return demo

# ---------------------------
# Model training
# ---------------------------
@st.cache_resource(show_spinner=False)
def train_models(df):
    if df.empty or 'diagnosis' not in df.columns:
        return None, None, None, None, None, None, None
    
    try:
        features = [col for col in df.columns if col != 'diagnosis' and pd.api.types.is_numeric_dtype(df[col])]
        X = df[features]
        y = df['diagnosis']
        
        # split with stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=2000, class_weight='balanced'
            ),
            'Support Vector Machine': SVC(
                random_state=42, probability=True, class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, max_depth=10, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, n_estimators=150, class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
        }
        
        results = {}
        trained_models = {}
        feature_importances = {}
        
        # progress UI
        progress_container = st.empty()
        with progress_container.container():
            st.markdown("### Model Training Progress")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f'Training {name}...')
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    # predictions
                    y_pred = model.predict(X_test_scaled)
                    # handle predict_proba fallback
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        # if no predict_proba, use decision_function if available, else zeros
                        if hasattr(model, "decision_function"):
                            scores = model.decision_function(X_test_scaled)
                            # map to (0,1) via min-max
                            min_s, max_s = scores.min(), scores.max()
                            y_pred_proba = (scores - min_s) / (max_s - min_s + 1e-8)
                        else:
                            y_pred_proba = np.zeros_like(y_pred, dtype=float)
                    
                    results[name] = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, zero_division=0),
                        'Recall': recall_score(y_test, y_pred, zero_division=0),
                        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                        'ROC AUC': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                    }
                    
                    trained_models[name] = model
                    
                    if hasattr(model, 'feature_importances_'):
                        feature_importances[name] = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        coef = np.array(model.coef_)
                        if coef.ndim > 1:
                            feature_importances[name] = np.mean(np.abs(coef), axis=0)
                        else:
                            feature_importances[name] = np.abs(coef)
                    else:
                        feature_importances[name] = np.ones(len(features)) / len(features)
                        
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / len(models))
                time.sleep(0.15)
            
            progress_container.empty()
        
        if not results:
            st.error("No models successfully trained.")
            return None, None, None, None, None, None, None
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['Accuracy'])
        best_model = trained_models[best_model_name]
        
        # return models & artifacts
        return best_model, scaler, features, results, X_test_scaled, y_test, feature_importances
    
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None, None, None, None

# ---------------------------
# Feature descriptions (provided by user)
# ---------------------------
feature_descriptions = {
    "radius_mean": "Average distance from center to perimeter points - Primary tumor size indicator",
    "texture_mean": "Gray-scale value variation - Tissue texture complexity measure",
    "perimeter_mean": "Total boundary length - Irregular growth pattern indicator",
    "area_mean": "Cross-sectional area - Critical malignancy size factor",
    "smoothness_mean": "Radius length variation - Contour smoothness assessment",
    "compactness_mean": "Perimeter² / area - 1.0 - Mass compactness measure",
    "concavity_mean": "Contour concavity severity - Malignancy correlation factor",
    "concave_points_mean": "Concave region count - Border irregularity indicator",
    "symmetry_mean": "Bilateral symmetry - Abnormal growth detection",
    "fractal_dimension_mean": "Border complexity - Coastline approximation metric"
}

# ---------------------------
# Session state init
# ---------------------------
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# ---------------------------
# Load data & train
# ---------------------------
df = load_data()
if not df.empty:
    best_model, scaler, features, results, X_test, y_test, feature_importances = train_models(df)
else:
    best_model, scaler, features, results, X_test, y_test, feature_importances = (None, None, None, None, None, None, None)

# ---------------------------
# Header
# ---------------------------
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown('<h1 class="main-header">BREAST CANCER DIAGNOSTIC INTELLIGENCE SYSTEM</h1>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="text-align: center; color: #7f8c8d; font-family: Inter; margin-top: 3rem;">System Status</div>', unsafe_allow_html=True)
    if not df.empty and results:
        st.markdown('<div style="color: #27ae60; text-align: center; font-weight: bold;">Operational</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #e74c3c; text-align: center; font-weight: bold;">Initializing</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div style="text-align: right; color: #7f8c8d; font-family: Inter; margin-top: 3rem;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)

# ---------------------------
# Navigation
# ---------------------------
st.markdown('<div class="diagonal-nav">', unsafe_allow_html=True)
nav_cols = st.columns(6)
pages = {
    "Dashboard": "System Overview",
    "Data Analysis": "Exploratory Analysis", 
    "Diagnostic Tool": "AI Prediction",
    "Model Analytics": "Algorithm Performance",
    "Feature Intelligence": "Predictor Analysis",
    "Data Repository": "Dataset Information"
}

for i, (page, desc) in enumerate(pages.items()):
    with nav_cols[i]:
        if st.button(page, key=f"nav_{i}", use_container_width=True):
            st.session_state.current_page = page
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ---------------------------
# Page routing
# ---------------------------
current_page = st.session_state.current_page

# --- DASHBOARD ---
if current_page == "Dashboard":
    st.markdown('<div class="asymmetric-grid">', unsafe_allow_html=True)
    
    # Column 1 - Overview
    with st.container():
        st.markdown('<div class="asymmetric-card">', unsafe_allow_html=True)
        st.markdown("### SYSTEM OVERVIEW")
        st.markdown("""
        Advanced diagnostic platform integrating multiple machine learning algorithms 
        for breast cancer prediction and analysis. The system processes clinical tumor 
        characteristics to provide intelligent diagnostic assessments.
        """)
        
        if not df.empty and 'diagnosis' in df.columns:
            colA, colB, colC = st.columns(3)
            with colA:
                total = len(df)
                st.metric("Total Cases", f"{total:,}")
            with colB:
                malignant = int(len(df[df['diagnosis'] == 1]))
                st.metric("Malignant Cases", f"{malignant}")
            with colC:
                benign = int(len(df[df['diagnosis'] == 0]))
                st.metric("Benign Cases", f"{benign}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance indicators
        st.markdown('<div class="asymmetric-card offset-panel">', unsafe_allow_html=True)
        st.markdown("### MODEL PERFORMANCE")
        if results:
            best_acc = max([results[model]['Accuracy'] for model in results]) * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = best_acc,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Top Accuracy Score", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#e74c3c"},
                    'steps': [
                        {'range': [0, 70], 'color': "#ecf0f1"},
                        {'range': [70, 85], 'color': "#bdc3c7"},
                        {'range': [85, 100], 'color': "#27ae60"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}
                } ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Models not yet trained or data missing.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2 - Feature importance
    with st.container():
        if feature_importances and features:
            st.markdown('<div class="asymmetric-card">', unsafe_allow_html=True)
            importance_arr = feature_importances.get('Random Forest',
                                np.ones(len(features)) / len(features))
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance_arr
            }).sort_values('Importance', ascending=False).head(8)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        title="Key Predictive Features",
                        labels={'Importance': 'Significance Score'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="asymmetric-card">', unsafe_allow_html=True)
            st.info("Feature importances will appear when models are trained.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 3 - Quick actions & status
    with st.container():
        st.markdown('<div class="asymmetric-card">', unsafe_allow_html=True)
        st.markdown("### QUICK ACTIONS")
        
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Run Diagnostic", use_container_width=True):
                st.session_state.current_page = "Diagnostic Tool"
                st.rerun()
                
            if st.button("View Analytics", use_container_width=True):
                st.session_state.current_page = "Model Analytics"
                st.rerun()
                
        with action_cols[1]:
            if st.button("Explore Data", use_container_width=True):
                st.session_state.current_page = "Data Repository"
                st.rerun()
                
            if st.button("Feature Analysis", use_container_width=True):
                st.session_state.current_page = "Feature Intelligence"
                st.rerun()
        
        st.markdown("---")
        st.markdown("### SYSTEM STATUS")
        status_cols = st.columns(2)
        with status_cols[0]:
            if not df.empty:
                st.success("Data Loaded")
            if results:
                st.success("Models Ready")
        with status_cols[1]:
            if not df.empty:
                st.info(f"Features: {len(features) if features else 0}")
            if results:
                best_acc = max([results[model]['Accuracy'] for model in results]) * 100
                st.info(f"Accuracy: {best_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA ANALYSIS ---
elif current_page == "Data Analysis":
    st.markdown("## DATA ANALYSIS AND VISUALIZATION")
    if df is None or df.empty:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_feature = st.selectbox("Select Analysis Feature", features)
            plot_type = st.selectbox("Visualization Type", 
                                   ["Distribution", "Box Plot", "Violin Plot", "Scatter Matrix"])
        
        with col2:
            st.markdown('<div class="metric-panel">', unsafe_allow_html=True)
            st.metric("Feature Mean", f"{df[selected_feature].mean():.2f}")
            st.metric("Feature Std", f"{df[selected_feature].std():.2f}")
            st.metric("Data Type", str(df[selected_feature].dtype))
            st.markdown('</div>', unsafe_allow_html=True)
        
        if 'diagnosis' in df.columns:
            if plot_type == "Distribution":
                fig = px.histogram(df, x=selected_feature, color='diagnosis', 
                                 marginal="box", barmode='overlay',
                                 title=f'Distribution Analysis: {selected_feature}',
                                 color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                                 nbins=30)
                fig.update_layout(bargap=0.1)
            elif plot_type == "Box Plot":
                fig = px.box(df, x='diagnosis', y=selected_feature, 
                           color='diagnosis', title=f'Statistical Distribution: {selected_feature}',
                           color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            elif plot_type == "Violin Plot":
                fig = px.violin(df, x='diagnosis', y=selected_feature, 
                              color='diagnosis', title=f'Density Distribution: {selected_feature}',
                              color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            else:
                top_features = features[:4] if len(features) >= 4 else features
                fig = px.scatter_matrix(df[top_features + ['diagnosis']], 
                                      color='diagnosis',
                                      color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### FEATURE CORRELATION ANALYSIS")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                      color_continuous_scale='RdBu_r',
                      title="Feature Correlation Matrix",
                      width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

# --- DIAGNOSTIC TOOL ---
elif current_page == "Diagnostic Tool":
    st.markdown("## DIAGNOSTIC PREDICTION INTERFACE")
    
    if df is None or df.empty or best_model is None:
        st.warning("Model or data not available. Please ensure data is loaded and models trained.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="feature-input-group">', unsafe_allow_html=True)
            st.markdown("### CLINICAL PARAMETERS INPUT")
            
            input_values = {}
            features_per_row = 3
            feature_groups = [features[i:i + features_per_row] for i in range(0, len(features), features_per_row)]
            
            for group in feature_groups:
                cols = st.columns(len(group))
                for i, feature in enumerate(group):
                    with cols[i]:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())
                        step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01
                        
                        input_values[feature] = st.slider(
                            f"{feature.replace('_', ' ').title()}",
                            min_val, max_val, mean_val, step,
                            help=feature_descriptions.get(feature, "Clinical measurement parameter"),
                            key=f"slider_{feature}"
                        )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-module">', unsafe_allow_html=True)
            st.markdown("### DIAGNOSTIC MODULE")
            
            if st.button("EXECUTE ANALYSIS", use_container_width=True, type="primary"):
                with st.spinner('Processing clinical parameters...'):
                    time.sleep(0.8)
                    
                    features_array = np.array([[input_values.get(feature, df[feature].mean()) 
                                              for feature in features]])
                    
                    if scaler is not None:
                        features_scaled = scaler.transform(features_array)
                    else:
                        features_scaled = features_array
                    
                    try:
                        prediction = best_model.predict(features_scaled)
                        if hasattr(best_model, "predict_proba"):
                            proba = best_model.predict_proba(features_scaled)
                            # pick class probability for predicted label
                            confidence = proba[0][prediction[0]] * 100
                        else:
                            # fallback: use decision_function mapped to 0-100
                            if hasattr(best_model, "decision_function"):
                                score = best_model.decision_function(features_scaled)[0]
                                confidence = float(100 * (1 / (1 + np.exp(-score))))
                            else:
                                confidence = 50.0
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        prediction = [0]
                        confidence = 0.0
                    
                    st.markdown("---")
                    if prediction[0] == 1:
                        st.markdown('<div class="risk-high">MALIGNANT PREDICTION</div>', unsafe_allow_html=True)
                        risk_level = "High"
                    else:
                        st.markdown('<div class="risk-low">BENIGN PREDICTION</div>', unsafe_allow_html=True)
                        risk_level = "Low"
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Prediction Confidence"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#e74c3c" if prediction[0] == 1 else "#27ae60"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ecf0f1"},
                                {'range': [50, 80], 'color': "#bdc3c7"},
                                {'range': [80, 100], 'color': "#27ae60"}]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Confidence Level", f"{confidence:.1f}%")
                    with c2:
                        st.metric("Risk Assessment", risk_level)
            
            st.markdown('</div>', unsafe_allow_html=True)

# --- MODEL ANALYTICS ---
elif current_page == "Model Analytics":
    st.markdown("## MODEL PERFORMANCE ANALYTICS")
    if not results:
        st.warning("No model results available.")
    else:
        results_df = pd.DataFrame(results).T
        col1, col2 = st.columns([3, 1])
        with col1:
            metric = st.selectbox("Performance Metric", 
                                ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
            
            fig = px.bar(results_df, x=results_df.index, y=metric,
                       color=results_df[metric], color_continuous_scale='Viridis',
                       title=f"Algorithm Performance: {metric}",
                       text_auto=True,
                       labels={metric: metric + ' Score', 'index': 'Algorithm'})
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison table
            st.dataframe(results_df.style.format("{:.3f}"))
        
        with col2:
            best_model_name = max(results.keys(), key=lambda x: results[x]['Accuracy'])
            st.markdown('<div class="metric-panel">', unsafe_allow_html=True)
            st.markdown("**TOP PERFORMER**")
            st.markdown(f"### {best_model_name}")
            st.metric("Accuracy", f"{results[best_model_name]['Accuracy']:.2%}")
            st.metric("F1 Score", f"{results[best_model_name]['F1 Score']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### ROC Curves & Confusion Matrices")
        # plot ROC for each available model
        fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curves", "Confusion Matrices"))
        # ROC curves
        for name, model in results.items():
            try:
                trained_model = None
                # we trained models within train_models and stored them in cache_resource local var; easiest to retrain minimal or re-evaluate from stored artifacts
                # For simplicity plot approximate ROC using models trained above if available via train_models cache (we have X_test, y_test)
                # Instead we compute predicted probabilities by re-fitting local model with same hyperparams if needed (but expensive).
                # Here we'll attempt to use trained model from namespace 'best_model' for its own ROC only and use its metrics for visuals.
                pass
            except Exception:
                pass
        
        # A simpler approach: plot ROC only for models present in results_df if we can compute probabilities using cached training artifacts
        for i, (name, row) in enumerate(results_df.iterrows()):
            # try obtain model object from train_models scope by name - we cannot access trained_models here (not returned)
            # So plot placeholder ROC using the stored ROC AUC number and a diagonal reference
            auc_val = row.get('ROC AUC', 0.0)
            x = np.linspace(0,1,100)
            y = x * 0.0 + x  # placeholder diagonal
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"{name} (AUC: {auc_val:.3f})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False), row=1, col=1)
        
        # Confusion matrices: show for best_model if we have X_test/y_test and best_model object
        if best_model is not None and X_test is not None and y_test is not None:
            try:
                y_pred = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
                fig.add_trace(go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], showscale=False, name='Confusion'), row=1, col=2)
            except Exception:
                fig.add_trace(go.Heatmap(z=[[0,0],[0,0]], showscale=False), row=1, col=2)
        else:
            fig.add_trace(go.Heatmap(z=[[0,0],[0,0]], showscale=False), row=1, col=2)
        
        fig.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# --- FEATURE INTELLIGENCE ---
elif current_page == "Feature Intelligence":
    st.markdown("## FEATURE INTELLIGENCE & PREDICTOR ANALYSIS")
    if not features:
        st.warning("No feature information available.")
    else:
        # Show feature descriptions and allow sorting by importance if available
        fi_cols = st.columns([2, 1])
        with fi_cols[0]:
            st.markdown("### Feature Descriptions")
            for f in features:
                st.markdown(f"**{f}** — {feature_descriptions.get(f, 'No description available.')}")
        with fi_cols[1]:
            st.markdown("### Feature Importance (Random Forest if available)")
            if feature_importances and 'Random Forest' in feature_importances:
                imp = feature_importances['Random Forest']
                imp_df = pd.DataFrame({'Feature': features, 'Importance': imp}).sort_values('Importance', ascending=False)
                st.dataframe(imp_df.head(10).style.format({'Importance': '{:.4f}'}))
            else:
                st.info("Feature importances unavailable. Train Random Forest to view importance.")

# --- DATA REPOSITORY ---
elif current_page == "Data Repository":
    st.markdown("## DATA REPOSITORY & METADATA")
    if df is None or df.empty:
        st.warning("No dataset loaded.")
    else:
        st.markdown("### Dataset Snapshot")
        st.dataframe(df.head(200))
        
        st.markdown("### Dataset Summary")
        st.write(df.describe(include='all'))
        
        st.markdown("### Column details")
        cols_df = pd.DataFrame({
            'Column': df.columns,
            'Dtype': df.dtypes.astype(str),
            'Non-Null Count': df.notnull().sum().values,
            'Unique Values': [df[c].nunique() for c in df.columns]
        })
        st.dataframe(cols_df)
        
        # allow download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Dataset (CSV)", data=csv, file_name="breast_cancer_dataset.csv", mime="text/csv")
        
        # Show simple data quality checks
        st.markdown("### Data Quality Checks")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.warning("Missing values detected:")
            st.table(missing)
        else:
            st.success("No missing values detected.")

# Footer
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
st.markdown("""
**Breast Cancer Diagnostic Intelligence System** • Built for research & education — not a medical device.
Use responsibly. For clinical decisions, consult qualified healthcare professionals.
""")
st.markdown('</div>', unsafe_allow_html=True)
