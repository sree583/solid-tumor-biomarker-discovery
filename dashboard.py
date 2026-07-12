import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import SurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

st.set_page_config(page_title="Multi-Omics Clinical AI", page_icon="🧬", layout="wide")
st.title("🧬 Cloud-Native Multi-Omics Integrative Pipeline")
st.markdown("### Early-Stage Solid Tumor Biomarker Discovery & Survival Prediction Engine")

# Sidebar
st.sidebar.header("Pipeline Configuration")
sample_size = st.sidebar.slider("Number of Patient Samples", 50, 500, 150, 50)
gene_count = st.sidebar.slider("Number of Genomic Features", 500, 10000, 5000, 500)
run_pipeline = st.sidebar.button("⚡ Run Pipeline Engine", type="primary")

def generate_data(n_samples, n_genes):
    np.random.seed(42)
    patient_ids = [f"TCGA-PDAC-{i:04d}" for i in range(n_samples)]
    rna_data = np.random.lognormal(mean=4.0, sigma=1.0, size=(n_samples, n_genes))
    gene_names = [f"ENSG_Gene_{i:04d}" for i in range(n_genes)]
    rna_df = pd.DataFrame(rna_data, index=patient_ids, columns=gene_names)
    
    signal_effect = (rna_data[:, 0] * 2.0) + (rna_data[:, 1] * -1.8) + (rna_data[:, 2] * 1.5)
    base_hazard = np.random.exponential(scale=1500, size=n_samples)
    survival_days = np.clip(base_hazard / np.exp(signal_effect * 0.1), 90, 3650)
    status = np.random.choice([True, False], size=n_samples, p=[0.75, 0.25])
    
    clinical_df = pd.DataFrame({'Status': status, 'Survival_Time': survival_days}, index=patient_ids)
    return rna_df, clinical_df

if run_pipeline:
    with st.spinner("Processing multi-omics integration and survival training..."):
        rna_data, clinical_data = generate_data(sample_size, gene_count)
        integrated = rna_data.join(clinical_data[['Status', 'Survival_Time']], how='inner')
        
        y = SurvivalAnalysis.structured_array(integrated['Status'].astype(bool), integrated['Survival_Time'].astype(float))
        X = integrated.drop(columns=['Status', 'Survival_Time'])
        
        # Scale and feature selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pseudo_target = np.array([t * float(s) for s, t in y])
        
        selector = ElasticNetCV(cv=5, random_state=42, n_jobs=-1)
        selector.fit(X_scaled, pseudo_target)
        selected_idx = np.where(selector.coef_ != 0)[0]
        if len(selected_idx) == 0:
            selected_idx = np.argsort(np.abs(selector.coef_))[-10:]
            
        X_pruned = X_scaled[:, selected_idx]
        
        # Model
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(X_pruned, y)
        c_index = model.score(X_pruned, y)
        
        # Stratification
        risk_scores = model.predict(X_pruned)
        median_risk = np.median(risk_scores)
        high_risk = risk_scores > median_risk
        low_risk = ~high_risk
        
        st.success("🎉 Pipeline Executed Successfully!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Concordance Index (C-Index)", f"{c_index:.4f}")
        col2.metric("Discovered Biomarkers", f"{len(selected_idx)} Genes")
        col3.metric("Total Patients Monitored", f"{len(risk_scores)}")
        
        st.markdown("---")
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.subheader("📊 Patient Survival Stratification (Kaplan-Meier)")
            fig, ax = plt.subplots(figsize=(10, 6))
            t_h, p_h = kaplan_meier_estimator(y["Status"][high_risk], y["Survival_Time"][high_risk])
            ax.step(t_h, p_h, where="post", label="High-Risk Patient Cohort", color="#D32F2F", linewidth=2.5)
            t_l, p_l = kaplan_meier_estimator(y["Status"][low_risk], y["Survival_Time"][low_risk])
            ax.step(t_l, p_l, where="post", label="Low-Risk Patient Cohort", color="#1976D2", linewidth=2.5)
            ax.set_xlabel("Days Post Early-Stage Diagnosis")
            ax.set_ylabel("Probability of Survival")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        with right_col:
            st.subheader("🧬 Top Novel Diagnostic Biomarkers")
            biomarker_summary = pd.DataFrame({
                'Biomarker ID': [rna_data.columns[i] for i in selected_idx],
                'Model Weight Contribution': model.feature_importances_
            }).sort_values(by='Model Weight Contribution', ascending=False)
            st.dataframe(biomarker_summary.head(10), use_container_width=True)
else:
    st.info("💡 Adjust parameters in the sidebar panel and click 'Run Pipeline Engine' to execute.")
