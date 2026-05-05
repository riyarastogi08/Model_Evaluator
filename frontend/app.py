"""
Streamlit Frontend — Automated Model Robustness & Fairness Evaluator
Pages: New Evaluation | History | Compare
"""
import io
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

import os

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="AI Robustness Evaluator", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .health-grade {
        font-size: 3.5rem; font-weight: 800; text-align: center;
        padding: 10px 30px; border-radius: 16px; display: inline-block;
        line-height: 1.2; margin: 6px auto;
    }
    .grade-A { background: linear-gradient(135deg, #00c853, #69f0ae); color: #1b5e20; }
    .grade-B { background: linear-gradient(135deg, #2196f3, #64b5f6); color: #0d47a1; }
    .grade-C { background: linear-gradient(135deg, #ff9800, #ffcc02); color: #e65100; }
    .grade-D { background: linear-gradient(135deg, #ff5722, #ff8a65); color: #bf360c; }
    .grade-F { background: linear-gradient(135deg, #f44336, #ef5350); color: #fff; }
    .severity-badge {
        display: inline-block; padding: 2px 10px; border-radius: 8px;
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    }
    .sev-critical { background: #f443361a; color: #f44336; border: 1px solid #f44336; }
    .sev-warning  { background: #ff98001a; color: #ff9800; border: 1px solid #ff9800; }
    .sev-info     { background: #4caf501a; color: #4caf50; border: 1px solid #4caf50; }
    .conf-metric  { background: #1e1e2e; border-radius: 10px; padding: 12px 16px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Automated Model Robustness & Fairness Evaluator")
st.markdown("Evaluate ML models for **robustness**, **fairness**, **drift**, and **explainability** — powered by the RICE framework.")

with st.sidebar:
    page = st.radio("Navigation", ["🔬 New Evaluation", "📜 History", "🔀 Compare"], index=0)

# ═══════════════════════════════════════════════════════
#  PAGE: New Evaluation
# ═══════════════════════════════════════════════════════
if page == "🔬 New Evaluation":
    with st.sidebar:
        st.header("Upload & Configure")
        model_file = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
        dataset_file = st.file_uploader("Upload Dataset (.csv, .json)", type=["csv", "json"])

        # ── Dataset Preview & Column Picker ──
        target_column = ""
        sensitive_attr = ""
        col_options = []

        if dataset_file:
            try:
                dataset_file.seek(0)
                if dataset_file.name.endswith(".csv"):
                    preview_df = pd.read_csv(dataset_file)
                else:
                    preview_df = pd.read_json(dataset_file)
                col_options = list(preview_df.columns)

                with st.expander("📋 Dataset Preview", expanded=False):
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    st.caption(f"{preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")

                dataset_file.seek(0)
            except Exception as e:
                st.warning(f"Could not preview dataset: {e}")
                preview_df = None

        st.subheader("Model Settings")
        task_type = st.selectbox("Task Type", ["Classification", "Regression"])

        if col_options:
            target_column = st.selectbox(
                "Target Column",
                options=col_options,
                index=len(col_options) - 1,
                help="The column your model is predicting. Defaults to the last column.",
            )
            sensitive_attr = st.selectbox(
                "Sensitive Attribute (for fairness)",
                options=["— none —"] + col_options,
                index=0,
                help="Demographic column to test for bias (e.g. gender, race, age_group).",
            )
            if sensitive_attr == "— none —":
                sensitive_attr = ""
        else:
            target_column = st.text_input(
                "Target Column",
                placeholder="e.g. Attrition",
                help="Leave blank to use the last column automatically.",
            )
            sensitive_attr = st.text_input(
                "Sensitive Attribute Column",
                placeholder="e.g. Gender",
            )

        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            if model_file and dataset_file:
                with st.spinner("Submitting evaluation task..."):
                    dataset_file.seek(0)
                    files = {
                        "model_file": (model_file.name, model_file, "application/octet-stream"),
                        "dataset_file": (
                            dataset_file.name, dataset_file,
                            "text/csv" if dataset_file.name.endswith(".csv") else "application/json"
                        ),
                    }
                    data = {
                        "model_name": model_file.name,
                        "dataset_name": dataset_file.name,
                        "task_type": task_type,
                        "target_column": target_column,
                        "sensitive_attr": sensitive_attr,
                    }
                    try:
                        response = requests.post(f"{API_BASE_URL}/evaluate/", files=files, data=data)
                        response.raise_for_status()
                        st.session_state["task_id"] = response.json()["id"]
                        st.session_state["eval_status"] = "pending"
                        st.session_state["task_data"] = None
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Connection Error: Ensure the backend is running. {e}")
            else:
                st.warning("Please upload both a model and a dataset.")

    # ── Poll Status ──
    if "task_id" in st.session_state and st.session_state.get("eval_status") in ["pending", "evaluating"]:
        task_id = st.session_state["task_id"]
        progress_bar = st.progress(0, text="⏳ Waiting for pipeline to start...")
        poll_count = 0
        stage_map = {"pending": 5, "evaluating": 40, "completed": 100, "failed": 100}

        while True:
            try:
                res = requests.get(f"{API_BASE_URL}/status/{task_id}")
                res.raise_for_status()
                task_data = res.json()
                status = task_data["status"]
                pct = min(stage_map.get(status, 20) + poll_count * 3, 95) if status == "evaluating" else stage_map.get(status, 10)
                progress_bar.progress(min(pct, 100), text=f"Pipeline status: **{status.upper()}**")
                if status in ["completed", "failed"]:
                    progress_bar.progress(100, text=f"Pipeline **{status.upper()}**")
                    st.session_state["eval_status"] = status
                    st.session_state["task_data"] = task_data
                    break
                poll_count += 1
                time.sleep(2)
            except Exception as e:
                st.error(f"Polling error: {e}")
                break

    # ── Dashboard ──
    if st.session_state.get("eval_status") == "completed":
        data = st.session_state["task_data"]
        robustness = data.get("robustness_details") or {}
        fairness = data.get("fairness_details") or {}
        explainability = data.get("explainability_details") or {}
        drift = data.get("data_drift_details") or {}
        rice_table = data.get("rice_priority_table") or []

        # ── Health Grade ──
        report_text = robustness.get("llm_audit_report", "")
        grade, composite_str = "?", ""
        for line in report_text.split("\n"):
            if "Overall Health Grade:" in line:
                parts = line.split("Grade:")
                if len(parts) > 1:
                    rest = parts[1].strip()
                    grade = rest.split("**")[0].strip().rstrip("*").strip()
                    if "Composite Score:" in rest:
                        composite_str = rest.split("Composite Score:")[1].replace(")", "").replace("*", "").strip()
                break

        st.success("✅ Evaluation Completed Successfully!")
        if data.get("duration_seconds"):
            st.caption(f"⏱ Completed in {data['duration_seconds']:.1f}s  |  Target column: `{data.get('target_column', 'N/A')}`")
        st.markdown("---")

        # ── Top Metrics ──
        gcol, col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1, 1])
        with gcol:
            grade_class = f"grade-{grade}" if grade in "ABCDF" else "grade-C"
            st.markdown(f'<div class="health-grade {grade_class}">{grade}</div>', unsafe_allow_html=True)
            if composite_str:
                st.caption(f"Composite: {composite_str}")

        robustness_score = data.get("robustness_score")
        col1.metric("Stability (Noise)", f"{robustness_score:.1f}%" if robustness_score is not None else "N/A",
                    f"Drop: {robustness.get('accuracy_drop', 0):.3f}")
        adv_score = robustness.get("adversarial_resilience")
        col2.metric("Adversarial Resilience", f"{adv_score:.1f}%" if adv_score is not None else "N/A",
                    f"Drop: {robustness.get('adversarial_drop', 0):.3f}" if adv_score is not None else "")
        fairness_score = data.get("fairness_score")
        col3.metric("Fairness Score", f"{fairness_score:.1f}%" if fairness_score is not None else "N/A",
                    f"DP Diff: {fairness.get('demographic_parity_diff', 0):.3f}" if fairness_score is not None else "")
        boundary_r = robustness.get("boundary_resilience")
        col4.metric("Boundary Resilience", f"{boundary_r:.1f}%" if boundary_r is not None else "N/A",
                    f"Drop: {robustness.get('boundary_drop', 0):.3f}" if boundary_r is not None else "")

        st.markdown("---")

        # ── Confidence Analysis (new) ──
        conf = robustness.get("confidence_analysis", {})
        if conf.get("available"):
            st.subheader("🎯 Prediction Confidence")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Mean Confidence", f"{conf.get('mean_confidence', 0)*100:.1f}%")
            cc2.metric("Min Confidence", f"{conf.get('min_confidence', 0)*100:.1f}%")
            low_frac = conf.get("low_confidence_fraction", 0)
            cc3.metric(
                f"Low-Confidence Predictions (<{conf.get('low_confidence_threshold',0.6)*100:.0f}%)",
                f"{low_frac*100:.1f}%",
                delta=None,
                delta_color="inverse" if low_frac > 0.3 else "normal",
            )
            if low_frac > 0.3:
                st.warning(f"⚠️ {low_frac*100:.1f}% of predictions are below the confidence threshold. Consider probability calibration.")
            st.markdown("---")

        # ── RICE Priority Table ──
        st.subheader("🌾 RICE Priority Action Items")
        if rice_table:
            df_rice = pd.DataFrame(rice_table)
            if "severity" in df_rice.columns:
                def _sev_badge(val):
                    cls = {"critical": "sev-critical", "warning": "sev-warning", "info": "sev-info"}.get(val, "sev-info")
                    return f'<span class="severity-badge {cls}">{val}</span>'
                df_display = df_rice[["title", "severity", "description", "remediation", "score"]].copy()
                df_display["severity"] = df_display["severity"].apply(_sev_badge)
                st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(df_rice, use_container_width=True)
        else:
            st.info("No critical issues found.")
        st.markdown("---")

        # ── SHAP + Audit Report ──
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.subheader("💡 Feature Importance (SHAP)")
            shap_vals = explainability if explainability else robustness.get("feature_importance", {})
            if shap_vals and "error" not in shap_vals:
                df_shap = pd.DataFrame(list(shap_vals.items()), columns=["Feature", "Importance"])
                fig = px.bar(df_shap, x="Importance", y="Feature", orientation="h",
                             title="Global SHAP Values", color="Importance",
                             color_continuous_scale="Tealgrn")
                fig.update_layout(yaxis_categoryorder="total ascending", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("SHAP analysis failed or is unsupported for this model type.")

        with r2c2:
            st.subheader("📝 AI Audit Report")
            if report_text:
                st.markdown(report_text)
            else:
                st.info("No report generated.")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                if report_text:
                    st.download_button("📄 Export (.md)", report_text, file_name="audit_report.md", mime="text/markdown")
            with exp_c2:
                if report_text:
                    try:
                        pdf_bytes = _generate_pdf(report_text)
                        st.download_button("📕 Export (.pdf)", pdf_bytes, file_name="audit_report.pdf", mime="application/pdf")
                    except Exception:
                        st.caption("PDF export unavailable (install fpdf2)")
        st.markdown("---")

        # ── Drift Visualization (PSI + KS) ──
        if drift and drift.get("feature_psi"):
            st.subheader("📈 Data Drift Analysis (PSI + KS Test)")
            psi_data = drift["feature_psi"]
            ks_data = drift.get("feature_ks", {})
            threshold = drift.get("threshold", 0.2)
            df_drift = pd.DataFrame(list(psi_data.items()), columns=["Feature", "PSI"])
            df_drift = df_drift.sort_values("PSI", ascending=True)

            fig_drift = px.bar(df_drift, x="PSI", y="Feature", orientation="h",
                               color="PSI", color_continuous_scale=["#4caf50", "#ff9800", "#f44336"],
                               title="Population Stability Index per Feature")
            fig_drift.add_vline(x=threshold, line_dash="dash", line_color="red",
                                annotation_text=f"PSI Threshold ({threshold})")
            fig_drift.update_layout(showlegend=False)
            st.plotly_chart(fig_drift, use_container_width=True)

            # KS table
            if ks_data:
                df_ks = pd.DataFrame([
                    {"Feature": k, "KS Stat": v["statistic"], "p-value": v["p_value"],
                     "Drifted": "⚠️ Yes" if v["drifted"] else "✅ No"}
                    for k, v in ks_data.items()
                ])
                with st.expander("🔬 KS Test Results per Feature"):
                    st.dataframe(df_ks, use_container_width=True)

            sev = drift.get("severity", "low").upper()
            high_feats = drift.get("high_drift_features", [])
            if high_feats:
                st.warning(f"**Drift Severity: {sev}** — High-drift features (PSI+KS): {', '.join(high_feats)}")
            else:
                st.success(f"**Drift Severity: {sev}** — No features exceed both drift thresholds.")
        st.markdown("---")

        # ── CV Stability + Feature Ablation ──
        cv = robustness.get("cv_stability", {})
        ablation = robustness.get("feature_ablation", {})
        cv_col, abl_col = st.columns(2)

        with cv_col:
            st.subheader("📊 Cross-Validation Stability")
            if cv and "fold_scores" in cv:
                fold_scores = cv["fold_scores"]
                df_cv = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(fold_scores))], "Score": fold_scores})
                fig_cv = px.bar(df_cv, x="Fold", y="Score",
                                title=f"{cv.get('folds','?')}-Fold CV (Mean: {cv.get('mean',0):.4f} ± {cv.get('std',0):.4f})")
                fig_cv.add_hline(y=cv.get("mean", 0), line_dash="dash", line_color="orange", annotation_text="Mean")
                fig_cv.update_layout(yaxis_range=[max(0, min(fold_scores) - 0.1), 1.0])
                st.plotly_chart(fig_cv, use_container_width=True)
            elif cv and "error" in cv:
                st.warning(f"CV analysis failed: {cv['error']}")
            else:
                st.info("Cross-validation data not available.")

        with abl_col:
            st.subheader("🔧 Feature Ablation Impact")
            if ablation and "feature_impact" in ablation:
                impacts = ablation["feature_impact"]
                df_abl = pd.DataFrame(list(impacts.items()), columns=["Feature", "Score Drop"])
                df_abl = df_abl.sort_values("Score Drop", ascending=True)
                fig_abl = px.bar(df_abl, x="Score Drop", y="Feature", orientation="h",
                                 title="Performance Drop When Feature Removed",
                                 color="Score Drop", color_continuous_scale="RdYlGn_r")
                fig_abl.update_layout(showlegend=False)
                st.plotly_chart(fig_abl, use_container_width=True)
            elif ablation and "error" in ablation:
                st.warning(f"Ablation analysis failed: {ablation['error']}")
            else:
                st.info("Feature ablation data not available.")

        st.markdown("---")

        # ── Dataset Demographics ──
        st.subheader("📊 Dataset Demographics Distribution")
        if dataset_file is not None and dataset_file.name.endswith(".csv"):
            try:
                dataset_file.seek(0)
                df_viz = pd.read_csv(dataset_file)
                target_col_name = data.get("target_column", df_viz.columns[-1])
                sensitive_col = data.get("sensitive_attr") or ""
                if sensitive_col and sensitive_col in df_viz.columns and target_col_name in df_viz.columns:
                    fig_dist = px.histogram(df_viz, x=sensitive_col, color=target_col_name,
                                            barmode="group",
                                            title=f"Distribution of '{target_col_name}' by '{sensitive_col}'",
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("Provide a valid sensitive attribute to plot group analytics.")
            except Exception:
                pass

    elif st.session_state.get("eval_status") == "failed":
        st.error("❌ Evaluation Failed")
        st.json(st.session_state.get("task_data", {}))


# ═══════════════════════════════════════════════════════
#  PAGE: History
# ═══════════════════════════════════════════════════════
elif page == "📜 History":
    st.subheader("📜 Evaluation History")
    try:
        res = requests.get(f"{API_BASE_URL}/evaluations/")
        res.raise_for_status()
        evaluations = res.json()
    except Exception as e:
        st.error(f"Could not load evaluation history: {e}")
        evaluations = []

    if not evaluations:
        st.info("No evaluations found yet. Run your first evaluation from the sidebar!")
    else:
        for ev in evaluations:
            status_icon = {"completed": "✅", "failed": "❌", "evaluating": "⏳", "pending": "🕐"}.get(ev["status"], "❓")
            label = (f"{status_icon} **{ev['model_name']}** vs {ev['dataset_name']} — "
                     f"{ev['status'].upper()} ({ev['created_at'][:10]})")
            with st.expander(label):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Task Type", ev["task_type"])
                mc2.metric("Robustness", f"{ev.get('robustness_score', 0) or 0:.1f}%")
                mc3.metric("Fairness", f"{ev.get('fairness_score', 0) or 0:.1f}%")
                mc4.metric("Duration", f"{ev.get('duration_seconds', 0) or 0:.1f}s")
                if ev.get("target_column"):
                    st.caption(f"Target column: `{ev['target_column']}`")

                bc1, bc2 = st.columns(2)
                with bc1:
                    if ev["status"] == "completed":
                        if st.button("🔍 View Full Results", key=f"view_{ev['id']}"):
                            try:
                                detail_res = requests.get(f"{API_BASE_URL}/status/{ev['id']}")
                                detail_res.raise_for_status()
                                st.session_state["task_id"] = ev["id"]
                                st.session_state["eval_status"] = "completed"
                                st.session_state["task_data"] = detail_res.json()
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Could not load details: {ex}")
                with bc2:
                    if st.button("🗑 Delete", key=f"del_{ev['id']}"):
                        try:
                            del_res = requests.delete(f"{API_BASE_URL}/evaluations/{ev['id']}")
                            del_res.raise_for_status()
                            st.success(f"Evaluation {ev['id']} deleted.")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Delete failed: {ex}")


# ═══════════════════════════════════════════════════════
#  PAGE: Compare
# ═══════════════════════════════════════════════════════
elif page == "🔀 Compare":
    st.subheader("🔀 Model Comparison")
    try:
        res = requests.get(f"{API_BASE_URL}/evaluations/")
        res.raise_for_status()
        evaluations = res.json()
        completed = [e for e in evaluations if e["status"] == "completed"]
    except Exception as e:
        st.error(f"Could not load evaluations: {e}")
        completed = []

    if len(completed) < 2:
        st.info("You need at least 2 completed evaluations to compare. Run more evaluations first!")
    else:
        options = {f"#{e['id']} — {e['model_name']} ({e['created_at'][:10]})": e["id"] for e in completed}
        selected = st.multiselect("Select evaluations to compare (2–5):", options.keys(), max_selections=5)

        if len(selected) >= 2 and st.button("📊 Compare Now", type="primary"):
            selected_ids = [options[s] for s in selected]
            ids_str = ",".join(str(i) for i in selected_ids)
            try:
                cmp_res = requests.get(f"{API_BASE_URL}/compare/?ids={ids_str}")
                cmp_res.raise_for_status()
                comp_data = cmp_res.json()["evaluations"]

                categories = ["Stability", "Adv. Resilience", "Boundary Resilience", "Fairness", "CV Mean"]
                fig_radar = go.Figure()
                for ev in comp_data:
                    rob = ev.get("robustness_details") or {}
                    fair = ev.get("fairness_details") or {}
                    cv_mean = (rob.get("cv_stability") or {}).get("mean", 0) * 100
                    values = [
                        rob.get("stability_score", 0),
                        rob.get("adversarial_resilience", 0),
                        rob.get("boundary_resilience", 0),
                        fair.get("fairness_score", 0) or 0,
                        cv_mean,
                    ]
                    values.append(values[0])
                    cats = categories + [categories[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=cats, fill="toself",
                        name=f"#{ev['id']} {ev['model_name']}",
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Model Comparison — Radar Chart",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.subheader("Comparison Table")
                rows = []
                for ev in comp_data:
                    rob = ev.get("robustness_details") or {}
                    fair = ev.get("fairness_details") or {}
                    conf = rob.get("confidence_analysis", {})
                    rows.append({
                        "ID": ev["id"],
                        "Model": ev["model_name"],
                        "Task": ev["task_type"],
                        "Target Col": ev.get("target_column", "N/A"),
                        "Stability %": rob.get("stability_score", "N/A"),
                        "Adv. Resilience %": rob.get("adversarial_resilience", "N/A"),
                        "Boundary %": rob.get("boundary_resilience", "N/A"),
                        "Fairness %": fair.get("fairness_score", "N/A"),
                        "CV Mean": (rob.get("cv_stability") or {}).get("mean", "N/A"),
                        "Mean Confidence": f"{conf.get('mean_confidence',0)*100:.1f}%" if conf.get("available") else "N/A",
                        "Duration (s)": ev.get("duration_seconds", "N/A"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Comparison failed: {e}")


# ═══════════════════════════════════════════════════════
#  PDF Helper
# ═══════════════════════════════════════════════════════
def _generate_pdf(markdown_text: str) -> bytes:
    """Generate a PDF from the markdown audit report using fpdf2."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in markdown_text.split("\n"):
        clean = line.replace("**", "").replace("###", "").replace("---", "────────────────")
        if line.startswith("###"):
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, clean.strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("####"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, clean.strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("- "):
            pdf.cell(5)
            pdf.multi_cell(0, 6, f"• {clean[2:].strip()}")
        elif line.strip() == "":
            pdf.ln(3)
        else:
            pdf.multi_cell(0, 6, clean.strip())
    return pdf.output()
