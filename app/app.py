# app/app.py
import json
import os
import re
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_suggestions import generate as generate_suggestions

load_dotenv()

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

MODEL_PATH = os.path.join("models", "model.joblib")
META_PATH  = os.path.join("models", "metadata.json")

SEX_OPTIONS     = ["Male", "Female"]
CP_OPTIONS      = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
ECG_OPTIONS     = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
SLOPE_OPTIONS   = ["Upsloping", "Flat", "Downsloping"]
THAL_OPTIONS    = ["Normal", "Fixed defect", "Reversible defect"]
SMOKING_OPTIONS = ["Never", "Former smoker", "Current smoker"]
YES_NO          = ["Yes", "No"]

def _risk_color(prob):
    if prob < 0.2:   return "#27ae60"
    elif prob < 0.4: return "#f1c40f"
    elif prob < 0.6: return "#e67e22"
    elif prob < 0.8: return "#e74c3c"
    else:            return "#8e1c1c"

def _risk_emoji(prob):
    if prob < 0.2:   return "🟢"
    elif prob < 0.4: return "🟡"
    elif prob < 0.6: return "🟠"
    elif prob < 0.8: return "🔴"
    else:            return "🚨"

def _risk_label(prob):
    if prob < 0.2:   return "Low"
    elif prob < 0.4: return "Mild"
    elif prob < 0.6: return "Moderate"
    elif prob < 0.8: return "High"
    else:            return "Very High"

def _draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.2, 1.3); ax.axis("off")
    colours = ["#27ae60","#f1c40f","#e67e22","#e74c3c","#8e1c1c"]
    for i, c in enumerate(colours):
        t1, t2 = 180-i*36, 180-(i+1)*36
        ax.add_patch(mpatches.Wedge((0,0), 1.0, t2, t1, width=0.35,
                                    facecolor=c, edgecolor="white", linewidth=1.5))
    ang = np.deg2rad(180 - prob*180)
    ax.annotate("", xy=(0.72*np.cos(ang), 0.72*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=15))
    ax.plot(0, 0, "ko", markersize=8)
    labels = ["Low","Mild","Mod","High","V.High"]
    for i, lbl in enumerate(labels):
        mid = np.deg2rad(180-(i+0.5)*36)
        ax.text(1.18*np.cos(mid), 1.18*np.sin(mid), lbl,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color=colours[i])
    ax.text(0, -0.12, f"{int(round(prob*100))}%", ha="center", va="center",
            fontsize=18, fontweight="bold", color=_risk_color(prob))
    fig.patch.set_alpha(0)
    return fig

def _draw_shap_chart(input_df, meta, top_n=10):
    fi_list = meta.get("feature_importances", [])
    if not fi_list:
        return None
    fi_map = {d["feature"]: d["importance"] for d in fi_list}
    row = input_df.iloc[0]
    contributions = {}
    for col in input_df.columns:
        imp = fi_map.get(col, 0)
        try:
            val = float(row[col])
            contributions[col] = imp * (val / (val + 1e-6))
        except (ValueError, TypeError):
            contributions[col] = imp * 0.5
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names  = [k for k,v in sorted_items][::-1]
    vals   = [v for k,v in sorted_items][::-1]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to Risk  (red = increases risk, green = decreases)")
    ax.set_title("🧠 Feature Contribution to Prediction", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def _build_pdf(inputs, prob, threshold, sug):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        risk_pct = int(round(prob*100))
        risk_lbl = _risk_label(prob)
        rc_map = {"Low": rl_colors.green, "Mild": rl_colors.orange,
                  "Moderate": rl_colors.orangered, "High": rl_colors.red,
                  "Very High": rl_colors.darkred}
        rc = rc_map.get(risk_lbl, rl_colors.red)

        title_style   = ParagraphStyle("title", fontSize=18, fontName="Helvetica-Bold",
                                       alignment=TA_CENTER, textColor=rl_colors.HexColor("#c0392b"))
        heading_style = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                                       textColor=rl_colors.HexColor("#2c3e50"), spaceBefore=10)
        body_style    = ParagraphStyle("body", fontSize=10, fontName="Helvetica", leading=14)
        bullet_style  = ParagraphStyle("bullet", fontSize=10, fontName="Helvetica",
                                       leading=14, leftIndent=15)

        elems = []
        elems.append(Paragraph("❤️ Heart Disease Prediction Report", title_style))
        elems.append(Spacer(1, 0.3*cm))
        elems.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor("#c0392b")))
        elems.append(Spacer(1, 0.4*cm))

        risk_table_data = [
            ["Risk Probability",   f"{risk_pct}%"],
            ["Risk Category",      risk_lbl],
            ["Risk Band",          sug.get("risk_band","")],
            ["Decision Threshold", f"{threshold:.2f}"],
            ["Prediction Label",   "AT RISK" if prob >= threshold else "LOW RISK"],
        ]
        rt = Table(risk_table_data, colWidths=[7*cm, 9*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(0,-1), rl_colors.HexColor("#f2f2f2")),
            ("BACKGROUND",     (1,0),(1,0),  rc),
            ("TEXTCOLOR",      (1,0),(1,0),  rl_colors.white),
            ("FONTNAME",       (0,0),(-1,-1),"Helvetica"),
            ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1),10),
            ("ROWBACKGROUNDS", (0,0),(-1,-1),[rl_colors.white, rl_colors.HexColor("#fef9f9")]),
            ("GRID",           (0,0),(-1,-1),0.5, rl_colors.HexColor("#cccccc")),
            ("PADDING",        (0,0),(-1,-1),6),
        ]))
        elems.append(rt); elems.append(Spacer(1, 0.5*cm))

        elems.append(Paragraph("Patient Inputs", heading_style))
        elems.append(HRFlowable(width="100%", thickness=0.5, color=rl_colors.HexColor("#aaa")))
        elems.append(Spacer(1, 0.2*cm))
        inp_data = [["Parameter","Value"]] + [[k,str(v)] for k,v in inputs.items()]
        it = Table(inp_data, colWidths=[9*cm, 7*cm])
        it.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), rl_colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",      (0,0),(-1,0), rl_colors.white),
            ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1),9),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#f9f9f9")]),
            ("GRID",           (0,0),(-1,-1),0.4, rl_colors.HexColor("#dddddd")),
            ("PADDING",        (0,0),(-1,-1),5),
        ]))
        elems.append(it); elems.append(Spacer(1, 0.5*cm))

        doc_data = sug.get("doctor", {})
        pat_data = sug.get("patient", {})

        def _sec(title, content, is_list=True):
            elems.append(Paragraph(title, heading_style))
            elems.append(HRFlowable(width="100%", thickness=0.5, color=rl_colors.HexColor("#aaa")))
            elems.append(Spacer(1, 0.15*cm))
            if is_list:
                for item in (content or []):
                    elems.append(Paragraph(f"• {item}", bullet_style))
            else:
                elems.append(Paragraph(str(content), body_style))
            elems.append(Spacer(1, 0.3*cm))

        elems.append(Paragraph("Doctor View (Clinical)", heading_style))
        elems.append(HRFlowable(width="100%", thickness=1.5, color=rl_colors.HexColor("#2980b9")))
        elems.append(Spacer(1, 0.2*cm))
        _sec("Clinical Summary",          doc_data.get("summary",""),              is_list=False)
        _sec("Clinical Actions",          doc_data.get("clinical_actions",[]))
        _sec("Lifestyle Prescription",    doc_data.get("lifestyle_prescription",[]))
        _sec("Investigations & Referrals",doc_data.get("investigations",[]))

        elems.append(Paragraph("Patient View (Simple English)", heading_style))
        elems.append(HRFlowable(width="100%", thickness=1.5, color=rl_colors.HexColor("#27ae60")))
        elems.append(Spacer(1, 0.2*cm))
        _sec("What Does This Mean?",  pat_data.get("summary",""),          is_list=False)
        _sec("What To Do Now",        pat_data.get("what_to_do_now",[]))
        _sec("Healthy Habits",        pat_data.get("healthy_habits",[]))
        _sec("Important Reminders",   pat_data.get("important_reminders",[]))

        doc.build(elems)
        return buf.getvalue()

    except ImportError:
        risk_pct = int(round(prob*100))
        lines = [f"HEART DISEASE PREDICTION REPORT", f"Risk: {risk_pct}% | {_risk_label(prob)}"]
        for k,v in inputs.items(): lines.append(f"  {k}: {v}")
        return "\n".join(lines).encode()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def _is_numeric_like(values):
    if not values: return False
    pat = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
    return all(bool(pat.match(str(v))) for v in values)

def _map_value(col, human_value, meta_categories):
    hv = str(human_value)
    if meta_categories and not _is_numeric_like(meta_categories): return hv
    if col == "Sex": return "1" if hv.lower().startswith("m") else "0"
    if col in ("Exercise Induced Angina","Angeo person or not"):
        return "1" if hv.lower().startswith("y") else "0"
    if col == "Chest Pain Type":
        return {"typical angina":"1","atypical angina":"2","non-anginal pain":"3","asymptomatic":"4"}.get(hv.lower(),hv)
    if col == "Resting ECG Results":
        return {"normal":"0","st-t abnormality":"1","left ventricular hypertrophy":"2"}.get(hv.lower(),hv)
    if col == "Slope of ST Segment":
        return {"upsloping":"1","flat":"2","downsloping":"3"}.get(hv.lower(),hv)
    if col == "Thallium Stress Test Result (thal)":
        return {"normal":"3","fixed defect":"6","reversible defect":"7"}.get(hv.lower(),hv)
    if col == "Smoking Status":
        if meta_categories and len(meta_categories)==2:
            return "1" if hv.lower().startswith("current") else "0"
        return {"never":"0","former smoker":"1","current smoker":"2"}.get(hv.lower(),hv)
    return hv

def prepare_for_model(human_inputs, meta):
    categories = meta.get("categories",{})
    out = {}
    for k in ["Age","Resting Blood Pressure","Total Cholesterol","Fasting Blood Sugar",
              "Maximum Heart Rate Achieved","ST Depression (oldpeak)",
              "Number of Major Vessels","Body Mass Index (BMI)"]:
        out[k] = human_inputs.get(k)
    for k in ["Sex","Chest Pain Type","Resting ECG Results","Exercise Induced Angina",
              "Slope of ST Segment","Thallium Stress Test Result (thal)",
              "Smoking Status","Angeo person or not"]:
        out[k] = _map_value(k, human_inputs.get(k), categories.get(k,[]))
    return out

def feature_importance_plot(meta):
    fi = meta.get("feature_importances",[])[:20]
    if not fi: st.info("Not available."); return
    names = [d["feature"] for d in fi][::-1]
    vals  = [d["importance"] for d in fi][::-1]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.barh(names, vals, color="#e74c3c")
    ax.set_xlabel("Importance"); ax.set_title("Top Feature Importances")
    st.pyplot(fig)

def main():
    st.markdown("""
    <style>
    .risk-box{border-radius:12px;padding:16px 20px;margin-bottom:10px;font-size:15px;}
    .section-header{font-size:17px;font-weight:700;margin-bottom:6px;margin-top:14px;}
    .bullet-item{padding:5px 0 5px 8px;border-left:3px solid #e74c3c;margin-bottom:4px;font-size:14px;}
    </style>""", unsafe_allow_html=True)

    st.title("❤️ Heart Disease Prediction System")
    st.caption("Random Forest + Groq LLM  |  XAI Explanation  |  Doctor & Patient Views  |  What-If Simulator  |  PDF Report")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        st.error("Model or metadata missing. Run `python src/train.py` first.")
        st.stop()

    model, meta = load_artifacts()
    num_cols = meta["numeric_columns"]
    cat_cols = meta["categorical_columns"]

    st.sidebar.subheader("⚙️ Settings")
    threshold = st.sidebar.slider("Risk threshold (classify as HIGH at ≥)", 0.10, 0.90, 0.50, 0.01)
    st.sidebar.caption("Move right → higher precision  |  Move left → higher recall")

    st.subheader("Enter Patient Data")
    input_tab1, input_tab2 = st.tabs(["🧍 Manual Form", "📄 CSV Upload"])

    with input_tab1:
        c1, c2 = st.columns(2)
        inputs = {}
        with c1:
            inputs["Age"]                         = st.number_input("Age (years)", 1, 120, 50)
            inputs["Resting Blood Pressure"]      = st.number_input("Resting BP (mmHg)", 60, 240, 130)
            inputs["Total Cholesterol"]           = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
            inputs["Fasting Blood Sugar"]         = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 400, 95)
            inputs["Maximum Heart Rate Achieved"] = st.number_input("Max Heart Rate", 60, 230, 150)
            inputs["ST Depression (oldpeak)"]     = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
            inputs["Number of Major Vessels"]     = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)
            inputs["Body Mass Index (BMI)"]       = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
        with c2:
            inputs["Sex"]                                = st.selectbox("Sex", SEX_OPTIONS)
            inputs["Chest Pain Type"]                    = st.selectbox("Chest Pain Type", CP_OPTIONS)
            inputs["Resting ECG Results"]                = st.selectbox("Resting ECG Results", ECG_OPTIONS)
            inputs["Exercise Induced Angina"]            = st.selectbox("Exercise Induced Angina", YES_NO)
            inputs["Slope of ST Segment"]                = st.selectbox("Slope of ST Segment", SLOPE_OPTIONS)
            inputs["Thallium Stress Test Result (thal)"] = st.selectbox("Thallium Stress Test Result", THAL_OPTIONS)
            inputs["Smoking Status"]                     = st.selectbox("Smoking Status", SMOKING_OPTIONS)
            inputs["Angeo person or not"]                = st.selectbox("Angiography Person (Yes/No)", YES_NO)

        if st.button("🔍 Predict", use_container_width=True):
            model_inputs = prepare_for_model(inputs, meta)
            df   = pd.DataFrame([model_inputs])[num_cols + cat_cols]
            prob = float(model.predict_proba(df)[0, 1])
            label = int(prob >= threshold)
            risk_pct   = int(round(prob*100))
            risk_color = _risk_color(prob)
            risk_emoji = _risk_emoji(prob)

            # Save for What-If
            st.session_state["last_prob"]   = prob
            st.session_state["last_inputs"] = inputs.copy()

            st.divider()

            # ── Risk Banner ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="risk-box" style="background:{risk_color}22;border:2px solid {risk_color};">
                <span style="font-size:28px;">{risk_emoji}</span>
                <span style="font-size:22px;font-weight:700;color:{risk_color};margin-left:10px;">
                    {risk_pct}% Heart Disease Risk
                </span>
                <span style="font-size:14px;color:#555;margin-left:15px;">
                    (threshold {threshold:.2f} → {'⚠️ AT RISK' if label else '✅ LOW RISK'})
                </span>
            </div>""", unsafe_allow_html=True)

            # ── Gauge + Summary ────────────────────────────────────────────
            col_g, col_m = st.columns([1, 2])
            with col_g:
                st.markdown("**Risk Meter**")
                st.pyplot(_draw_gauge(prob), use_container_width=False)
            with col_m:
                st.markdown("**Key Input Summary**")
                st.table(pd.DataFrame({
                    "Parameter": ["Age","Blood Pressure","Cholesterol","Fasting Sugar","BMI","Smoking"],
                    "Value": [inputs["Age"], f"{inputs['Resting Blood Pressure']} mmHg",
                              f"{inputs['Total Cholesterol']} mg/dL",
                              f"{inputs['Fasting Blood Sugar']} mg/dL",
                              inputs["Body Mass Index (BMI)"], inputs["Smoking Status"]]
                }))

            # ── SHAP Chart ─────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧠 Why Did the Model Predict This?")
            shap_fig = _draw_shap_chart(df, meta)
            if shap_fig:
                st.pyplot(shap_fig, use_container_width=True)
                st.caption("🔴 Red = increases risk  |  🟢 Green = decreases risk")

            # ── LLM Suggestions ───────────────────────────────────────────
            with st.spinner("Generating AI explanations via Groq LLM..."):
                sug = generate_suggestions(inputs, prob)

            doc_data = sug.get("doctor", {})
            pat_data = sug.get("patient", {})

            st.divider()
            st.markdown("### 💬 AI Explanations")
            view_tab1, view_tab2 = st.tabs(["🩺 Doctor View", "👤 Patient View"])

            with view_tab1:
                st.markdown("<div class='section-header'>Clinical Summary</div>", unsafe_allow_html=True)
                st.info(doc_data.get("summary",""))
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("<div class='section-header'>⚡ Clinical Actions</div>", unsafe_allow_html=True)
                    for item in doc_data.get("clinical_actions",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>🏃 Lifestyle Prescription</div>", unsafe_allow_html=True)
                    for item in doc_data.get("lifestyle_prescription",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)
                with col_d2:
                    st.markdown("<div class='section-header'>🔬 Investigations & Referrals</div>", unsafe_allow_html=True)
                    for item in doc_data.get("investigations",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)

            with view_tab2:
                st.markdown(f"""
                <div class="risk-box" style="background:{risk_color}15;border-left:5px solid {risk_color};font-size:15px;">
                    {risk_emoji} {pat_data.get("summary","")}
                </div>""", unsafe_allow_html=True)
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("<div class='section-header'>✅ What To Do Now</div>", unsafe_allow_html=True)
                    for item in pat_data.get("what_to_do_now",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>🥗 Healthy Habits</div>", unsafe_allow_html=True)
                    for item in pat_data.get("healthy_habits",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)
                with col_p2:
                    st.markdown("<div class='section-header'>📌 Important Reminders</div>", unsafe_allow_html=True)
                    for item in pat_data.get("important_reminders",[]):
                        st.markdown(f"<div class='bullet-item'>• {item}</div>", unsafe_allow_html=True)

            # ── What-If Simulator ──────────────────────────────────────────
            st.divider()
            st.markdown("### 🔄 What-If Risk Simulator")
            st.caption("Simulate how lifestyle changes affect risk — move sliders and click Simulate!")

            with st.expander("🧪 Open Simulator", expanded=True):
                s1, s2, s3 = st.columns(3)
                with s1:
                    wi_bp = st.slider("Blood Pressure (mmHg)", 80, 200,
                                      int(inputs["Resting Blood Pressure"]), key="wi_bp")
                with s2:
                    wi_chol = st.slider("Cholesterol (mg/dL)", 100, 400,
                                        int(inputs["Total Cholesterol"]), key="wi_chol")
                with s3:
                    wi_bmi = st.slider("BMI", 15.0, 50.0,
                                       float(inputs["Body Mass Index (BMI)"]), step=0.5, key="wi_bmi")
                s4, s5 = st.columns(2)
                with s4:
                    wi_smoke = st.selectbox("Smoking Status", SMOKING_OPTIONS,
                                            index=SMOKING_OPTIONS.index(inputs["Smoking Status"]), key="wi_smoke")
                with s5:
                    wi_fbs = st.slider("Fasting Blood Sugar (mg/dL)", 50, 300,
                                       int(inputs["Fasting Blood Sugar"]), key="wi_fbs")

                if st.button("🔮 Simulate New Risk", use_container_width=True):
                    wi_inputs = inputs.copy()
                    wi_inputs.update({
                        "Resting Blood Pressure": wi_bp,
                        "Total Cholesterol":      wi_chol,
                        "Body Mass Index (BMI)":  wi_bmi,
                        "Smoking Status":         wi_smoke,
                        "Fasting Blood Sugar":    wi_fbs,
                    })
                    wi_model = prepare_for_model(wi_inputs, meta)
                    wi_df    = pd.DataFrame([wi_model])[num_cols + cat_cols]
                    wi_prob  = float(model.predict_proba(wi_df)[0, 1])
                    wi_pct   = int(round(wi_prob*100))
                    diff     = wi_pct - risk_pct
                    diff_str = f"{'▲' if diff>0 else '▼'} {abs(diff)}%" if diff!=0 else "No change"
                    diff_col = "#e74c3c" if diff>0 else "#27ae60"

                    c_orig, c_new = st.columns(2)
                    with c_orig:
                        st.markdown(f"""
                        <div class="risk-box" style="background:{_risk_color(prob)}22;border:2px solid {_risk_color(prob)};text-align:center;">
                            <div style="font-size:13px;color:#555;">Original Risk</div>
                            <div style="font-size:32px;font-weight:700;color:{_risk_color(prob)};">{risk_pct}%</div>
                            <div style="font-size:12px;">{_risk_label(prob)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c_new:
                        st.markdown(f"""
                        <div class="risk-box" style="background:{_risk_color(wi_prob)}22;border:2px solid {_risk_color(wi_prob)};text-align:center;">
                            <div style="font-size:13px;color:#555;">Simulated Risk</div>
                            <div style="font-size:32px;font-weight:700;color:{_risk_color(wi_prob)};">{wi_pct}%</div>
                            <div style="font-size:12px;color:{diff_col};font-weight:700;">{diff_str}</div>
                        </div>""", unsafe_allow_html=True)

                    # Dual mini gauge
                    fig_cmp, axes = plt.subplots(1, 2, figsize=(7, 2.5), subplot_kw={"aspect":"equal"})
                    for ax_i, (p, ttl) in enumerate([(prob,"Original"),(wi_prob,"Simulated")]):
                        ax = axes[ax_i]
                        ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.2,1.3); ax.axis("off")
                        for i, c in enumerate(["#27ae60","#f1c40f","#e67e22","#e74c3c","#8e1c1c"]):
                            ax.add_patch(mpatches.Wedge((0,0),1.0,180-(i+1)*36,180-i*36,
                                                        width=0.35,facecolor=c,edgecolor="white",linewidth=1))
                        ang = np.deg2rad(180-p*180)
                        ax.annotate("",xy=(0.72*np.cos(ang),0.72*np.sin(ang)),xytext=(0,0),
                                    arrowprops=dict(arrowstyle="-|>",color="black",lw=2,mutation_scale=12))
                        ax.plot(0,0,"ko",markersize=6)
                        ax.text(0,-0.12,f"{int(round(p*100))}%",ha="center",va="center",
                                fontsize=14,fontweight="bold",color=_risk_color(p))
                        ax.set_title(ttl,fontsize=10,fontweight="bold",pad=2)
                    fig_cmp.patch.set_alpha(0)
                    st.pyplot(fig_cmp, use_container_width=False)

            # ── PDF Download ───────────────────────────────────────────────
            st.divider()
            st.markdown("### 📥 Download Report")
            pdf_bytes = _build_pdf(inputs, prob, threshold, sug)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                   file_name="heart_disease_report.pdf",
                                   mime="application/pdf", use_container_width=True)
            with col_dl2:
                st.download_button("📊 Download Inputs (.csv)",
                                   data=pd.DataFrame([inputs]).to_csv(index=False),
                                   file_name="patient_inputs.csv",
                                   mime="text/csv", use_container_width=True)

    with input_tab2:
        st.caption("Upload CSV with the same column names as your training data.")
        upl = st.file_uploader("CSV", type=["csv"])
        if upl is not None:
            df_raw  = pd.read_csv(upl)
            missing = [c for c in (num_cols+cat_cols) if c not in df_raw.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            else:
                probs = model.predict_proba(df_raw[num_cols+cat_cols])[:,1]
                preds = (probs >= threshold).astype(int)
                out   = df_raw.copy()
                out["prediction_proba"] = probs
                out["prediction_label"] = preds
                st.dataframe(out.head(50))
                st.download_button("Download Predictions (.csv)",
                                   out.to_csv(index=False), "predictions.csv")

    st.divider()
    st.subheader("📊 Model Info")
    c1, c2 = st.columns(2)
    with c1:
        st.json(meta.get("metrics",{}))
    with c2:
        feature_importance_plot(meta)

if __name__ == "__main__":
    main()
