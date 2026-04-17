# app/llm_suggestions.py
import os
import json
from typing import Dict, List, Optional


def _risk_bucket(prob: float) -> str:
    pct = int(round(prob * 100))
    low = (pct // 10) * 10
    high = min(low + 10, 100)
    return f"{low}-{high}%"


def _risk_label(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    elif prob < 0.4:
        return "Mild"
    elif prob < 0.6:
        return "Moderate"
    elif prob < 0.8:
        return "High"
    else:
        return "Very High"


def _rule_based(patient: Dict, risk_prob: float) -> Dict[str, object]:
    risk_pct = int(round(risk_prob * 100))

    cp = str(patient.get("Chest Pain Type", "")).lower()
    fbs = float(patient.get("Fasting Blood Sugar", 0) or 0)
    chol = float(patient.get("Total Cholesterol", 0) or 0)
    rbp = float(patient.get("Resting Blood Pressure", 0) or 0)
    bmi = float(patient.get("Body Mass Index (BMI)", 0) or 0)
    ex_ang = str(patient.get("Exercise Induced Angina", "")).lower()
    smoke = str(patient.get("Smoking Status", "")).lower()

    # ── Doctor-facing ──────────────────────────────────────────────────────────
    doc_summary = (
        f"Patient presents with a predicted cardiac event probability of {risk_pct}% "
        f"(risk band: {_risk_bucket(risk_prob)}, category: {_risk_label(risk_prob)}). "
        f"Key contributing factors identified from Random Forest feature importances include "
        f"resting blood pressure ({rbp} mmHg), total cholesterol ({chol} mg/dL), "
        f"fasting blood sugar ({fbs} mg/dL), and BMI ({bmi}). "
        f"Clinical review and risk stratification are recommended."
    )

    doc_actions: List[str] = []
    if rbp >= 140:
        doc_actions.append(f"BP {rbp} mmHg — consider antihypertensive therapy; ABPM may be indicated.")
    elif rbp >= 130:
        doc_actions.append(f"BP {rbp} mmHg — stage 1 hypertension; lifestyle modification + monitor.")
    if chol >= 240:
        doc_actions.append(f"Total cholesterol {chol} mg/dL — high; evaluate LDL-C, consider statin initiation.")
    elif chol >= 200:
        doc_actions.append(f"Total cholesterol {chol} mg/dL — borderline; full lipid panel recommended.")
    if fbs >= 126:
        doc_actions.append(f"FBS {fbs} mg/dL — meets diabetes threshold; order HbA1c + glucose tolerance test.")
    elif fbs >= 100:
        doc_actions.append(f"FBS {fbs} mg/dL — pre-diabetic range; repeat fasting glucose + HbA1c.")
    if "typical" in cp:
        doc_actions.append("Typical angina pattern — stress ECG or myocardial perfusion imaging warranted.")
    if "yes" in ex_ang or ex_ang in {"1", "true"}:
        doc_actions.append("Exercise-induced angina present — urgent cardiology referral for further evaluation.")
    if not doc_actions:
        doc_actions.append("No critical red flags; continue risk-factor monitoring per guidelines.")

    doc_lifestyle: List[str] = []
    doc_lifestyle.append("Prescribe DASH diet; restrict sodium <2.3 g/day; limit saturated fats.")
    doc_lifestyle.append("Target 150 min/week moderate aerobic exercise; escalate gradually under supervision.")
    if bmi >= 30:
        doc_lifestyle.append(f"BMI {bmi} — obesity class; refer to structured weight management programme.")
    elif bmi >= 25:
        doc_lifestyle.append(f"BMI {bmi} — overweight; target 5–10% body weight reduction.")
    if "current" in smoke:
        doc_lifestyle.append("Active smoker — prescribe NRT + varenicline; smoking cessation counseling.")

    doc_recommendations: List[str] = []
    doc_recommendations.append("Order: CBC, CMP, fasting lipid panel, HbA1c, ECG, echocardiogram if indicated.")
    doc_recommendations.append("Risk stratify using Framingham or ASCVD 10-year risk score.")
    doc_recommendations.append("Review all current medications for cardiotoxicity or drug interactions.")
    doc_recommendations.append("Schedule follow-up in 4–6 weeks; sooner if symptoms escalate.")

    # ── Patient-facing ─────────────────────────────────────────────────────────
    if risk_prob < 0.2:
        pat_summary = (
            f"Good news! Your heart health risk score is {risk_pct}%, which is LOW. "
            f"This means your heart looks fairly healthy right now. "
            f"Keep up your good habits and do regular checkups to stay this way!"
        )
    elif risk_prob < 0.4:
        pat_summary = (
            f"Your heart risk score is {risk_pct}% — a MILD level of concern. "
            f"This doesn't mean you have heart disease, but your body is showing some early warning signs. "
            f"Small lifestyle changes now can make a big difference!"
        )
    elif risk_prob < 0.6:
        pat_summary = (
            f"Your heart risk score is {risk_pct}% — a MODERATE level. "
            f"This means some of your health numbers need attention. "
            f"Please visit your doctor soon so they can check your heart more carefully."
        )
    elif risk_prob < 0.8:
        pat_summary = (
            f"Your heart risk score is {risk_pct}% — this is HIGH. "
            f"Your health numbers suggest your heart may be under stress. "
            f"Please see a doctor as soon as possible for a full heart check-up."
        )
    else:
        pat_summary = (
            f"Your heart risk score is {risk_pct}% — this is VERY HIGH. "
            f"Please do not delay — visit a cardiologist or hospital right away for a thorough evaluation."
        )

    pat_actions: List[str] = []
    if rbp >= 130:
        pat_actions.append(f"Your blood pressure ({rbp}) is high — check it daily at home and show the readings to your doctor.")
    if chol >= 200:
        pat_actions.append(f"Your cholesterol ({chol}) is above normal — get a blood test (lipid panel) soon.")
    if fbs >= 100:
        pat_actions.append(f"Your blood sugar ({fbs}) is higher than normal — ask your doctor about a diabetes test.")
    if "typical" in cp:
        pat_actions.append("You have chest pain — avoid heavy exercise and see a doctor promptly.")
    if "yes" in ex_ang or ex_ang in {"1", "true"}:
        pat_actions.append("Your chest hurts during exercise — stop strenuous activity and see your doctor soon.")
    if not pat_actions:
        pat_actions.append("No urgent steps needed right now — just keep your next routine check-up appointment.")

    pat_lifestyle: List[str] = []
    pat_lifestyle.append("Eat more fruits, vegetables, and whole grains. Avoid oily, salty, and processed food.")
    pat_lifestyle.append("Walk for at least 30 minutes every day — even a gentle walk helps your heart.")
    if bmi >= 25:
        pat_lifestyle.append(f"Your weight (BMI {bmi}) is a little high — losing even a few kilos will help your heart.")
    if "current" in smoke:
        pat_lifestyle.append("Smoking is harming your heart — talk to your doctor about quitting. It's the single best thing you can do.")
    pat_lifestyle.append("Sleep 7–8 hours each night and try to reduce stress through relaxation or light yoga.")

    pat_recommendations: List[str] = []
    pat_recommendations.append("Visit your doctor with this report — they will guide you on next steps.")
    pat_recommendations.append("Keep a small diary — note any chest pain, breathlessness, or dizziness and when it happens.")
    pat_recommendations.append("If you ever feel sudden chest pain, sweating, or can't breathe — go to hospital immediately.")
    pat_recommendations.append("Bring your family member along to your doctor visit so they understand your situation too.")

    return {
        "prediction_value": f"{risk_pct}%",
        "risk_band": _risk_bucket(risk_prob),
        "risk_label": _risk_label(risk_prob),
        # Doctor view
        "doctor": {
            "summary": doc_summary,
            "clinical_actions": doc_actions,
            "lifestyle_prescription": doc_lifestyle,
            "investigations": doc_recommendations,
        },
        # Patient view
        "patient": {
            "summary": pat_summary,
            "what_to_do_now": pat_actions,
            "healthy_habits": pat_lifestyle,
            "important_reminders": pat_recommendations,
        },
    }


def _try_langchain(patient: Dict, risk_prob: float) -> Optional[Dict[str, object]]:
    """Use Groq via LangChain. Returns None on any failure."""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        llm = ChatGroq(model=model_name, temperature=0.2)

        prompt = PromptTemplate.from_template(
            """You are a cardiac risk assistant. Output STRICT JSON only — no markdown, no extra text.
Schema:
{{
  "doctor": {{
    "summary": "<clinical paragraph <=100 words using medical terminology>",
    "clinical_actions": ["<action 1>", "<action 2>", "..."],
    "lifestyle_prescription": ["<prescription 1>", "..."],
    "investigations": ["<test/referral 1>", "..."]
  }},
  "patient": {{
    "summary": "<simple English paragraph <=80 words, no jargon>",
    "what_to_do_now": ["<simple step 1>", "..."],
    "healthy_habits": ["<habit 1>", "..."],
    "important_reminders": ["<reminder 1>", "..."]
  }}
}}

Rules:
- Doctor view: clinical terms, evidence-based, objective.
- Patient view: simple everyday language, reassuring tone, avoid medical jargon.
- No diagnosis. No unsafe advice.

Patient data (JSON):
{patient_json}

Predicted risk probability (0–1): {risk_prob}
"""
        )

        chain = prompt | llm | StrOutputParser()
        txt = chain.invoke({
            "patient_json": json.dumps(patient),
            "risk_prob": round(float(risk_prob), 4)
        })

        # Strip possible markdown fences
        txt = txt.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(txt)

        risk_pct = int(round(risk_prob * 100))
        return {
            "prediction_value": f"{risk_pct}%",
            "risk_band": _risk_bucket(risk_prob),
            "risk_label": _risk_label(risk_prob),
            "doctor": {
                "summary": data["doctor"].get("summary", "").strip(),
                "clinical_actions": data["doctor"].get("clinical_actions", []),
                "lifestyle_prescription": data["doctor"].get("lifestyle_prescription", []),
                "investigations": data["doctor"].get("investigations", []),
            },
            "patient": {
                "summary": data["patient"].get("summary", "").strip(),
                "what_to_do_now": data["patient"].get("what_to_do_now", []),
                "healthy_habits": data["patient"].get("healthy_habits", []),
                "important_reminders": data["patient"].get("important_reminders", []),
            },
        }

    except Exception:
        return None


def generate(patient: Dict, risk_prob: float) -> Dict[str, object]:
    out = _try_langchain(patient, risk_prob)
    return out if out else _rule_based(patient, risk_prob)
