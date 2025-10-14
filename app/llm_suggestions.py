# app/llm_suggestions.py
import os
import json
from typing import Dict, List, Optional

def _risk_bucket(prob: float) -> str:
    pct = int(round(prob * 100))
    low = (pct // 10) * 10
    high = min(low + 10, 100)
    return f"{low}-{high}"

def _rule_based(patient: Dict, risk_prob: float) -> Dict[str, object]:
    msgs_lifestyle: List[str] = []
    msgs_immediate: List[str] = []
    msgs_extra: List[str] = []

    risk_pct = int(round(risk_prob * 100))

    cp = str(patient.get("Chest Pain Type", "")).lower()
    fbs = float(patient.get("Fasting Blood Sugar", 0) or 0)
    chol = float(patient.get("Total Cholesterol", 0) or 0)
    rbp = float(patient.get("Resting Blood Pressure", 0) or 0)
    bmi = float(patient.get("Body Mass Index (BMI)", 0) or 0)
    ex_ang = str(patient.get("Exercise Induced Angina", "")).lower()
    smoke = str(patient.get("Smoking Status", "")).lower()

    # Lifestyle
    if bmi >= 25:
        msgs_lifestyle.append("Aim for 5–10% weight reduction with a calorie deficit and 150 min/week moderate exercise.")
    msgs_lifestyle.append("Adopt a DASH/Mediterranean-style diet: more vegetables, whole grains; limit salt and added sugars.")
    msgs_lifestyle.append("Sleep 7–8 hours; manage stress with mindfulness or brisk walking.")
    if "smok" in smoke or smoke in {"yes", "1", "true", "current", "current smoker"}:
        msgs_lifestyle.append("Implement a supervised smoking cessation plan (NRT + counseling).")

    # Immediate actions
    if rbp >= 130:
        msgs_immediate.append("Start home BP monitoring; discuss antihypertensive therapy with your physician.")
    if chol >= 200:
        msgs_immediate.append("Request a fasting lipid panel; discuss statin eligibility.")
    if fbs >= 126:
        msgs_immediate.append("Get HbA1c test to evaluate diabetes; plan glucose management.")
    if "typical" in cp or cp in {"ta", "1", "typical angina"}:
        msgs_immediate.append("Chest pain suggests angina—avoid strenuous exertion and seek a supervised stress test.")
    if "yes" in ex_ang or ex_ang in {"1", "true"}:
        msgs_immediate.append("Exercise triggers chest discomfort—pause intense activity until cardiology review.")

    # Extra recommendations
    msgs_extra.append("Keep a symptom diary (pain timing, triggers, relief) for your next clinic visit.")
    msgs_extra.append("Carry recent labs (lipids, HbA1c) and medication list to appointments.")
    msgs_extra.append("If symptoms escalate (rest pain, sweating, breathlessness), seek urgent care.")

    # Risk-tiered banner
    if risk_prob < 0.2:
        banner = "Low risk: Maintain healthy lifestyle and annual checks."
    elif risk_prob < 0.4:
        banner = "Mild risk: Tighten lifestyle controls; recheck BP/lipids in ~3 months."
    elif risk_prob < 0.6:
        banner = "Moderate risk: Book a primary care/cardiology consult for preventive therapy."
    elif risk_prob < 0.8:
        banner = "High risk: Prioritize cardiology appointment; consider imaging-based risk stratification."
    else:
        banner = "Very high risk: Seek prompt cardiology evaluation."

    summary = (
        f"Predicted heart disease risk ≈ {risk_pct}%. "
        f"Risk band { _risk_bucket(risk_prob) }. {banner}"
    )

    return {
        "prediction_value": f"{risk_pct}%",
        "risk_band": _risk_bucket(risk_prob),
        "summary": summary,
        "immediate_actions": msgs_immediate or ["No urgent red flags from inputs; continue routine monitoring."],
        "lifestyle": msgs_lifestyle or ["Maintain balanced diet and regular activity."],
        "recommendations": msgs_extra
    }

def _try_langchain(patient: Dict, risk_prob: float) -> Optional[Dict[str, object]]:
    """Use Groq via LangChain to produce structured guidance. If anything fails, return None."""
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
            """You are a careful cardiac risk assistant. Output STRICT JSON with keys:
{
 "summary": "<one short paragraph, <=90 words>",
 "immediate_actions": ["<bullet 1>", "<bullet 2>", "..."],
 "lifestyle": ["<bullet 1>", "<bullet 2>", "..."],
 "recommendations": ["<bullet 1>", "<bullet 2>", "..."]
}
Rules: no diagnosis, keep practical, avoid unsafe advice.

Patient (JSON):
{patient_json}

Predicted risk probability (0-1): {risk_prob}
"""
        )
        chain = prompt | llm | StrOutputParser()
        txt = chain.invoke({"patient_json": json.dumps(patient), "risk_prob": round(float(risk_prob), 4)})
        data = json.loads(txt)

        return {
            "prediction_value": f"{int(round(risk_prob*100))}%",
            "risk_band": _risk_bucket(risk_prob),
            "summary": data.get("summary", "").strip(),
            "immediate_actions": data.get("immediate_actions", []),
            "lifestyle": data.get("lifestyle", []),
            "recommendations": data.get("recommendations", []),
        }
    except Exception:
        return None

def generate(patient: Dict, risk_prob: float) -> Dict[str, object]:
    out = _try_langchain(patient, risk_prob)
    return out if out else _rule_based(patient, risk_prob)
