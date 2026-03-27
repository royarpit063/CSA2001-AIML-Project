# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
╔══════════════════════════════════════════════════════════════════╗
║         MEDICAL DIAGNOSIS EXPERT SYSTEM                         ║
║         AI Lab Assignment — Knowledge-Based + ML Hybrid         ║
╚══════════════════════════════════════════════════════════════════╝

Concepts Covered:
  • Knowledge Representation  (CO2) — disease-symptom knowledge base
  • Logical Inference         (CO6) — rule-based Prolog-style reasoning
  • Machine Learning          (CO4) — Naive Bayes probabilistic classifier
  • Expert System Design      (CO5) — forward chaining + confidence scoring
"""

# ─────────────────────────────────────────────────────────────────
# DEPENDENCIES  (all standard library + scikit-learn)
# pip install scikit-learn
# ─────────────────────────────────────────────────────────────────

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import numpy as np


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — KNOWLEDGE BASE  (Knowledge Representation)
# Think of this as a medical textbook encoded as Python dicts.
# Each disease has: required symptoms, optional symptoms, severity
# ═══════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "Flu": {
        "required":  ["fever", "body_ache", "fatigue"],
        "optional":  ["cough", "headache", "chills", "sore_throat"],
        "severity":  "moderate",
        "advice":    "Rest, stay hydrated, take paracetamol. See a doctor if fever > 3 days."
    },
    "Common Cold": {
        "required":  ["runny_nose", "sneezing"],
        "optional":  ["sore_throat", "mild_fever", "cough", "headache"],
        "severity":  "mild",
        "advice":    "Rest, warm fluids, vitamin C. Usually resolves in 7–10 days."
    },
    "COVID-19": {
        "required":  ["fever", "cough"],
        "optional":  ["loss_of_smell", "loss_of_taste", "fatigue", "shortness_of_breath", "body_ache"],
        "severity":  "high",
        "advice":    "Isolate immediately. Get tested. Seek emergency care if breathing is difficult."
    },
    "Malaria": {
        "required":  ["fever", "chills"],
        "optional":  ["sweating", "headache", "nausea", "vomiting", "body_ache"],
        "severity":  "high",
        "advice":    "Seek medical attention immediately. Blood test required for confirmation."
    },
    "Typhoid": {
        "required":  ["prolonged_fever", "abdominal_pain"],
        "optional":  ["headache", "fatigue", "loss_of_appetite", "nausea", "constipation"],
        "severity":  "high",
        "advice":    "See a doctor urgently. Requires antibiotic treatment and blood/stool tests."
    },
    "Migraine": {
        "required":  ["severe_headache"],
        "optional":  ["nausea", "vomiting", "light_sensitivity", "sound_sensitivity", "visual_aura"],
        "severity":  "moderate",
        "advice":    "Rest in a dark, quiet room. OTC pain relievers may help. Consult neurologist if frequent."
    },
    "Dengue": {
        "required":  ["high_fever", "severe_headache", "joint_pain"],
        "optional":  ["rash", "eye_pain", "nausea", "vomiting", "fatigue"],
        "severity":  "high",
        "advice":    "Go to hospital immediately. Monitor platelet count. Stay hydrated."
    },
    "Gastroenteritis": {
        "required":  ["diarrhea", "nausea"],
        "optional":  ["vomiting", "abdominal_pain", "mild_fever", "fatigue"],
        "severity":  "moderate",
        "advice":    "ORS (oral rehydration solution), bland diet. See doctor if symptoms persist > 2 days."
    },
    "Hypertension": {
        "required":  ["headache", "dizziness"],
        "optional":  ["blurred_vision", "chest_pain", "shortness_of_breath", "nosebleed"],
        "severity":  "high",
        "advice":    "Check blood pressure immediately. Reduce salt, manage stress. Doctor visit is essential."
    },
    "Asthma": {
        "required":  ["shortness_of_breath", "wheezing"],
        "optional":  ["cough", "chest_tightness", "fatigue"],
        "severity":  "moderate",
        "advice":    "Use prescribed inhaler. Avoid triggers. Go to ER if breathing worsens rapidly."
    },
}

# Master list of all possible symptoms (our feature space)
ALL_SYMPTOMS = sorted({
    symptom
    for disease in KNOWLEDGE_BASE.values()
    for symptom in disease["required"] + disease["optional"]
})


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — RULE-BASED INFERENCE ENGINE  (Prolog-style Logic)
# Implements FORWARD CHAINING:
#   "If the patient HAS all required symptoms, fire this rule"
# ═══════════════════════════════════════════════════════════════

def rule_based_diagnosis(patient_symptoms: list[str]) -> list[dict]:
    """
    Forward chaining rule engine.
    
    For each disease:
      1. Check if ALL required symptoms are present (rule fires = True)
      2. Count how many optional symptoms match (strength score)
      3. Compute confidence = matched / total_possible × 100
    
    Returns ranked list of possible diagnoses.
    """
    patient_set = set(patient_symptoms)
    results = []

    for disease, rules in KNOWLEDGE_BASE.items():
        required   = set(rules["required"])
        optional   = set(rules["optional"])
        total_syms = len(required) + len(optional)

        # RULE FIRES only if ALL required symptoms are present
        rule_fired = required.issubset(patient_set)

        if rule_fired:
            matched_optional  = len(optional & patient_set)
            matched_required  = len(required)
            total_matched     = matched_required + matched_optional

            # Confidence formula: weighted toward required symptoms
            confidence = round((total_matched / total_syms) * 100, 1)

            results.append({
                "disease":     disease,
                "confidence":  confidence,
                "severity":    rules["severity"],
                "advice":      rules["advice"],
                "matched_req": sorted(required & patient_set),
                "matched_opt": sorted(optional & patient_set),
            })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — ML CLASSIFIER  (Naive Bayes Probabilistic Model)
# Treats symptoms as features, diseases as labels.
# Gives probability distribution over all diseases.
# ═══════════════════════════════════════════════════════════════

def build_training_data() -> tuple[np.ndarray, list[str]]:
    """
    Auto-generate training data from the knowledge base.
    Each disease gets multiple synthetic training samples:
      - 1 sample with all symptoms present
      - 1 sample with only required symptoms
      - 1 sample with required + random optional
    """
    X, y = [], []
    symptom_index = {s: i for i, s in enumerate(ALL_SYMPTOMS)}

    for disease, rules in KNOWLEDGE_BASE.items():
        required = rules["required"]
        optional = rules["optional"]

        def encode(symptoms):
            vec = [0] * len(ALL_SYMPTOMS)
            for s in symptoms:
                if s in symptom_index:
                    vec[symptom_index[s]] = 1
            return vec

        # Sample 1: all symptoms
        X.append(encode(required + optional))
        y.append(disease)

        # Sample 2: only required
        X.append(encode(required))
        y.append(disease)

        # Sample 3: required + first half optional
        half = optional[:len(optional)//2]
        X.append(encode(required + half))
        y.append(disease)

    return np.array(X), y


def train_ml_model():
    """Train Naive Bayes classifier on symptom→disease data."""
    X, y = build_training_data()
    model = MultinomialNB()
    model.fit(X, y)
    return model, y


def ml_diagnosis(patient_symptoms: list[str], model, classes: list[str]) -> list[dict]:
    """
    ML-based diagnosis using trained Naive Bayes.
    Returns probability of each disease for the given symptom vector.
    """
    symptom_index = {s: i for i, s in enumerate(ALL_SYMPTOMS)}
    vec = [0] * len(ALL_SYMPTOMS)
    for s in patient_symptoms:
        if s in symptom_index:
            vec[symptom_index[s]] = 1

    proba = model.predict_proba([vec])[0]
    results = [
        {"disease": cls, "probability": round(p * 100, 1)}
        for cls, p in zip(model.classes_, proba)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results[:5]  # Top 5


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — HYBRID DIAGNOSIS  (Combine Rule + ML)
# Final score = 0.6 × rule_confidence + 0.4 × ml_probability
# ═══════════════════════════════════════════════════════════════

def hybrid_diagnosis(patient_symptoms: list[str], model, classes) -> list[dict]:
    """
    Merges rule-based and ML outputs into a single ranked list.
    Rule engine catches only matching diseases (precise).
    ML adds probabilistic coverage for borderline cases.
    """
    rule_results = {r["disease"]: r for r in rule_based_diagnosis(patient_symptoms)}
    ml_results   = {r["disease"]: r["probability"] for r in ml_diagnosis(patient_symptoms, model, classes)}

    all_diseases = set(rule_results) | set(ml_results)
    final = []

    for disease in all_diseases:
        rule_conf = rule_results[disease]["confidence"] if disease in rule_results else 0
        ml_prob   = ml_results.get(disease, 0)
        hybrid    = round(0.6 * rule_conf + 0.4 * ml_prob, 1)

        info = KNOWLEDGE_BASE[disease]
        final.append({
            "disease":       disease,
            "hybrid_score":  hybrid,
            "rule_conf":     rule_conf,
            "ml_prob":       ml_prob,
            "severity":      info["severity"],
            "advice":        info["advice"],
            "rule_matched":  disease in rule_results,
        })

    final.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return final


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

SEVERITY_ICON = {"mild": "🟢", "moderate": "🟡", "high": "🔴"}
SEVERITY_LABEL = {"mild": "MILD", "moderate": "MODERATE", "high": "HIGH RISK"}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        🏥  MEDICAL DIAGNOSIS EXPERT SYSTEM  v1.0               ║
║        Knowledge-Based + Machine Learning Hybrid                ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  DISCLAIMER: For educational purposes only.                 ║
║     Always consult a qualified doctor for real diagnosis.       ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def display_results(results: list[dict], top_n=3):
    """Pretty-print the hybrid diagnosis results."""
    if not results:
        print("\n  ❌  No matching diagnosis found for the given symptoms.")
        print("     Please provide more symptoms or consult a doctor.\n")
        return

    print_section("📊  DIAGNOSIS RESULTS  (Top Matches)")

    for i, r in enumerate(results[:top_n], 1):
        sev   = r["severity"]
        icon  = SEVERITY_ICON[sev]
        label = SEVERITY_LABEL[sev]
        tag   = "✅ Rule Matched" if r["rule_matched"] else "🤖 ML Predicted"

        print(f"""
  [{i}] {r['disease']}  {icon} {label}
      ┌─────────────────────────────────────────┐
      │  Hybrid Score  : {r['hybrid_score']:>5.1f}%               │
      │  Rule Engine   : {r['rule_conf']:>5.1f}%  {tag:<20}│
      │  ML (Naive Bayes): {r['ml_prob']:>5.1f}%                │
      └─────────────────────────────────────────┘
      💊 Advice: {r['advice']}
""")


def show_symptom_list():
    """Print all known symptoms neatly."""
    print_section("📋  KNOWN SYMPTOMS  (choose from these)")
    cols = 3
    for i, sym in enumerate(ALL_SYMPTOMS, 1):
        end = "\n" if i % cols == 0 else ""
        print(f"  {i:>2}. {sym:<22}", end=end)
    print("\n")


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════

def get_symptoms_from_user() -> list[str]:
    """Interactive symptom collection with validation."""
    show_symptom_list()
    print("  Enter your symptoms separated by commas.")
    print("  Example: fever, cough, fatigue\n")

    raw = input("  Your symptoms: ").strip().lower()
    entered = [s.strip().replace(" ", "_") for s in raw.split(",") if s.strip()]

    # Validate
    valid   = [s for s in entered if s in ALL_SYMPTOMS]
    invalid = [s for s in entered if s not in ALL_SYMPTOMS]

    if invalid:
        print(f"\n  ⚠️  Unrecognised symptoms (ignored): {', '.join(invalid)}")

    if not valid:
        print("\n  ❌  No valid symptoms entered.")
        return []

    print(f"\n  ✅  Valid symptoms recorded: {', '.join(valid)}")
    return valid


def explain_system():
    """Show how the system works — good for viva/lab explanation."""
    print_section("🧠  HOW THIS EXPERT SYSTEM WORKS")
    print("""
  This system uses a TWO-LAYER hybrid approach:

  LAYER 1 — RULE-BASED ENGINE  (Knowledge Representation + Inference)
  ┌────────────────────────────────────────────────────────────┐
  │  • Encodes medical knowledge as IF-THEN rules              │
  │  • Uses FORWARD CHAINING (like Prolog)                     │
  │  • A disease is considered only if ALL required symptoms   │
  │    are present (hard logical constraint)                   │
  │  • Optional symptoms increase the confidence score         │
  └────────────────────────────────────────────────────────────┘

  LAYER 2 — NAIVE BAYES ML CLASSIFIER
  ┌────────────────────────────────────────────────────────────┐
  │  • Treats each symptom as a binary feature (0 or 1)        │
  │  • Learns symptom→disease associations from training data  │
  │  • Outputs PROBABILITY for each disease                    │
  │  • Handles uncertainty and overlapping symptoms            │
  └────────────────────────────────────────────────────────────┘

  HYBRID FUSION:
    Final Score = 0.6 × Rule Confidence + 0.4 × ML Probability

  Why hybrid? Rules are precise but brittle; ML is flexible but
  can misfire. Together they cover more ground.
""")


def run_demo_cases(model, classes):
    """Run 3 preset demo cases to show the system in action."""
    demos = [
        ("Flu",     ["fever", "body_ache", "fatigue", "cough", "chills"]),
        ("Dengue",  ["high_fever", "severe_headache", "joint_pain", "rash", "eye_pain"]),
        ("Migraine",["severe_headache", "nausea", "light_sensitivity", "visual_aura"]),
    ]

    print_section("🎬  DEMO MODE — Pre-set Test Cases")
    for label, symptoms in demos:
        print(f"\n  ━━━  Testing: {label}  ━━━")
        print(f"  Symptoms entered: {', '.join(symptoms)}")
        results = hybrid_diagnosis(symptoms, model, classes)
        display_results(results, top_n=2)
        input("  [Press Enter to continue...]")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print_banner()

    # Train the ML model once at startup
    print("  🔄  Training ML model from knowledge base...")
    model, classes = train_ml_model()
    print("  ✅  Model ready!\n")

    while True:
        print("\n  ══════  MAIN MENU  ══════")
        print("  [1] Diagnose (enter your symptoms)")
        print("  [2] Run demo cases")
        print("  [3] How this system works")
        print("  [4] View all symptoms")
        print("  [5] Exit")
        choice = input("\n  Choose option: ").strip()

        if choice == "1":
            symptoms = get_symptoms_from_user()
            if symptoms:
                results = hybrid_diagnosis(symptoms, model, classes)
                display_results(results, top_n=3)

        elif choice == "2":
            run_demo_cases(model, classes)

        elif choice == "3":
            explain_system()

        elif choice == "4":
            show_symptom_list()

        elif choice == "5":
            print("\n  👋  Thank you for using the Medical Expert System. Stay healthy!\n")
            break

        else:
            print("  ⚠️  Invalid option. Please enter 1–5.")


if __name__ == "__main__":
    main()
