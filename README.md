<div align="center">

```
╔══════════════════════════════════════════════════════════════════════╗
                                                                       
        🏥  MEDICAL DIAGNOSIS EXPERT SYSTEM  v1.0                     

Knowledge-Based Reasoning  ×  Machine Learning

╚══════════════════════════════════════════════════════════════════════╝
```

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![AI](https://img.shields.io/badge/AI-Expert%20System-8B5CF6?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational-22C55E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-EC4899?style=for-the-badge)

<br/>

> **A hybrid AI system that combines rule-based logical inference with a Naive Bayes ML classifier to diagnose diseases from patient symptoms — built for AI lab assignments covering CO2, CO4, CO5, and CO6.**

<br/>

---

</div>

<br/>

## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [What This Project Does](#-what-this-project-does) |
| 2 | [Why Hybrid? The Core Idea](#-why-hybrid-the-core-idea) |
| 3 | [System Architecture](#-system-architecture) |
| 4 | [AI Concepts Covered](#-ai-concepts-covered) |
| 5 | [Knowledge Base Design](#-knowledge-base-design) |
| 6 | [Rule-Based Engine — How It Works](#-rule-based-engine--how-it-works) |
| 7 | [ML Classifier — How It Works](#-ml-classifier--how-it-works) |
| 8 | [Hybrid Fusion Formula](#-hybrid-fusion-formula) |
| 9 | [Supported Diseases](#-supported-diseases) |
| 10 | [All Symptoms Reference](#-all-symptoms-reference) |
| 11 | [Installation & Running](#-installation--running) |
| 12 | [Sample Output](#-sample-output) |
| 13 | [Project Structure](#-project-structure) |
| 14 | [Disclaimer](#%EF%B8%8F-disclaimer) |

<br/>

---

## 🔬 What This Project Does

This is a **Medical Expert System** — a classic AI application that mimics the reasoning of a doctor. When you enter your symptoms, the system:

1. **Applies medical rules** — checks which diseases match your symptoms logically
2. **Runs a trained ML model** — computes probability scores using Naive Bayes
3. **Fuses both results** — produces a ranked diagnosis with confidence scores and medical advice

It's not just a lookup table. It *reasons* — the same way a real expert system would.

<br/>

---

## 💡 Why Hybrid? The Core Idea

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PROBLEM WITH RULES ALONE:                                     │
│   Very precise, but brittle. If one required symptom is         │
│   missing, the entire rule fails — even if 9/10 symptoms match. │
│                                                                 │
│   PROBLEM WITH ML ALONE:                                        │
│   Flexible and probabilistic, but can misfire on small/         │
│   synthetic datasets. No hard medical constraints.              │
│                                                                 │
│   HYBRID SOLUTION:                                              │
│   Rules act as a hard medical filter.                           │
│   ML fills the probabilistic gap.                               │
│   Together → more accurate, more robust.                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────┐
                    │    Patient Symptoms       │
                    │  (fever, cough, fatigue)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │      Knowledge Base       │
                    │  10 diseases × rules +    │
                    │  required & optional syms │
                    └──────┬──────────┬─────────┘
                           │          │
              ┌────────────▼──┐    ┌──▼────────────────┐
              │  Rule-Based   │    │   Naive Bayes ML   │
              │    Engine     │    │    Classifier      │
              │               │    │                    │
              │ Forward chain │    │ Symptom vector     │
              │ IF required   │    │ → P(disease|syms)  │
              │ THEN fire     │    │                    │
              │               │    │ Trained on KB data │
              │ Confidence %  │    │ Probability %      │
              └──────┬────────┘    └──────┬─────────────┘
                     │                    │
                     │   60%        40%   │
                     └──────────┬─────────┘
                                │
                   ┌────────────▼─────────────┐
                   │      Hybrid Fusion        │
                   │                           │
                   │  Score = 0.6×Rule +       │
                   │          0.4×ML_Prob      │
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │    Ranked Diagnosis       │
                   │                           │
                   │  #1 Disease — 87.4%  🔴   │
                   │  #2 Disease — 61.2%  🟡   │
                   │  #3 Disease — 34.0%  🟢   │
                   │                           │
                   │  + Medical Advice         │
                   └──────────────────────────┘
```

<br/>

---

## 🧠 AI Concepts Covered

This project is specifically designed to cover **4 course outcomes** in one system:

<br/>

### ✅ CO2 — Knowledge Representation

The **Knowledge Base** is how the system stores medical expertise. Instead of hardcoding `if fever and cough: print("flu")`, we represent knowledge as structured data:

```python
KNOWLEDGE_BASE = {
    "Flu": {
        "required":  ["fever", "body_ache", "fatigue"],   # must ALL be present
        "optional":  ["cough", "headache", "chills"],     # boost confidence score
        "severity":  "moderate",
        "advice":    "Rest, stay hydrated, take paracetamol..."
    },
    ...
}
```

This is **declarative knowledge representation** — the system knows *what* is true about diseases without needing explicit procedural code for each one. Adding a new disease = adding one dictionary entry.

<br/>

### ✅ CO6 — Logical Inference (Forward Chaining)

The rule engine uses **forward chaining** — starting from known facts (symptoms) and deriving conclusions (diagnoses):

```
FACTS:    patient has {fever, cough, loss_of_smell, fatigue}

RULE:     IF fever AND cough → COVID-19 rule fires
          → confidence = matched_symptoms / total_symptoms × 100

INFERENCE: COVID-19 = 75% confidence  ✓ rule fired
           Flu      = 62% confidence  ✓ rule fired
           Dengue   = 0%              ✗ rule didn't fire (missing joint_pain, high_fever)
```

This is the same logic used in Prolog-based expert systems. The rule only "fires" when **all required symptoms** are present — a hard logical constraint.

<br/>

### ✅ CO4 — Machine Learning

The **Naive Bayes classifier** treats symptoms as binary features and learns which combinations point to which disease:

```
Feature vector for "fever + cough + fatigue":
[0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
 ↑                ↑              ↑
 abdominal_pain  cough          fatigue
 (absent=0)      (present=1)    (present=1)

P(COVID-19 | fever, cough, fatigue) → 0.43
P(Flu      | fever, cough, fatigue) → 0.38
P(Malaria  | fever, cough, fatigue) → 0.11
...
```

Training data is auto-generated from the Knowledge Base — each disease produces 3 synthetic training samples (all symptoms, required only, required + partial optional).

<br/>

### ✅ CO5 — Expert System Design

The full system is an **Expert System** with all three classical components:

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Knowledge Base │    │ Inference Engine │    │  User Interface  │
│                 │    │                 │    │                  │
│ • Disease rules │───▶│ • Forward chain │───▶│ • CLI menu       │
│ • Symptom maps  │    │ • Hybrid fusion │    │ • Demo mode      │
│ • Severity info │    │ • Ranking logic │    │ • Explanation    │
│ • Medical advice│    │ • ML + Rules    │    │ • Advice output  │
└─────────────────┘    └─────────────────┘    └──────────────────┘
```

<br/>

---

## 📚 Knowledge Base Design

The knowledge base encodes 10 diseases. Here's the full schema for each entry:

```python
"Disease Name": {
    "required": [...],   # ALL must be present for the rule to fire
    "optional": [...],   # each present one increases confidence score
    "severity": "mild" | "moderate" | "high",
    "advice":   "plain English medical advice string"
}
```

### Confidence Scoring Formula:

```
                  matched_required + matched_optional
confidence  =  ─────────────────────────────────────  × 100
                total_required + total_optional
```

Example — Flu with symptoms `{fever, body_ache, fatigue, cough}`:

```
required matched: 3/3  (fever ✓, body_ache ✓, fatigue ✓)
optional matched: 1/4  (cough ✓, headache ✗, chills ✗, sore_throat ✗)

confidence = (3 + 1) / (3 + 4) × 100 = 57.1%
```

<br/>

---

## ⚙️ Rule-Based Engine — How It Works

```python
def rule_based_diagnosis(patient_symptoms):
    patient_set = set(patient_symptoms)

    for disease, rules in KNOWLEDGE_BASE.items():
        required = set(rules["required"])
        optional = set(rules["optional"])

        # FORWARD CHAINING: rule fires only if ALL required symptoms present
        rule_fired = required.issubset(patient_set)

        if rule_fired:
            matched_optional = len(optional & patient_set)
            confidence = (len(required) + matched_optional) / (len(required) + len(optional)) * 100
            # → add to results
```

**Step-by-step trace** for input `[fever, chills, headache, nausea]`:

```
Checking Flu      → required: {fever, body_ache, fatigue} → body_ache MISSING → ✗ skip
Checking Malaria  → required: {fever, chills}             → ALL present       → ✓ FIRES
                   optional matched: headache ✓, nausea ✓ (sweating ✗, vomiting ✗, body_ache ✗)
                   confidence = (2+2)/(2+5) × 100 = 57.1%
Checking COVID-19 → required: {fever, cough}              → cough MISSING     → ✗ skip
...
```

<br/>

---

## 🤖 ML Classifier — How It Works

### Step 1 — Build Training Data

Auto-generated from the Knowledge Base. Each disease produces 3 training rows:

| Sample | What's included | Label |
|--------|----------------|-------|
| Full | required + all optional symptoms | Disease |
| Minimal | required symptoms only | Disease |
| Partial | required + first half of optional | Disease |

### Step 2 — Encode as Binary Vector

```
ALL_SYMPTOMS (sorted alphabetically, 32 features):
['abdominal_pain', 'blurred_vision', 'body_ache', 'chest_pain',
 'chest_tightness', 'chills', 'constipation', 'cough', ...]

Input: [fever, cough, fatigue]

Vector: [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...]
                                ↑  ↑     ↑
                              cough fatigue fever
```

### Step 3 — Naive Bayes Prediction

```
P(disease | symptoms) ∝ P(symptoms | disease) × P(disease)

For each disease:
  P(Flu | fever,cough,fatigue) = P(fever|Flu) × P(cough|Flu) × P(fatigue|Flu) × P(Flu)

Returns probability distribution over all 10 diseases.
```

<br/>

---

## 🔀 Hybrid Fusion Formula

```
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   Hybrid Score = 0.6 × Rule_Confidence              ║
║                + 0.4 × ML_Probability               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

**Why 60/40?**
- Rule confidence is medically grounded (hard constraints), so it gets higher trust
- ML probability handles uncertainty and borderline cases, so it gets a supporting role
- If a disease has 0% rule confidence (rule didn't fire), ML can still surface it as a secondary suggestion

**Example fusion:**

| Disease | Rule Conf | ML Prob | Hybrid Score |
|---------|-----------|---------|-------------|
| Malaria | 57.1% | 48.3% | **0.6×57.1 + 0.4×48.3 = 53.6%** |
| Flu | 0% (didn't fire) | 22.1% | **0.6×0 + 0.4×22.1 = 8.8%** |
| Dengue | 0% | 15.4% | **0.6×0 + 0.4×15.4 = 6.2%** |

<br/>

---

## 🦠 Supported Diseases

| # | Disease | Severity | Key Required Symptoms |
|---|---------|----------|-----------------------|
| 1 | 🤧 **Flu** | 🟡 Moderate | fever, body_ache, fatigue |
| 2 | 🤒 **Common Cold** | 🟢 Mild | runny_nose, sneezing |
| 3 | 😷 **COVID-19** | 🔴 High | fever, cough |
| 4 | 🦟 **Malaria** | 🔴 High | fever, chills |
| 5 | 🌡️ **Typhoid** | 🔴 High | prolonged_fever, abdominal_pain |
| 6 | 🧠 **Migraine** | 🟡 Moderate | severe_headache |
| 7 | 🦷 **Dengue** | 🔴 High | high_fever, severe_headache, joint_pain |
| 8 | 🤢 **Gastroenteritis** | 🟡 Moderate | diarrhea, nausea |
| 9 | 💓 **Hypertension** | 🔴 High | headache, dizziness |
| 10 | 💨 **Asthma** | 🟡 Moderate | shortness_of_breath, wheezing |

<br/>

---

## 💊 All Symptoms Reference

The system recognises **32 symptoms** total:

```
abdominal_pain      blurred_vision      body_ache           chest_pain
chest_tightness     chills              constipation        cough
diarrhea            dizziness           eye_pain            fatigue
headache            high_fever          joint_pain          light_sensitivity
loss_of_appetite    loss_of_smell       loss_of_taste       mild_fever
nausea              nosebleed           prolonged_fever     rash
runny_nose          severe_headache     shortness_of_breath sneezing
sore_throat         sound_sensitivity   sweating            visual_aura
vomiting            wheezing
```

> 💡 When entering symptoms, use **underscores** for multi-word symptoms: `body_ache`, `loss_of_smell`, `severe_headache`

<br/>

---

## 🚀 Installation & Running

### Prerequisites

```bash
Python 3.10+
scikit-learn
```

### Install Dependencies

```bash
pip install scikit-learn
```

### Run the System

```bash
python medical_diagnosis_expert.py
```

### Windows Terminal Fix (if you see encoding errors)

Run this **before** the script in your terminal:

```bash
chcp 65001
python medical_diagnosis_expert.py
```

Or add these 3 lines at the very top of the `.py` file:

```python
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

<br/>

---

## 🖥️ Sample Output

```
╔══════════════════════════════════════════════════════════════════╗
║        🏥  MEDICAL DIAGNOSIS EXPERT SYSTEM  v1.0               ║
║        Knowledge-Based + Machine Learning Hybrid                ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  DISCLAIMER: For educational purposes only.                 ║
╚══════════════════════════════════════════════════════════════════╝

  🔄  Training ML model from knowledge base...
  ✅  Model ready!

  ══════  MAIN MENU  ══════
  [1] Diagnose (enter your symptoms)
  [2] Run demo cases
  [3] How this system works
  [4] View all symptoms
  [5] Exit

  Choose option: 1

  Your symptoms: fever, cough, loss_of_smell, fatigue

  ✅  Valid symptoms recorded: fever, cough, loss_of_smell, fatigue

──────────────────────────────────────────────────────────────
  📊  DIAGNOSIS RESULTS  (Top Matches)
──────────────────────────────────────────────────────────────

  [1] COVID-19  🔴 HIGH RISK
      ┌─────────────────────────────────────────┐
      │  Hybrid Score  :  74.3%               │
      │  Rule Engine   :  44.4%  ✅ Rule Matched│
      │  ML (Naive Bayes):  38.1%              │
      └─────────────────────────────────────────┘
      💊 Advice: Isolate immediately. Get tested. Seek emergency
                 care if breathing is difficult.

  [2] Flu  🟡 MODERATE
      ┌─────────────────────────────────────────┐
      │  Hybrid Score  :  38.2%               │
      │  Rule Engine   :  57.1%  ✅ Rule Matched│
      │  ML (Naive Bayes):  32.0%              │
      └─────────────────────────────────────────┘
      💊 Advice: Rest, stay hydrated, take paracetamol.

  [3] Common Cold  🟢 MILD
      ┌─────────────────────────────────────────┐
      │  Hybrid Score  :  12.4%               │
      │  Rule Engine   :   0.0%  🤖 ML Predicted│
      │  ML (Naive Bayes):  18.1%              │
      └─────────────────────────────────────────┘
      💊 Advice: Rest, warm fluids, vitamin C.
```

<br/>

---

## 📁 Project Structure

```
medical-diagnosis-expert-system/
│
├── medical_diagnosis_expert.py     ← main file (everything in one script)
│
└── README.md                       ← this file
```

### Inside `medical_diagnosis_expert.py`:

```
Module 1 — KNOWLEDGE_BASE          (lines ~40–100)
           Disease rules, symptoms, severity, advice

Module 2 — rule_based_diagnosis()  (lines ~115–155)
           Forward chaining inference engine

Module 3 — train_ml_model()        (lines ~160–200)
           Naive Bayes training on KB-generated data

Module 4 — hybrid_diagnosis()      (lines ~205–240)
           Fusion of rule + ML scores

Module 5 — Display helpers         (lines ~245–285)
           Pretty-printing, banners, menus

Module 6 — main()                  (lines ~290–end)
           Interactive CLI with menu
```

<br/>

---

## 🎓 Course Outcome Mapping

| Course Outcome | Concept | Where in Code |
|----------------|---------|---------------|
| **CO2** | Knowledge Representation | `KNOWLEDGE_BASE` dictionary — declarative disease-symptom encoding |
| **CO4** | Machine Learning | `train_ml_model()`, `ml_diagnosis()` — Naive Bayes classifier |
| **CO5** | Expert System Design | Full system: KB + Inference Engine + UI |
| **CO6** | Logical Inference | `rule_based_diagnosis()` — forward chaining, rule firing |

<br/>

---

## ⚡ Quick Viva Answers

**Q: Why Naive Bayes for this problem?**
> Naive Bayes works well with binary features (symptom present/absent). It's fast, interpretable, and performs surprisingly well even with small training datasets — ideal for a symptom-based classifier.

**Q: What is forward chaining?**
> Starting from known facts (symptoms) and applying rules to derive new conclusions (diseases). The opposite of backward chaining, which starts from a hypothesis and tries to prove it.

**Q: Why not use a neural network instead?**
> For a 10-disease, 32-symptom system, Naive Bayes is more appropriate — neural networks need large datasets to generalize, and they're not interpretable. Expert systems prioritize explainability.

**Q: What's the difference between required and optional symptoms?**
> Required symptoms are the diagnostic criteria — a disease can't be ruled in without them. Optional symptoms add confidence but aren't mandatory. This mirrors real clinical diagnosis.

<br/>

---

## ⚠️ Disclaimer

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   This project is built for EDUCATIONAL PURPOSES ONLY.         │
│                                                                 │
│   It is NOT a medical device, NOT a clinical tool, and          │
│   should NEVER be used to make real health decisions.           │
│                                                                 │
│   Always consult a qualified medical professional               │
│   for any health concerns.                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

<div align="center">

**Built for AI Lab Assignment · Knowledge Representation · Logical Inference · Machine Learning · Expert Systems**

```
Made with 🧠 + ☕ for CSA / AI coursework
```

</div>
