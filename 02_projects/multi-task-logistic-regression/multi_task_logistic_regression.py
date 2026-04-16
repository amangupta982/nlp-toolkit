# =============================================================
#  Multi-Task Logistic Regression
#  Task 1: Disease Detection
#  Task 2: Treatment Recommendation
#  Uses: Python, NLTK, Scikit-learn, Pandas
# =============================================================

import pandas as pd
import numpy as np
import nltk
import re
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# =============================================================
#  STEP 1 — LARGE BUILT-IN DATASET (100 records, 4 diseases)
# =============================================================

data = {
    "clinical_notes": [

        # ── DIABETES (25 notes) ──────────────────────────────
        "Patient reports frequent urination, excessive thirst, and fatigue. Blood sugar elevated.",
        "High blood glucose levels, weight loss, blurred vision observed in patient.",
        "Increased hunger, slow healing wounds, tingling in feet. Sugar levels high.",
        "Patient has polyuria, polydipsia, unexplained weight loss.",
        "Fatigue, frequent infections, high fasting glucose detected in lab tests.",
        "Patient complains of numbness in hands and feet, high blood sugar level.",
        "Recurring urinary tract infections, elevated HbA1c, dry mouth reported.",
        "Blurred vision, frequent urination at night, extreme thirst and hunger.",
        "Patient has type 2 diabetes symptoms: weight gain, fatigue, high glucose.",
        "Slow wound healing on foot, elevated glucose, patient feels tired all day.",
        "Blood sugar 280 mg/dL, patient reports dizziness and excessive hunger.",
        "Tingling sensation in legs, frequent urination, elevated fasting sugar.",
        "Patient diagnosed with prediabetes, complains of fatigue and thirst.",
        "High sugar in urine, weight loss despite eating well, blurred vision.",
        "Neuropathy symptoms: burning feet, high blood glucose, poor circulation.",
        "HbA1c 9.5%, patient has fatigue, frequent thirst, and blurred eyesight.",
        "Diabetes follow-up: sugar uncontrolled, patient has polyuria and fatigue.",
        "Sweet smell in breath, high glucose, patient feels weak and thirsty.",
        "Recurring skin infections, elevated blood sugar, slow healing cuts.",
        "Patient has gestational diabetes signs: high glucose, fatigue, thirst.",
        "Numbness in toes, high fasting glucose, increased urination at night.",
        "Unexplained weight loss, polydipsia, polyphagia and glucose 300 mg/dL.",
        "Patient complains of dry itchy skin, high blood sugar and frequent urination.",
        "Fatigue after meals, sugar spikes, tingling in hands and feet noted.",
        "Blood sugar elevated to 350, patient confused, excessive thirst reported.",

        # ── HEART DISEASE (25 notes) ─────────────────────────
        "Chest pain radiating to left arm, shortness of breath, sweating.",
        "Patient complains of palpitations, dizziness and high blood pressure.",
        "Severe chest tightness, nausea, irregular heartbeat on ECG.",
        "Hypertension, elevated cholesterol, family history of cardiac arrest.",
        "Breathlessness on exertion, ankle swelling, reduced exercise tolerance.",
        "Sharp chest pain, cold sweats, pain radiating to jaw and left shoulder.",
        "Patient has arrhythmia, irregular pulse, dizziness and chest discomfort.",
        "Elevated troponin levels, ECG changes, chest pressure and sweating.",
        "Angina symptoms: squeezing chest pain during exercise, relieved by rest.",
        "Heart palpitations, shortness of breath at night, swollen feet.",
        "Patient has history of MI, reports chest pressure and left arm tingling.",
        "Severe breathlessness, chest heaviness, heart rate 110 bpm on monitor.",
        "Sudden onset chest pain, nausea, vomiting, ECG shows ST elevation.",
        "Chronic chest pain, high LDL cholesterol, family history of heart attack.",
        "Dizziness, fainting episodes, low ejection fraction on echocardiogram.",
        "Patient reports racing heartbeat, chest discomfort, and anxiety.",
        "Bilateral ankle swelling, dyspnea at rest, reduced cardiac output.",
        "Chest tightness after climbing stairs, shortness of breath, fatigue.",
        "Atrial fibrillation detected, irregular heartbeat, dizziness and fatigue.",
        "Patient has coronary artery disease, chest pain on exertion, high BP.",
        "Left ventricular hypertrophy, breathlessness, chest pain, high BP.",
        "Pericarditis symptoms: sharp stabbing chest pain, worse when lying down.",
        "Cardiomyopathy: fatigue, shortness of breath, swollen legs and abdomen.",
        "Heart failure symptoms: breathlessness, leg swelling, inability to exercise.",
        "Troponin elevated, patient has chest pain, sweating and vomiting.",

        # ── FLU / INFECTION (25 notes) ───────────────────────
        "High fever, body aches, runny nose, and sore throat for 3 days.",
        "Cough, congestion, chills, and fatigue. No known allergies.",
        "Sudden onset of fever, headache, muscle pain, and weakness.",
        "Sneezing, mild fever, nasal discharge, mild cough present.",
        "Patient has viral symptoms: fever, fatigue, loss of appetite.",
        "High temperature 103F, chills, sweating, body ache and sore throat.",
        "Flu-like illness: runny nose, cough, headache, fatigue and fever.",
        "Patient reports loss of taste and smell, fever and dry persistent cough.",
        "Severe body pain, high fever, vomiting and fatigue for 2 days.",
        "Respiratory infection: productive cough, fever, chest tightness.",
        "Influenza symptoms: sudden high fever, myalgia, headache, dry cough.",
        "Patient has tonsillitis: sore throat, difficulty swallowing, fever.",
        "Viral pharyngitis: red throat, mild fever, runny nose, fatigue.",
        "Bronchitis symptoms: persistent cough, low-grade fever, chest congestion.",
        "Patient reports fever 101F, body aches, nasal congestion and fatigue.",
        "Seasonal flu: sneezing, coughing, runny nose, fever and muscle aches.",
        "Post-viral fatigue, mild cough, low-grade fever persisting for a week.",
        "Severe headache, high fever, stiff neck, sensitivity to light.",
        "Acute upper respiratory infection: nasal discharge, cough, mild fever.",
        "Patient has sinus infection: facial pain, nasal congestion, fever.",
        "COVID symptoms: fever, dry cough, fatigue, loss of smell and taste.",
        "Viral gastroenteritis: fever, nausea, vomiting, diarrhea and body pain.",
        "Strep throat: severe sore throat, fever 102F, difficulty swallowing.",
        "Ear infection: earache, fever, headache and nasal congestion.",
        "Pneumonia: high fever, chills, productive cough, shortness of breath.",

        # ── HYPERTENSION (25 notes) ──────────────────────────
        "Persistent headaches, blurred vision, elevated blood pressure readings.",
        "High BP 160/100, dizziness, occasional nosebleeds.",
        "Patient reports stress-related headaches and consistently high BP.",
        "Swollen legs, shortness of breath, hypertension stage 2.",
        "Chronic hypertension, on medication, reports dizziness and fatigue.",
        "Blood pressure 170/110, patient has throbbing headaches and chest pain.",
        "Stage 1 hypertension: BP 145/95, occasional dizziness and headaches.",
        "Patient has white coat hypertension, consistently high readings at clinic.",
        "Hypertensive crisis: BP 200/130, severe headache, visual disturbances.",
        "Secondary hypertension due to kidney disease, high BP and fatigue.",
        "High blood pressure, pounding heartbeat, shortness of breath on exertion.",
        "Uncontrolled hypertension, patient on 3 medications, BP still elevated.",
        "Malignant hypertension: papilledema, BP 210/140, severe headache.",
        "Patient has essential hypertension, family history, BP 155/100.",
        "Resistant hypertension: BP remains high despite multiple medications.",
        "High BP 165/105, dizziness on standing, blurred vision in mornings.",
        "Hypertension with left ventricular hypertrophy, shortness of breath.",
        "Isolated systolic hypertension: BP 170/80, headache, fatigue.",
        "Patient reports morning headaches, BP spikes, nosebleeds at night.",
        "Hypertension follow-up: BP 150/95, patient has palpitations and fatigue.",
        "Elevated BP 158/102, patient anxious, headache and neck stiffness.",
        "Hypertension stage 2, proteinuria detected, on ACE inhibitor therapy.",
        "High BP, renal involvement suspected, patient has headaches and edema.",
        "Persistent high BP despite lifestyle changes, fatigue and dizziness.",
        "Patient BP 180/115, emergency visit, severe headache and confusion.",
    ],

    "disease": (
        ["Diabetes"]     * 25 +
        ["HeartDisease"] * 25 +
        ["Flu"]          * 25 +
        ["Hypertension"] * 25
    ),

    "treatment": (
        ["Insulin/Metformin"]    * 25 +
        ["Aspirin/Beta-Blocker"] * 25 +
        ["Rest/Antivirals"]      * 25 +
        ["ACE-Inhibitors"]       * 25
    ),
}

df = pd.DataFrame(data)

print("=" * 60)
print("  MULTI-TASK LOGISTIC REGRESSION — CLINICAL NLP PROJECT")
print("=" * 60)
print(f"\n📋 Dataset loaded  : {len(df)} clinical records")
print(f"   Diseases        : {df['disease'].unique().tolist()}")
print(f"   Treatments      : {df['treatment'].unique().tolist()}\n")


# =============================================================
#  STEP 2 — NLTK PREPROCESSING
# =============================================================

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean_notes"] = df["clinical_notes"].apply(preprocess)

print("✅ NLTK preprocessing done.")
print(f"   Original : {df['clinical_notes'][0]}")
print(f"   Cleaned  : {df['clean_notes'][0]}\n")


# =============================================================
#  STEP 3 — TF-IDF FEATURES
# =============================================================

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X     = tfidf.fit_transform(df["clean_notes"])
print(f"✅ TF-IDF features  : {X.shape[1]} features\n")


# =============================================================
#  STEP 4 — ENCODE LABELS
# =============================================================

le_disease   = LabelEncoder()
le_treatment = LabelEncoder()

y_disease   = le_disease.fit_transform(df["disease"])
y_treatment = le_treatment.fit_transform(df["treatment"])
Y           = np.column_stack([y_disease, y_treatment])


# =============================================================
#  STEP 5 — TRAIN / TEST SPLIT
# =============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=y_disease
)
print(f"✅ Data split → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")


# =============================================================
#  STEP 6 — TRAIN MODEL
# =============================================================

base_model  = LogisticRegression(max_iter=2000, solver="lbfgs",
                                  C=5.0, random_state=42)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, Y_train)
print("✅ Model training complete.\n")


# =============================================================
#  STEP 7 — EVALUATE BOTH TASKS
# =============================================================

Y_pred = multi_model.predict(X_test)

task_names = ["Disease Detection", "Treatment Recommendation"]
encoders   = [le_disease, le_treatment]

for i, (task, le) in enumerate(zip(task_names, encoders)):
    y_true = Y_test[:, i]
    y_pred = Y_pred[:, i]
    acc    = accuracy_score(y_true, y_pred)

    print(f"{'─'*55}")
    print(f"  TASK {i+1}: {task}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc * 100:.1f}%\n")
    print(classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))


# =============================================================
#  STEP 8 — LIVE PREDICTIONS
# =============================================================

def predict_patient(note: str) -> None:
    cleaned  = preprocess(note)
    features = tfidf.transform([cleaned])
    preds    = multi_model.predict(features)[0]
    proba_d  = multi_model.estimators_[0].predict_proba(features)[0]

    disease   = le_disease.inverse_transform([preds[0]])[0]
    treatment = le_treatment.inverse_transform([preds[1]])[0]
    conf      = max(proba_d) * 100

    print(f"\n{'='*55}")
    print(f"  NOTE      : {note}")
    print(f"  Disease   → {disease}  (confidence: {conf:.1f}%)")
    print(f"  Treatment → {treatment}")
    print(f"{'='*55}")


print("\n📌 LIVE PREDICTIONS ON NEW PATIENTS\n")
predict_patient("Patient has blurred vision, frequent urination, extreme fatigue and high glucose.")
predict_patient("Severe chest pain, shortness of breath, irregular heartbeat. ECG abnormal.")
predict_patient("Fever, cough, body aches, and runny nose since yesterday.")
predict_patient("Persistent high blood pressure 170/110, headaches, and swollen ankles.")


# =============================================================
#  STEP 9 — TOP KEYWORDS PER DISEASE
# =============================================================

print("\n\n📊 TOP KEYWORDS PER DISEASE\n")
feature_names = tfidf.get_feature_names_out()
disease_clf   = multi_model.estimators_[0]

for idx, class_name in enumerate(le_disease.classes_):
    coef      = disease_clf.coef_[idx]
    top_idx   = np.argsort(coef)[-8:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"  {class_name:<15} → {', '.join(top_words)}")

print("\n✅ Project complete!\n")