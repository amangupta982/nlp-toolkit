# =============================================================
#  Multi-Task Logistic Regression
#  Task 1: Disease Detection
#  Task 2: Treatment Recommendation
#  Uses: Python, NLTK, Scikit-learn, Pandas
# =============================================================

# ── STEP 1: Install / import libraries ───────────────────────
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
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)
from sklearn.multioutput import MultiOutputClassifier
from scipy.sparse import hstack

# Download required NLTK data (runs once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── STEP 2: Create a sample clinical dataset ─────────────────
#
#  In a real project replace this block with:
#      df = pd.read_csv("your_dataset.csv")
#
#  Recommended free datasets (see README at bottom):
#      • UCI Heart Disease  → disease label
#      • MedNLI / MTSamples → clinical text
#      • Kaggle "Symptom & Disease" datasets

data = {
    "clinical_notes": [
        # Diabetes cases
        "Patient reports frequent urination, excessive thirst, and fatigue. Blood sugar elevated.",
        "High blood glucose levels, weight loss, blurred vision observed in patient.",
        "Increased hunger, slow healing wounds, tingling in feet. Sugar levels high.",
        "Patient has polyuria, polydipsia, unexplained weight loss.",
        "Fatigue, frequent infections, high fasting glucose detected in lab tests.",

        # Heart Disease cases
        "Chest pain radiating to left arm, shortness of breath, sweating.",
        "Patient complains of palpitations, dizziness and high blood pressure.",
        "Severe chest tightness, nausea, irregular heartbeat on ECG.",
        "Hypertension, elevated cholesterol, family history of cardiac arrest.",
        "Breathlessness on exertion, ankle swelling, reduced exercise tolerance.",

        # Flu / Infection cases
        "High fever, body aches, runny nose, and sore throat for 3 days.",
        "Cough, congestion, chills, and fatigue. No known allergies.",
        "Sudden onset of fever, headache, muscle pain, and weakness.",
        "Sneezing, mild fever, nasal discharge, mild cough present.",
        "Patient has viral symptoms: fever, fatigue, loss of appetite.",

        # Hypertension cases
        "Persistent headaches, blurred vision, elevated blood pressure readings.",
        "High BP 160/100, dizziness, occasional nosebleeds.",
        "Patient reports stress-related headaches and consistently high BP.",
        "Swollen legs, shortness of breath, hypertension stage 2.",
        "Chronic hypertension, on medication, reports dizziness and fatigue.",
    ],

    # Task 1 label: Disease
    "disease": [
        "Diabetes",    "Diabetes",    "Diabetes",    "Diabetes",    "Diabetes",
        "HeartDisease","HeartDisease","HeartDisease","HeartDisease","HeartDisease",
        "Flu",         "Flu",         "Flu",         "Flu",         "Flu",
        "Hypertension","Hypertension","Hypertension","Hypertension","Hypertension",
    ],

    # Task 2 label: Treatment
    "treatment": [
        "Insulin/Metformin","Insulin/Metformin","Insulin/Metformin","Insulin/Metformin","Insulin/Metformin",
        "Aspirin/Beta-Blocker","Aspirin/Beta-Blocker","Aspirin/Beta-Blocker","Aspirin/Beta-Blocker","Aspirin/Beta-Blocker",
        "Rest/Antivirals","Rest/Antivirals","Rest/Antivirals","Rest/Antivirals","Rest/Antivirals",
        "ACE-Inhibitors","ACE-Inhibitors","ACE-Inhibitors","ACE-Inhibitors","ACE-Inhibitors",
    ],
}

df = pd.DataFrame(data)
print("=" * 60)
print("  MULTI-TASK LOGISTIC REGRESSION — CLINICAL NLP PROJECT")
print("=" * 60)
print(f"\n📋 Dataset loaded: {len(df)} clinical records")
print(f"   Diseases   : {df['disease'].unique().tolist()}")
print(f"   Treatments : {df['treatment'].unique().tolist()}\n")


# ── STEP 3: Text Pre-processing with NLTK ────────────────────

stemmer   = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    """Lower → remove punctuation → tokenize → remove stopwords → stem."""
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)          # remove non-alpha chars
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean_notes"] = df["clinical_notes"].apply(preprocess)
print("✅ NLTK pre-processing complete.")
print("   Sample cleaned note:")
print(f"   Original : {df['clinical_notes'][0]}")
print(f"   Cleaned  : {df['clean_notes'][0]}\n")


# ── STEP 4: Feature Engineering (TF-IDF) ─────────────────────

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X     = tfidf.fit_transform(df["clean_notes"])

print(f"✅ TF-IDF features created: {X.shape[1]} features\n")


# ── STEP 5: Encode Labels ─────────────────────────────────────

le_disease   = LabelEncoder()
le_treatment = LabelEncoder()

y_disease   = le_disease.fit_transform(df["disease"])
y_treatment = le_treatment.fit_transform(df["treatment"])

# Stack both targets into a single matrix for multi-output learning
Y = np.column_stack([y_disease, y_treatment])


# ── STEP 6: Train / Test Split ───────────────────────────────

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=y_disease
)

print(f"✅ Data split → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")


# ── STEP 7: Train Multi-Task Logistic Regression ─────────────

base_model   = LogisticRegression(max_iter=1000, solver="lbfgs",
                                   multi_class="auto", random_state=42)
multi_model  = MultiOutputClassifier(base_model)
multi_model.fit(X_train, Y_train)

print("✅ Model training complete.\n")


# ── STEP 8: Evaluate Both Tasks ──────────────────────────────

Y_pred = multi_model.predict(X_test)

task_names     = ["Disease Detection", "Treatment Recommendation"]
label_encoders = [le_disease, le_treatment]

for i, (task, le) in enumerate(zip(task_names, label_encoders)):
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


# ── STEP 9: Live Prediction on New Patient Notes ─────────────

def predict_patient(note: str) -> None:
    """Given a raw clinical note, predict disease and treatment."""
    cleaned  = preprocess(note)
    features = tfidf.transform([cleaned])
    preds    = multi_model.predict(features)[0]

    disease   = le_disease.inverse_transform([preds[0]])[0]
    treatment = le_treatment.inverse_transform([preds[1]])[0]

    print(f"\n{'='*55}")
    print(f"  PATIENT NOTE  : {note}")
    print(f"  Detected Disease   → {disease}")
    print(f"  Recommended Rx     → {treatment}")
    print(f"{'='*55}")


# ── Test with new patient notes ──────────────────────────────
print("\n📌 LIVE PREDICTIONS ON NEW PATIENTS\n")

predict_patient(
    "Patient has blurred vision, frequent urination, extreme fatigue and high glucose."
)
predict_patient(
    "Severe chest pain, shortness of breath, irregular heartbeat. ECG abnormal."
)
predict_patient(
    "Fever, cough, body aches, and runny nose since yesterday."
)
predict_patient(
    "Persistent high blood pressure 170/110, headaches, and swollen ankles."
)


# ── STEP 10: Feature Importance (Top keywords per disease) ───

print("\n\n📊 TOP KEYWORDS PER DISEASE (from Logistic Regression weights)\n")

feature_names = tfidf.get_feature_names_out()
disease_clf   = multi_model.estimators_[0]    # first task = disease

for idx, class_name in enumerate(le_disease.classes_):
    coef      = disease_clf.coef_[idx]
    top_idx   = np.argsort(coef)[-8:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"  {class_name:<15} → {', '.join(top_words)}")

print("\n✅ Project complete! \n")


# =============================================================
#  📚 RECOMMENDED FREE DATASETS
# =============================================================
#
#  1. UCI Heart Disease Dataset
#     https://archive.ics.uci.edu/ml/datasets/heart+Disease
#
#  2. MTSamples – Real Clinical Notes (NLP)
#     https://www.mtsamples.com/
#     https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
#
#  3. Symptom → Disease Dataset (Kaggle)
#     https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
#
#  4. MIMIC-III (requires free registration)
#     https://physionet.org/content/mimiciii/
#
#  5. MedNLI – Clinical NLI dataset
#     https://jgc128.github.io/mednli/
#
# =============================================================
