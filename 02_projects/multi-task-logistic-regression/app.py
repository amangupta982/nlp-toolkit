# =============================================================
#  Multi-Task Logistic Regression — Streamlit App
#  Run with:  streamlit run app.py
#  Install :  pip install streamlit nltk scikit-learn pandas numpy matplotlib seaborn
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical NLP — Multi-Task LR",
    page_icon="🏥",
    layout="wide",
)

# ── Download NLTK data ────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

# ── Dataset ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        "clinical_notes": [
            "Patient reports frequent urination, excessive thirst, and fatigue. Blood sugar elevated.",
            "High blood glucose levels, weight loss, blurred vision observed in patient.",
            "Increased hunger, slow healing wounds, tingling in feet. Sugar levels high.",
            "Patient has polyuria, polydipsia, unexplained weight loss.",
            "Fatigue, frequent infections, high fasting glucose detected in lab tests.",
            "Chest pain radiating to left arm, shortness of breath, sweating.",
            "Patient complains of palpitations, dizziness and high blood pressure.",
            "Severe chest tightness, nausea, irregular heartbeat on ECG.",
            "Hypertension, elevated cholesterol, family history of cardiac arrest.",
            "Breathlessness on exertion, ankle swelling, reduced exercise tolerance.",
            "High fever, body aches, runny nose, and sore throat for 3 days.",
            "Cough, congestion, chills, and fatigue. No known allergies.",
            "Sudden onset of fever, headache, muscle pain, and weakness.",
            "Sneezing, mild fever, nasal discharge, mild cough present.",
            "Patient has viral symptoms: fever, fatigue, loss of appetite.",
            "Persistent headaches, blurred vision, elevated blood pressure readings.",
            "High BP 160/100, dizziness, occasional nosebleeds.",
            "Patient reports stress-related headaches and consistently high BP.",
            "Swollen legs, shortness of breath, hypertension stage 2.",
            "Chronic hypertension, on medication, reports dizziness and fatigue.",
        ],
        "disease": [
            "Diabetes","Diabetes","Diabetes","Diabetes","Diabetes",
            "Heart Disease","Heart Disease","Heart Disease","Heart Disease","Heart Disease",
            "Flu","Flu","Flu","Flu","Flu",
            "Hypertension","Hypertension","Hypertension","Hypertension","Hypertension",
        ],
        "treatment": [
            "Insulin / Metformin","Insulin / Metformin","Insulin / Metformin","Insulin / Metformin","Insulin / Metformin",
            "Aspirin / Beta-Blocker","Aspirin / Beta-Blocker","Aspirin / Beta-Blocker","Aspirin / Beta-Blocker","Aspirin / Beta-Blocker",
            "Rest / Antivirals","Rest / Antivirals","Rest / Antivirals","Rest / Antivirals","Rest / Antivirals",
            "ACE Inhibitors","ACE Inhibitors","ACE Inhibitors","ACE Inhibitors","ACE Inhibitors",
        ],
    }
    return pd.DataFrame(data)

# ── NLTK preprocessing ────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = load_data()
    df["clean_notes"] = df["clinical_notes"].apply(preprocess)

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X     = tfidf.fit_transform(df["clean_notes"])

    le_d = LabelEncoder()
    le_t = LabelEncoder()
    y_d  = le_d.fit_transform(df["disease"])
    y_t  = le_t.fit_transform(df["treatment"])
    Y    = np.column_stack([y_d, y_t])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=y_d
    )

    base  = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model = MultiOutputClassifier(base)
    model.fit(X_train, Y_train)

    return model, tfidf, le_d, le_t, X_test, Y_test, df

model, tfidf, le_disease, le_treatment, X_test, Y_test, df = train_model()

# ── Colour map for diseases ───────────────────────────────────
DISEASE_COLORS = {
    "Diabetes":      "#378ADD",
    "Heart Disease": "#E24B4A",
    "Flu":           "#639922",
    "Hypertension":  "#BA7517",
}
DISEASE_ICONS = {
    "Diabetes":      "🩸",
    "Heart Disease": "❤️",
    "Flu":           "🤧",
    "Hypertension":  "💊",
}

# =============================================================
#  SIDEBAR
# =============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=64)
    st.title("🏥 Clinical NLP")
    st.caption("Multi-Task Logistic Regression")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🔬 Predict Patient", "📊 Model Performance", "📋 Dataset Explorer", "📚 How It Works"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python · NLTK · Scikit-learn · Streamlit")

# =============================================================
#  PAGE 1 — PREDICT PATIENT
# =============================================================
if page == "🔬 Predict Patient":
    st.title("🔬 Patient Diagnosis & Treatment")
    st.write("Enter a clinical note below and the model will predict the disease and recommend treatment.")

    # Sample notes
    samples = {
        "Select a sample note...": "",
        "🩸 Diabetic patient":     "Patient has blurred vision, frequent urination, extreme fatigue and high glucose levels.",
        "❤️ Heart patient":        "Severe chest pain, shortness of breath, irregular heartbeat. ECG abnormal.",
        "🤧 Flu patient":          "Fever, cough, body aches, and runny nose since yesterday.",
        "💊 Hypertension patient": "Persistent high blood pressure 170/110, headaches, and swollen ankles.",
    }

    chosen = st.selectbox("Quick-fill with a sample note", list(samples.keys()))
    note   = st.text_area(
        "Clinical Note",
        value=samples[chosen],
        height=120,
        placeholder="e.g. Patient reports chest pain, shortness of breath and high blood pressure...",
    )

    if st.button("🔍 Analyse Note", use_container_width=True, type="primary"):
        if note.strip() == "":
            st.warning("Please enter a clinical note first.")
        else:
            cleaned  = preprocess(note)
            features = tfidf.transform([cleaned])
            preds    = model.predict(features)[0]
            proba_d  = model.estimators_[0].predict_proba(features)[0]
            proba_t  = model.estimators_[1].predict_proba(features)[0]

            disease   = le_disease.inverse_transform([preds[0]])[0]
            treatment = le_treatment.inverse_transform([preds[1]])[0]
            conf_d    = max(proba_d) * 100
            conf_t    = max(proba_t) * 100
            color     = DISEASE_COLORS.get(disease, "#888")
            icon      = DISEASE_ICONS.get(disease, "🏥")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="background:{color}18; border-left:4px solid {color};
                                border-radius:8px; padding:20px 24px;">
                        <p style="margin:0;font-size:13px;color:#666;">Detected Disease</p>
                        <p style="margin:4px 0 0;font-size:28px;font-weight:600;color:{color};">
                            {icon} {disease}
                        </p>
                        <p style="margin:6px 0 0;font-size:13px;color:#888;">
                            Confidence: <b>{conf_d:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="background:#1D9E7518; border-left:4px solid #1D9E75;
                                border-radius:8px; padding:20px 24px;">
                        <p style="margin:0;font-size:13px;color:#666;">Recommended Treatment</p>
                        <p style="margin:4px 0 0;font-size:28px;font-weight:600;color:#0F6E56;">
                            💊 {treatment}
                        </p>
                        <p style="margin:6px 0 0;font-size:13px;color:#888;">
                            Confidence: <b>{conf_t:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probability bars for all diseases
            st.subheader("Disease Probability Breakdown")
            prob_df = pd.DataFrame({
                "Disease":     le_disease.classes_,
                "Probability": proba_d * 100,
            }).sort_values("Probability", ascending=False)

            fig, ax = plt.subplots(figsize=(7, 2.8))
            colors  = [DISEASE_COLORS.get(d, "#888") for d in prob_df["Disease"]]
            bars    = ax.barh(prob_df["Disease"], prob_df["Probability"], color=colors, height=0.5)
            for bar, val in zip(bars, prob_df["Probability"]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=11)
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            ax.spines[["top","right"]].set_visible(False)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Preprocessed tokens
            with st.expander("🔤 See NLTK preprocessed tokens"):
                tokens = cleaned.split()
                st.write(" · ".join(f"`{t}`" for t in tokens))

# =============================================================
#  PAGE 2 — MODEL PERFORMANCE
# =============================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    Y_pred = model.predict(X_test)

    task_names = ["Disease Detection", "Treatment Recommendation"]
    encoders   = [le_disease, le_treatment]

    acc_d = accuracy_score(Y_test[:, 0], Y_pred[:, 0]) * 100
    acc_t = accuracy_score(Y_test[:, 1], Y_pred[:, 1]) * 100

    # Top metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Disease Accuracy",   f"{acc_d:.1f}%")
    m2.metric("Treatment Accuracy", f"{acc_t:.1f}%")
    m3.metric("Test Samples",       len(Y_test))

    st.divider()

    for i, (task, le) in enumerate(zip(task_names, encoders)):
        st.subheader(f"Task {i+1}: {task}")
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]

        col1, col2 = st.columns([1, 1])

        # Classification report as table
        with col1:
            report = classification_report(
                y_true, y_pred,
                target_names=le.classes_,
                output_dict=True,
                zero_division=0,
            )
            rows = []
            for cls in le.classes_:
                r = report[cls]
                rows.append({
                    "Class":     cls,
                    "Precision": f"{r['precision']:.2f}",
                    "Recall":    f"{r['recall']:.2f}",
                    "F1-Score":  f"{r['f1-score']:.2f}",
                    "Support":   int(r['support']),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrix
        with col2:
            cm  = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=ax, linewidths=0.5,
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual",    fontsize=10)
            ax.set_title("Confusion Matrix", fontsize=11)
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.yticks(rotation=0,  fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

    # Top keywords chart
    st.subheader("🔑 Top Keywords per Disease")
    feature_names = tfidf.get_feature_names_out()
    disease_clf   = model.estimators_[0]
    diseases      = le_disease.classes_

    fig, axes = plt.subplots(1, len(diseases), figsize=(14, 3.5))
    for ax, (idx, dname) in zip(axes, enumerate(diseases)):
        coef     = disease_clf.coef_[idx]
        top_idx  = np.argsort(coef)[-8:]
        words    = [feature_names[j] for j in top_idx]
        vals     = [coef[j]          for j in top_idx]
        color    = DISEASE_COLORS.get(dname, "#888")
        ax.barh(words, vals, color=color, height=0.6)
        ax.set_title(dname, fontsize=10, fontweight="bold", color=color)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("Log-odds coefficients — higher = stronger signal", fontsize=10, color="#666")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# =============================================================
#  PAGE 3 — DATASET EXPLORER
# =============================================================
elif page == "📋 Dataset Explorer":
    st.title("📋 Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Diseases",      df["disease"].nunique())
    col3.metric("Treatments",    df["treatment"].nunique())

    st.divider()

    # Filter
    disease_filter = st.multiselect(
        "Filter by disease",
        options=df["disease"].unique().tolist(),
        default=df["disease"].unique().tolist(),
    )
    filtered = df[df["disease"].isin(disease_filter)]

    st.dataframe(
        filtered[["clinical_notes", "disease", "treatment"]].rename(columns={
            "clinical_notes": "Clinical Note",
            "disease":        "Disease",
            "treatment":      "Treatment",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Disease Distribution")
    dist = df["disease"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = [DISEASE_COLORS.get(d, "#888") for d in dist.index]
    ax.bar(dist.index, dist.values, color=colors, width=0.5)
    ax.set_ylabel("Number of records")
    ax.spines[["top","right"]].set_visible(False)
    for i, v in enumerate(dist.values):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# =============================================================
#  PAGE 4 — HOW IT WORKS
# =============================================================
elif page == "📚 How It Works":
    st.title("📚 How It Works")

    st.markdown("""
    This project uses **Multi-Task Logistic Regression** to simultaneously predict:
    - **Task 1** → Disease (Diabetes, Heart Disease, Flu, Hypertension)
    - **Task 2** → Treatment Recommendation

    ---
    ### Pipeline Overview
    """)

    steps = [
        ("1️⃣", "Raw Clinical Note",   "Free-text patient symptom description"),
        ("2️⃣", "NLTK Preprocessing",  "Lowercase → Remove punctuation → Tokenise → Remove stopwords → Stem"),
        ("3️⃣", "TF-IDF Vectoriser",   "Converts cleaned text into 500 numeric features"),
        ("4️⃣", "Multi-Output Model",  "Two Logistic Regression classifiers trained simultaneously"),
        ("5️⃣", "Dual Predictions",    "Disease label + Treatment label output together"),
    ]

    for icon, title, desc in steps:
        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-start;gap:16px;
                        padding:14px 18px;margin-bottom:10px;
                        background:var(--background-color);
                        border:1px solid #e0e0e0;border-radius:10px;">
                <span style="font-size:26px;">{icon}</span>
                <div>
                    <p style="margin:0;font-weight:600;font-size:15px;">{title}</p>
                    <p style="margin:2px 0 0;color:#666;font-size:13px;">{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("📦 Recommended Real Datasets")
    datasets = [
        ("UCI Heart Disease",        "https://archive.ics.uci.edu/ml/datasets/heart+Disease"),
        ("MTSamples Clinical Notes", "https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions"),
        ("Symptom → Disease",        "https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset"),
        ("MIMIC-III (free reg.)",    "https://physionet.org/content/mimiciii/"),
        ("MedNLI",                   "https://jgc128.github.io/mednli/"),
    ]
    for name, url in datasets:
        st.markdown(f"- [{name}]({url})")

    st.divider()
    st.subheader("⚙️ Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("🐍 Python 3.x")
    c2.info("📝 NLTK")
    c3.info("🤖 Scikit-learn")
    c4.info("📊 Streamlit")