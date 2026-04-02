# ============================================
# Clinical Outcome Prediction using NLTK + Logistic Regression
# ============================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import nltk
import string
import warnings

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Suppress warnings (important for clean output)
warnings.filterwarnings("ignore")

# Download NLTK resources (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

# ============================================
# 2. Load Dataset
# ============================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

# ============================================
# 3. Generate Symptom Text
# ============================================

def generate_symptoms(row):
    symptoms = []
    
    if row["Glucose"] > 140:
        symptoms.append("high sugar")
    if row["BMI"] > 30:
        symptoms.append("obesity")
    if row["BloodPressure"] > 90:
        symptoms.append("high blood pressure")
    if row["Age"] > 50:
        symptoms.append("fatigue")
    
    return " ".join(symptoms)

df["Symptoms"] = df.apply(generate_symptoms, axis=1)

# ============================================
# 4. NLTK Preprocessing
# ============================================

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(tokens)

df["Symptoms"] = df["Symptoms"].apply(preprocess_text)

# ============================================
# 5. Feature Engineering
# ============================================

df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
df["Age_Glucose"] = df["Age"] * df["Glucose"]

# ============================================
# 6. Split Data
# ============================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 7. Pipeline (BEST PRACTICE 🔥)
# ============================================

numeric_features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "Glucose_BMI", "Age_Glucose"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("text", TfidfVectorizer(), "Symptoms")
    ]
)

model = LogisticRegression(
    max_iter=3000,
    solver="liblinear",
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# ============================================
# 8. Train Model
# ============================================

pipeline.fit(X_train, y_train)

# ============================================
# 9. Evaluation
# ============================================

y_pred = pipeline.predict(X_test)

print("===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# 10. Cross Validation (CORRECT WAY ✅)
# ============================================

cv_scores = cross_val_score(pipeline, X, y, cv=5)

print(f"\nCross Validation Accuracy: {cv_scores.mean():.4f}")

# ============================================
# 11. Final Output
# ============================================

print("\nModel uses NLTK + TF-IDF + Logistic Regression with feature interaction.")