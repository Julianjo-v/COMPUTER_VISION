import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load CSV generated from HSV shape pipeline

df = pd.read_csv("shape_descriptors_project_640x480.csv")


# REQUIRED DESCRIPTORS (MUST MATCH CLASSIFICATION)

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "Circularity",
    "Compactness",
    "Convexity"
]

TARGET_COLUMN = "Label"


# Sanity check (prevents silent bugs)

missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")


# Select features and labels explicitly

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# Encode string labels → integers

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Train decision tree

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42
)
clf.fit(X, y_encoded)


# Save model bundle (TRAINING ↔ INFERENCE MATCH)

joblib.dump(
    {
        "decision_tree": clf,
        "label_encoder": label_encoder,
        "feature_names": FEATURE_COLUMNS
    },
    "decision_tree_model_final.joblib"
)

print(" Model trained and saved successfully")
print("Feature order:", FEATURE_COLUMNS)
print("Classes:", label_encoder.classes_)
