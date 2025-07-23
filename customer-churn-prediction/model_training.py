import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("customerChurn.csv")

# Clean data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Select features and target
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
