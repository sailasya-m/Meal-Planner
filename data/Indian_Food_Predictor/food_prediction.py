# food_prediction.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Load dataset ---
file_path = "D:\\Meal Planner\\data\\Indian_Food_Predictor\\IndianFoodDatasetCSV.csv"
df = pd.read_csv(file_path)

# --- Inspect dataset ---
print("Columns in dataset:", df.columns)
print(df.head())

# --- Identify categorical columns automatically ---
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

# --- Encode categorical columns ---
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # Convert all to string before encoding

# --- Define features and target ---
target_column = 'Cuisine'  # Predict cuisine type

X = df.drop(columns=[target_column])
y = df[target_column]

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
