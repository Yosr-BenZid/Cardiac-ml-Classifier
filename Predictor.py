import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

print("--- GENERATING LOCAL CLINICAL MATRIX ENGINE ---")

# 1. Automatically generate 500 rows of mock patient heart data mathematically
X_raw, y_raw = make_classification(
    n_samples=500, 
    n_features=12, 
    n_informative=10, 
    random_state=42
)

# 2. Map the data into a structured clinical data table
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
df = pd.DataFrame(X_raw, columns=column_names)
df['target'] = y_raw

# 3. Separate input features from the target output prediction
X = df.drop(columns=['target'])
y = df['target']

# Split data into 80% teaching data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the standalone machine learning engine
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"--- CLINICAL ENGINE COMPILED SUCCESSFULLY ---")
print(f"Automated Model Training Accuracy: {accuracy * 100:.2f}%\n")

# 5. Start interactive questionnaire
def evaluate_patient():
    print("=== PULSESCALE ANALYTICS AUTOMATED PREDICTOR ===")
    age = float(input("Enter Patient Age (Standardized value, e.g. 1.0): "))
    sex = float(input("Enter Sex Metrics (e.g. 0.0 or 1.0): "))
    cp = float(input("Chest Pain Profile value: "))
    trestbps = float(input("Resting Blood Pressure coefficient: "))
    chol = float(input("Serum Cholesterol index: "))
    fbs = float(input("Fasting Blood Sugar index: "))
    restecg = float(input("Resting ECG variant: "))
    thalach = float(input("Maximum Heart Rate metric: "))
    exang = float(input("Exercise Induced Angina metric: "))
    oldpeak = float(input("ST Depression dynamic: "))
    slope = float(input("Slope of Peak Exercise ST segment: "))
    ca = float(input("Number of Major Vessels metric: "))
    
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]
    prediction = model.predict([features])
    
    print("\n" + "="*45)
    if prediction == 1:
        print("ALERT: HIGH CARDIOVASCULAR METRIC PATTERNS DETECTED.")
    else:
        print("NORMAL: BASELINE PARAMETERS CONFIRMED STABLE.")
    print("="*45)

evaluate_patient()
