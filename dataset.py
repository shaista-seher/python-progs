# MINIMAL ML CODE FOR BEGINNERS
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("="*40)
print("SIMPLE MACHINE LEARNING PROJECT")
print("="*40)
print(f"\nDataset: Breast Cancer Wisconsin")
print(f"Model: Random Forest Classifier")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nPrediction for first test sample:")
print(f"  Predicted: {'Benign' if y_pred[0] == 1 else 'Malignant'}")
print(f"  Actual: {'Benign' if y_test[0] == 1 else 'Malignant'}")
print("="*40)