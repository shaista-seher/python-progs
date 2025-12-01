# Simple MNIST classification with Random Forest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

print("Loading MNIST dataset...")
# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data[:5000], mnist.target[:5000].astype(int)  # Smaller subset

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize predictions
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
axes = axes.ravel()

for i in range(24):
    axes[i].imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"True: {y_test.iloc[i]}\nPred: {y_pred[i]}", 
                      color='green' if y_test.iloc[i] == y_pred[i] else 'red',
                      fontsize=9)

plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()