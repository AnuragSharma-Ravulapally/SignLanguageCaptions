import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# 2. Separate features (X) and labels (y)
# 'label' is the first column; the rest are the 63 XYZ coordinates
X = df.drop('label', axis=1)
y = df['label']

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Classifier
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the model's accuracy on the unseen 20% of data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model saved successfully as 'gesture_model.pkl'")