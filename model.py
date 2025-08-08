import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation (1).csv")

# Check for nulls (optional for debugging)
print(df.isnull().sum())

# Split features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print("✅ Accuracy:", accuracy)

# Predict on a sample
new_features = [[36, 58, 25, 28.66024, 59, 36.549, 8.984]]
predicted_crop_encoded = model.predict(new_features)
predicted_crop = le.inverse_transform(predicted_crop_encoded)
print("🌱 Predicted Crop:", predicted_crop[0])

# Save model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to crop_model.pkl")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Label encoder saved to label_encoder.pkl")
