import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 1: Simulate medical image data (100 images with 256 pixel features)
X = np.random.rand(100, 256)  # Feature matrix: 100 images, each with 256 features (simulated pixels)
y = np.random.randint(0, 2, 100)  # Binary labels: 0 (healthy), 1 (tumor)

# Step 2: Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 decision trees for better accuracy
model.fit(X, y)

# Step 3: Predict on new simulated medical image
new_image = np.random.rand(1, 256)  # Generate a random new image
prediction = model.predict(new_image)  # Make a prediction

# Step 4: Display the result
print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")
