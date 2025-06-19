# 🧠 Predictive Maintenance using CNN with SHAP and LIME

# 📦 Step 1: Import Required Libraries
print("Step 1: Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# 📂 Step 2: Upload and Load Dataset
print("Step 2: Loading dataset...")
df = pd.read_csv("predictive_maintenance.csv")
print("Dataset loaded successfully. Sample:")
print(df.head())

# ⚙️ Step 3: Preprocess the Data
print("Step 3: Preprocessing data...")
selected_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

features = df[selected_columns]
labels = df["Target"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = features_scaled.reshape(-1, features.shape[1], 1, 1)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data preprocessing completed.")

# 🏗️ Step 4: Build the CNN Model
print("Step 4: Building CNN model...")
model = Sequential()
model.add(Conv2D(16, (3, 1), activation="relu", input_shape=(X.shape[1], 1, 1)))
model.add(MaxPooling2D((2, 1)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 📈 Step 5: Train the CNN
print("Step 5: Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
print("Model training completed.")

# ✅ Step 6: Evaluate the Model
print("Step 6: Evaluating model...")
preds = model.predict(X_test)

y_pred_class = np.argmax(preds, axis=1)
y_pred_rate = preds[:, 1]

print("\nClassification Report (Defect Status):")
print(classification_report(np.argmax(y_test, axis=1), y_pred_class))

print("\nSample predictions:")
for i in range(5):
    print(
        f"Sample {i+1}: Defect Status = {y_pred_class[i]}, Defect Rate = {y_pred_rate[i]:.4f}"
    )

# 🔍 Step 7: SHAP Explanation for Model Transparency
# print("\nStep 7: Generating SHAP explanation...")
# X_shap = X_test.reshape(X_test.shape[0], X_test.shape[1])
# explainer = shap.KernelExplainer(
#     lambda x: model.predict(x.reshape(-1, X.shape[1], 1, 1))[:, 1], X_shap[:100]
# )
# shap_values = explainer.shap_values(X_shap[:1])
# # shap.initjs()

# sample_features = X_shap[0]
# shap_vals = shap_values[0]

# print("SHAP values shape:", shap_vals.shape)
# print("Feature values shape:", sample_features.shape)

# # shap.force_plot(
# #     explainer.expected_value, shap_vals, sample_features, feature_names=selected_columns
# # )
# shap.plots._bar.bar(shap_vals, feature_names=selected_columns, max_display=5)

# SHAP explanation using KernelExplainer
X_shap = X_test.reshape(X_test.shape[0], X_test.shape[1])
explainer = shap.KernelExplainer(
    lambda x: model.predict(x.reshape(-1, X.shape[1], 1, 1))[:, 1], X_shap[:100]
)

shap_values = explainer.shap_values(X_shap[:10])  # Get for 10 samples

# Convert to Explanation object manually
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.array([explainer.expected_value] * len(shap_values)),
    data=X_shap[:10],
    feature_names=selected_columns,
)

# Plot
# shap.plots.bar(explanation, max_display=5)
plt.figure()
shap.plots.bar(explanation, max_display=5, show=False)
plt.tight_layout()
plt.savefig("shap_bar_plot.png")
print("SHAP bar plot saved as 'shap_bar_plot.png'.")


# 🔬 Step 8: LIME Explanation for Local Interpretability
print("\nStep 8: Generating LIME explanation...")
X_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])
explainer = LimeTabularExplainer(
    X_flat,
    feature_names=selected_columns,
    class_names=["No Failure", "Failure"],
    discretize_continuous=True,
)

exp = explainer.explain_instance(
    X_flat[0], lambda x: model.predict(x.reshape(-1, X.shape[1], 1, 1)), num_features=5
)
print("\n📝 LIME Human-Friendly Explanation:")
for feature, weight in exp.as_list():
    status = "increased" if weight > 0 else "reduced"
    print(
        f"- {feature} → This {status} the predicted failure risk by approx {abs(weight):.2%}"
    )

print("LIME explanation generated.")
