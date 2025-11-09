from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os
from sklearn.model_selection import train_test_split
from train_model import gather_embeddings  # reuse the gather_embeddings function
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "model/bird_yamnet_classifier.h5"
LABELS_PATH = "model/labels.npy"

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Train it first.")

X, y, class_names = gather_embeddings()
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = tf.keras.models.load_model(MODEL_PATH)
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# ✅ Add this block here
test_acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"\nTest Accuracy: {test_acc:.4f}\n")
print(report)

# Save the metrics report to a text file
os.makedirs("model", exist_ok=True)
with open("model/classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# Generate and save the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
print("Confusion matrix saved to model/confusion_matrix.png")
