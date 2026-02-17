import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('data/output', exist_ok=True)

# Space for custom lost, eval etc.
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0) :
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)

def focal_loss(y_true, y_pred, alpha=0.15, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses on hard-to-classify examples.

    Parameters:
    -----------
    alpha : float
        Weighting factor (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_mean(focal_loss_value) 


model = keras.models.load_model('examples/assignment.h5',compile=False)
print("Model loaded successfully!")

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 5e-4),
    loss = weighted_binary_crossentropy,
    metrics = ['accuracy', keras.metrics.AUC(name='auc')]
)

model.summary()

df = pd.read_csv('data/tourism.csv', sep=',') #If the data seperate column by comma change here

X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Predict value for report
y_pred_proba = model.predict(X_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Report Section
print("\n" + "="*60)
print("INFERENCE RESULTS")

accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

try:
    auc_score = roc_auc_score(y, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
except:
    print("AUC Score: Could not calculate")

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Bank Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add additional statistics as text
tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()

# Save the figure
output_path = 'data/output/assignment_result_cf.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: {output_path}")

# Also create a normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Bank Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path_normalized = 'data/output/assignment_result_cfn.jpg'
plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")