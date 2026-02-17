import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# [Data Preparation]+[Feature Extraction]
# Load the data 
print("Loading data...")
df = pd.read_csv('data/tourism.csv', sep=',') #If the data seperate column by comma change here

print(f"Dataset shape: {df.shape}") 
#ดูข้อมูลว่ามีกี่เเถวกี่Column 
print(f"\nColumn names: {df.columns.tolist()}") 
#Print Column Heading มาดู
print(f"\nTarget distribution:\n{df['ProdTaken'].value_counts()}")
#ดูว่าตอนนี้ y มีอย่างละเท่าไหร่ #เปลี่ยนชื่อ y ให้เป็นค่าที่อยากรู้ in this case is 'ProdTaken'

# Separate features and target เเยก input กับ Output จะได้เอาไป Train ได้ 
X = df.drop('ProdTaken', axis=1)
#drop คือไม่เอา y #เปลี่ยนชื่อ y ด้วยนะ 
y = df['ProdTaken']

# Encode target variable (yes -> 1, no -> 0) อันนี้ค่อยใช้ในกรณัืที่ output ของข้อมูลี่ได้มา ไม่ใช่ตัวเลข เราเบลยต้องเปลี่ยนให้เป็น 0,1 ก่แน 
#label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y)

#เปลี่ยนข้อมูลที่เป็น category เป็นตัวเลข 0,1 
# Handle categorical variables using one-hot encoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")
# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print(f"Features after encoding: {X_encoded.shape[1]}")
print(X_encoded.head())

# ใช้ เเยกข้อมูลไป test
#เปลี่ยนชื่อ Variable ตามตัวเเปรที่มี เพื่อ
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
#stratify คือทำให้ มีจำนวน y อยุ่ใน train test เท่ากัน ไม่ bias 

#scale values ของข้อมูลให้อยู๋ในระดับเดียวกันไม่มีอันไหนเว่อไป เพราะพวกเงินเดือนหรือที่เลขสูงๆ ถ้าไม่ปรับจะมีผลมากเกิน 
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

# เริ่มสร้าง Model #Custom this
model = keras.Sequential([
    # Input layer + First hidden layer
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    

    # Second hidden layer
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    # Third hidden layer
    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(16, activation='relu'),

    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

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

metrics = ['accuracy', 
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name = 'precision'),
            keras.metrics.Recall(name = 'recall')]

# Compile the model 
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 5e-4),
    loss = weighted_binary_crossentropy,
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
    # metrics = metrics
)

# Compile the model 
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy', keras.metrics.AUC(name='auc')]
# )
# Display model summary

model.summary()

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # callbacks.ModelCheckpoint('examples/best_cnn_model.h5', save_best_only=True),
    # callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]


# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=callbacks_list
)

#Train เเล้วมา Test # Evaluate the model on test set

print("\nEvaluating the model...")
result = model.evaluate(X_test_scaled, y_test, verbose=0)
print(result)

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
model.save('examples/assignment.h5')
print("\nModel saved to test")