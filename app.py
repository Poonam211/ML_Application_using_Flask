from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)
MODEL_PATH = "model_store/alphabet_model.h5"
os.makedirs("model_store", exist_ok=True)


# ------------------ TRAIN ENDPOINT ------------------
@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file missing"}), 400
    
    f = request.files['file']
    df = pd.read_csv(f)
    
    if 'label' not in df.columns:
        return jsonify({"error": "CSV must have a 'label' column"}), 400

    # Separate features and labels
    y = df['label']
    X = df.drop('label', axis=1).values

    # Normalize and reshape for CNN (28x28 images, 1 channel)
    X = X.reshape(-1, 28, 28, 1) / 255.0

    # Convert labels to numeric (A=0, B=1, ... Z=25)
    if y.dtype == object:
        y_num = np.array([ord(str(c).upper()) - 65 for c in y])
    else:
        y_num = y.values.astype(int)

    # One-hot encode labels
    ya = to_categorical(y_num, num_classes=26)

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X, ya, batch_size=64, epochs=15, validation_split=0.2, verbose=1, shuffle=True)

    # Save model
    model.save(MODEL_PATH)

    return jsonify({"message": "Model trained and saved!"})


# ------------------ TEST ENDPOINT ------------------
@app.route('/test', methods=['POST'])
def test():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 400

    model = load_model(MODEL_PATH)

    f = request.files['file']
    df = pd.read_csv(f)

    if 'label' not in df.columns:
        return jsonify({"error": "CSV must have a 'label' column to compute accuracy"}), 400

    # Separate features and labels
    X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_true = df['label'].values

    # Convert labels
    y_true_num = np.array([ord(str(c).upper()) - 65 for c in y_true])
    y_true_cat = to_categorical(y_true_num, num_classes=26)

    # Evaluate model
    loss, accuracy = model.evaluate(X, y_true_cat, verbose=0)

    # Predictions
    preds = model.predict(X).argmax(axis=1)
    alpha = np.array([chr(i) for i in range(65, 91)])
    pred_letters = [alpha[p] for p in preds]

    return jsonify({
        "loss": float(loss),
        "accuracy": float(accuracy),
        "predictions": pred_letters
    })


# ------------------ PREDICT ENDPOINT ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 400
    
    model = load_model(MODEL_PATH)
    alpha = np.array([chr(i) for i in range(65, 91)])  # A–Z
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
    
    # JSON must contain 784 pixel values (28x28 image flattened)
    try:
        X = np.array(data["pixels"]).reshape(1, 28, 28, 1) / 255.0
    except Exception as e:
        return jsonify({"error": f"Invalid pixel data: {str(e)}"}), 400
    
    probs = model.predict(X)[0]
    prediction = alpha[np.argmax(probs)]

    return jsonify({
        "prediction": prediction,
        "probabilities": {alpha[i]: float(probs[i]) for i in range(26)}
    })


if __name__ == "__main__":
    app.run(debug=True)
