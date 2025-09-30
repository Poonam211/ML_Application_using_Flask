from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = "model_store/alphabet_model.h5"
os.makedirs("model_store", exist_ok=True)

# Train endpoint
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
    X = df.drop('label', axis=1) / 255  # normalize pixel values

    # Convert labels to numeric 0-25
    if y.dtype == object:  # letters
        try:
            y_num = np.array([ord(str(c).upper()) - 65 for c in y])
        except Exception as e:
            return jsonify({"error": f"Failed to convert labels: {str(e)}"}), 400
    else:  # already numeric
        y_num = y.values.astype(int)

    # One-hot encode labels
    ya = to_categorical(y_num, num_classes=26)

    # Build a simple neural network
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(26, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model (you can increase epochs later)
    model.fit(X, ya, batch_size=64, epochs=5, verbose=1)

    # Save model
    model.save(MODEL_PATH)

    return jsonify({"message": "Model trained and saved!"})



# Test endpoint
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
    X = df.drop('label', axis=1).values / 255
    y_true = df['label'].values

    # Convert letters to numbers 0-25
    y_true_num = np.array([ord(c.upper()) - 65 for c in y_true])
    y_true_cat = to_categorical(y_true_num, num_classes=26)

    # Evaluate loss and accuracy
    loss, accuracy = model.evaluate(X, y_true_cat, verbose=0)

    # Get predictions
    preds = model.predict(X).argmax(axis=1)
    alpha = np.array([chr(i) for i in range(65, 91)])
    pred_letters = [alpha[p] for p in preds]

    return jsonify({
        "loss": float(loss),
        "accuracy": float(accuracy),
        "predictions": pred_letters
    })

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 400
    
    model = load_model(MODEL_PATH)
    alpha = np.array([chr(i) for i in range(65, 91)])
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
    
    # JSON can contain flattened pixel values
    X = np.array(data["pixels"]).reshape(1, 784) / 255
    yp = model.predict(X).argmax()
    return jsonify({"prediction": alpha[yp]})

if __name__ == "__main__":
    app.run(debug=True)
