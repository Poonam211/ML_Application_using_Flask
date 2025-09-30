# ML Application using Flask (API Development)

## Project Description
This project is a Machine Learning API developed using Flask. It allows users to train, test, and predict handwritten alphabet characters using a neural network model. The API accepts CSV datasets for training/testing and JSON data for predictions.  

## Features
- **/train** : Train a neural network model on a provided dataset (CSV).  
- **/test** : Evaluate the trained model on a test dataset (CSV) and return accuracy and predictions.  
- **/predict** : Make predictions on individual data points provided as JSON.

## Folder Structure
ML_Application_using_Flask/
│
├── app.py # Flask application with API endpoints
├── requirements.txt # Python dependencies
├── A_Z_small.csv # Original dataset (optional)
├── train_small.csv # Training dataset
├── test_small_labeled.csv # Test dataset
├── model_store/ # Folder to store trained models
└── README.md # Project documentation
