import pandas as pd
import numpy as np
import fasttext
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump, load
import datetime

# Load the dataset from a CSV file
def load_data(file_path):
    """
    Load the data from a CSV file, with columns 'url' and 'phishing'.
    """
    data = pd.read_csv(file_path)
    return data

# Preprocess the data (vectorize the URLs using TF-IDF)
def preprocess_data(data):
    """
    Preprocess the data by splitting it into training and testing sets, and vectorizing the URLs using the TF-IDF method.
    """

     # Load the FastText pre-trained models
    ft_eng = fasttext.load_model('cc.en.300.bin')
    ft_esp = fasttext.load_model('cc.es.300.bin')
    def domain_to_vector(domain):
        tokens = tokenize(domain)
        vectors = []
        for token in tokens:
            if token in ft_eng.words:
                vectors.append(ft_eng[token])
            elif token in ft_esp.words:
                vectors.append(ft_esp[token])
        if not vectors:
            return np.zeros(300)
        return np.mean(vectors, axis=0)

    X = np.array([domain_to_vector(domain) for domain in domains])
    X = data['url']
    y = data['phishing']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test, vectorizer

# Train the model using logistic regression and grid search for hyperparameter tuning
def train_model(X_train, y_train):
    """
    Train the model using logistic regression and random forest classifiers, and perform grid search to find the best hyperparameters.
    """
    # Define the pipeline
    pipe = Pipeline([
        ('clf', None)
    ])

    # Set the hyperparameters to search
    param_grid = [
        {
            'clf': [LogisticRegression(solver='liblinear', max_iter=1000)],  # Increased the number of iterations
            'clf__C': np.logspace(-4, 4, 20),
            'clf__penalty': ['l1', 'l2']
        },
        {
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [10, 50, 100, 200],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        }
    ]

    # Perform grid search
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set and print the classification report.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Save the model
def save_model(model, vectorizer, file_name):
    """
    Save the trained model and vectorizer to a file, including the current timestamp.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{file_name}_{timestamp}.joblib"
    
    dump({'model': model, 'vectorizer': vectorizer}, file_name)
    print(f"Model saved to {file_name}")

# Load the saved model
def load_saved_model(file_name):
    """
    Load the saved model and vectorizer from a file.
    """
    loaded = load(file_name)
    return loaded['model'], loaded['vectorizer']

def get_existing_models(directory):
    """
    Get a list of existing model files in the specified directory.
    """
    model_files = [f for f in os.listdir(directory) if f.endswith('.joblib') and f.startswith('trained_model')]
    return model_files

def compare_models(new_model, existing_models, X_test, y_test):
    """
    Compare the new model to existing models and return the best model.
    """
    best_model = new_model
    best_score = new_model.score(X_test, y_test)

    for model_file in existing_models:
        loaded_model, _ = load_saved_model(model_file)
        loaded_score = loaded_model.score(X_test, y_test)

        if loaded_score > best_score:
            best_model = loaded_model
            best_score = loaded_score

    return best_model

if __name__ == '__main__':
    print("Loading data...")
    data = load_data('c:/Users/Nano/Desktop/APruebavirtualenv/Ahorasiquesi/.venv/datos_entrenamiento.csv')
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    
    print("Training the model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    print("Checking for existing models...")
    existing_models = get_existing_models('.')
    
    print("Comparing models...")
    best_model = compare_models(model, existing_models, X_test, y_test)
    
    if best_model == model:
        print("Saving the new model...")
        save_model(model, vectorizer, 'trained_model')
    else:
        print("The new model is not better than the existing models. No new model will be saved.")