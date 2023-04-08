import pandas as pd
import numpy as np
import fasttext
import pickle
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import time
import os
import re
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from joblib import dump, load

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




#La idea de esto era usar el FastText de facebook pero me peta el ordena con la de 300 y con el .py para recortarla a 100 tamnien asi que...Hugging Face instead of FastText 

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
"""
Necesito una barra de progreso porque el pc que tengo tarda mucho en entrenar el modelo, pero usando sklearn no h encotrado manera alguna de hacerlo sencillo
Asi que voy a contar el progreso usando una custom callback function en cada vuelta de la validacion cruzada. Luego otra implemetacion de una clase de sklearn(Ideas de ChatGPT) 
"""

class TqdmCustomCallback:
    def __init__(self, total_folds):
        self.total_folds = total_folds
        self.progress_bar = tqdm(total=self.total_folds, desc="Progreso: ", unit="Vuelta")

    def __call__(self, index, _):
        self.progress_bar.update(1)

    def close(self):
        self.progress_bar.close()

def progress_scorer(y_true, y_pred):
    global fit_count
    fit_count += 1
    print(f"Progress: {fit_count}/{total_fits} completed")
    return accuracy_score(y_true, y_pred)





nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text




def tokenize(url):
    tokens = re.split(r'\W+', url)
    return tokens

def domain_to_vector(domain, model):
    embeddings = model.encode([domain])
    return embeddings[0]

def preprocess_data(data):
    """
    Preprocess the data by splitting it into training and testing sets, and vectorizing the URLs using the TF-IDF method.
    """
    # Preprocess the URLs
    data['url'] = data['url'].apply(preprocess_text)

    # Split the data into training and testing sets
    X = data['url']
    y = data['phishing']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test, vectorizer  # Include vectorizer que se te olvida fiera

def train_model(X_train, y_train):
    """
    Train the model using logistic regression, gradient boosting, and random forest classifiers, and perform grid search to find the best hyperparameters.
    """

    #Soy consciente (en parte) de la burrada que es esto, pero es que si no le añado estas mierdas no hay forma de saber
    #si está evolucionando o ha fallado o qué
    global fit_count,total_fits 
    fit_count = 0
    total_fits = 901 
    total_folds = 5
    callback = TqdmCustomCallback(total_folds=total_folds)






    # Define the pipeline
    pipe = Pipeline([
        ('clf', None)
    ])

    # Set the hyperparameters to search
    param_grid = [

          {
            'clf': [LogisticRegression(solver='liblinear', max_iter=10000)],  # Increased the number of iterations
            'clf__C': np.logspace(-6, 1, 30),  # Extended the range of C values
            'clf__penalty': ['l1', 'l2']
        },
        
        {
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [5, 10, 50, 100],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        },
        {
            'clf': [GradientBoostingClassifier()],
            'clf__n_estimators': [10, 30,50],
            'clf__learning_rate': [0.001, 0.01, 0.1, 1],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        }
    ]
    
     # Check if the best parameters file exists
    best_params_file = "best_params.pkl"
    if os.path.exists(best_params_file):
        # Load the best parameters from the file
        with open(best_params_file, "rb") as f:
            best_params = pickle.load(f)
        # Set the best parameters to the pipeline
        pipe.set_params(**best_params)
        pipe.fit(X_train, y_train)
    else:
        # Perform the grid search
        grid_search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5), scoring=make_scorer(progress_scorer), verbose=3)
        grid_search.fit(X_train, y_train)
        callback.close()

        # Save the best parameters to a file
        best_params = grid_search.best_params_
        with open(best_params_file, "wb") as f:
            pickle.dump(best_params, f)

        # Set the best parameters to the pipeline
        pipe.set_params(**best_params)
        pipe.fit(X_train, y_train)

    return pipe

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int).flatten()
    print(classification_report(y_test, y_pred))

def save_model(model, file_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{file_name}_{timestamp}.h5"
    
    model.save(file_name)
    print(f"Model saved to {file_name}")

def load_saved_model(file_name):
    return tf.keras.models.load_model(file_name)

def get_existing_models(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5') and f.startswith('trained_model')]
    return model_files

def compare_models(new_model, existing_models, X_test, y_test):
    best_model = new_model
    best_score = new_model.evaluate(X_test, y_test, verbose=0)[1]

    for model_file in existing_models:
        loaded_model = load_saved_model(model_file)
        loaded_score = loaded_model.evaluate(X_test, y_test, verbose=0)[1]

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
        save_model(model, 'trained_model')
    else:
        print("The new model is not better than the existing models. No new model will be saved.")