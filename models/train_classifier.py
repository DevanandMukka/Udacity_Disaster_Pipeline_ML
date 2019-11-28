import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

import re

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        categories -> used for app building in the later stages
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('msgs_with_cat',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X,Y


def tokenize(text):
    """
    Function to normalize & token the passed text
    Input : Text as string
    Output : Cleaned tokens from the given string
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    pipeline_1 = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                    ])

    
    #Parameter set to tune the performance of model
    parameters_1 = {#     'tfidf__use_idf': (True, False)
        'tfidf__max_df': (0.75, 1.0)
#         'clf__estimator__n_estimators': [10, 20],
#         'clf__estimator__learning_rate': [1,2]
    }
    

#     Considered only the Max df parameter as if we include others grid search is taking much more time
    cv_1 = GridSearchCV(pipeline_1, param_grid=parameters_1)
    return cv_1


def evaluate_model(model, X_test, Y_test):
    print("\nBest Parameters:", model.best_params_)
    
    Y_pred = model.predict(X_test)
    # Using the classification report to get the accuracy related metrics
    for i, col in enumerate(Y_test.columns.values):
            print(col)
            print(classification_report(Y_test.loc[:,col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        Y_pred=model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model,model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()