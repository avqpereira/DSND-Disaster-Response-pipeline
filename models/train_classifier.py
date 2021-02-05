import sys
import pandas as pd
import sqlite3
import re
import pickle
import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    A function for loading a specific table from a given database. It is then properly sliced, 
    in order to be used on Machine Learning applications.
    Inputs:
        database_filepath: String, filepath for the database
    Output:
        X: Pandas Series containing 'message' column from table queried
        y: Pandas Dataframe containing the possile categories for each message.
        category_names: List of categories column names
  """
    conn = sqlite3.connect(database_filepath)
    
    # load data from database
    df = pd.read_sql("SELECT * FROM MessagesCategories", conn)
    
    conn.close()
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    """
    A function used to normalize, tokenize and lemmatize a given text.
    Input:
        text: String containing a message to be handled.
    Output:
        clean_tokens: List of clean tokens created from the given text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalize case
    text = text.lower()
    
    # tokenize
    tokens = word_tokenize(text)
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    A function used to build a model for NLP projects, using a sklearn pipeline.
    Inputs:
        None
    Outputs:
        None
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
             'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__random_state':[42],
         'clf__estimator__max_depth': [3, 10, None]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=99)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    A function used to evaluate a model using sklearns `classificationreport`.
    It predicts classifications for the test set, that get evaluated against the y_test.
    Inputs:
        model: Machine Learning model to be evaluated
        X_test: Array of messages that are  part of the test set.
        y_test: Array of categories that are  part of the test set.
        category_names: List of categories column names
    Outputs:
        None
    """
    y_pred = model.predict(X_test)
    
    for i in range(len(y_test.columns)):
        print(f"Feature #{i+1}: {y_test.columns[i]}")
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
        print("\n")
    return None


def save_model(model, model_filepath):
    """
    A function that saves a given model to a pickle file.
    Input:
        model: A trained Machine Learning model
        model_filepath: String, filepath for the pickle file
    Output:
        None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    return None


def main():
    """
    A functions that
    - Loads a table from a specific database.
    - Builds a Machine Learning model
    - Splits it into X and y s
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        duration = end - start
        print(f'It took {duration} to fit')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()