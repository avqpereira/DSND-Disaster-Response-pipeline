# Disaster Response Pipeline Project


## Table of Contents  
- [General](#general)
- [Files](#files)
- [Technologies](#technologies)
- [Setup](#setup)

<!-- toc -->


## General
This project is a part of [Udacitys](https://www.udacity.com/) Data Science NanoDegree. The projects goal is to build a model for an API that classifies disaster messages from real-life events and, so that NGOs and governments can act accordingly.

There are three main sections in this project:
#### Data preprocessing - ETL
The first part of your data pipeline is the Extract, Transform, and Load process. Here, we read the dataset, clean the data, and then store it in a SQLite database. 

#### Machine Learning Pipeline
Then, the clean dataset get split into a training set and a test set and a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, this model is exported to a pickle file.

#### Flask App
The results are then displayed in a Flask web app, which is connected to the database and pickled model file. The web app contains visualizations and a front-end to categorize new messages.

## Files

    app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    data
    |- disaster_categories.csv # CSV file with the categories for the messages in disaster_messages.csv file
    |- disaster_messages.csv # CSV file with the disaster messages from real-life events
    |- process_data.py # Data preprocessing script
    models
    |- train_classifier.py # Machine Learning Pipeline script
    README.md


## Technologies

- Pandas
- SQLite3
- Pickle
- NLTK
- Scikit-Learn
- Plotly
- Flask
- Joblib

## Setup

Run the following commands in the project's root directory:
```
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
$ cd app
$ python run.py
```

On your web browser go to http://0.0.0.0:3001/.

To quit, press Ctrl + C.
