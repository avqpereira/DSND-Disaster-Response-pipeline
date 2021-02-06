# Disaster Response Pipeline Project

## Description
This project is a part of [Udacitys](https://www.udacity.com/) Data Science NanoDegree. The project consists of tweets from real-life disasters and the task here is to use NLP to correctly categorize the messages, so that NGOs and governments can act accordingly.
There are three main sections in this project:
- Data preprocessing
- Machine Learning Pipeline, to build, train and save a model
- A web app to display some visualizations.

## Repo Structure

    app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py # Data preprocessing script
    |- InsertDatabaseName.db # database to save clean data to
    models
    |- train_classifier.py # Machine Learning Pipeline script
    |- classifier.pkl # saved model
    README.md


## Installing

git clone https://github.com/avqpereira/DSND-Disaster-Response-pipeline.git

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
