# Disaster Response Pipeline Project

## Description
This project is a part of [Udacitys](https://www.udacity.com/) Data Science NanoDegree. The project consists of tweets from real-life disasters and the task here is to use NLP to correctly categorize the messages, so that NGOs and governments can act accordingly.
There are three main sections in this project:
- Data preprocessing
- Machine Learning Pipeline, to build, train and save a model
- A web app to display some visualizations.

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
