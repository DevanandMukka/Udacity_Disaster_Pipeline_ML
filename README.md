# Udacity_Disaster_Pipeline_ML
Udacity Data science project for Disaster Management pipeline

### Table of Contents

 * [Installation](#Installation)

 * [Project Overview](#Project_Motivation)

 * [File Descriptions](#File_Descriptions)
 
 * [Approach](#Approach)

 * [Results](#Results)

 * [Licensing, Authors, and Acknowledgements](#Licensing)
<a name="Installation"></a>
#### Installation 

It is not necessary to install any libraries outside of the Anaconda suit of python. Everything is built in Python 3. version only
<a name="Project_Motivation"></a>
#### Project Overview 

This project is to mainly deal with the emergency messaging system and categorizing them in to appropriate sections, such that relavant rescue or operation teams can go and control the situation.

In this project, i have applied the data engineering techniques to read , analyze and segment the data from Figure Eight with the help of a model for an API.

#### File Descriptions 

Messages: List of emergency messages 

Categories: Probable categories of the emergency messages (Primary id will be ID)

#### Approach

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv DisasterResponse.db

To run ML pipeline that trains classifier and saves python ./models/train_classifier.py ./DisasterResponse.db classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

How to see the APP -

If you are working in Udacity workspace : Kindly follow this link : https://knowledge.udacity.com/questions/20899

If you are running in the local machine : Go to http://0.0.0.0:3001/


#### Results 

Subject to the succesful execution of the code, you can able to get the information in the final app about the segment of emergency thus making easy for any organization to take steps.

#### Licensing, Authors, Acknowledgements 

Udacity :- Data science nano degree project 
