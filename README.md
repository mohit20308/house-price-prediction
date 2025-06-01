# Flask Web Application for House Price Prediction Using ML Model

## Objective

To streamline the housing price estimation process for buyers and real estate professionals, developed a web-based application using Flask and a machine learning model that predicts house prices based on user-input features such as number of bedrooms, bathrooms, area size etc.

## Installing Dependencies

The modules are added in requirements.txt. To install run,

	pip install -r requirements.txt

## About Dataset

[**House Sales in King County, USA**](https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/)

The House Sales in King County, USA dataset contains information on house sales in King County, Washington (which includes Seattle and surrounding areas) between May 2014 and May 2015. The task is to predict the price of the house. It can be considered as continuous values (Regression problem).

There are 21,613 rows and 21 columns.

The dataset (kc_house_data.csv) is loaded using pandas read_csv() function.

## Data Preprocessing

Following Data Cleaning & Pre-processing Steps are performed:

- Checking for NaN (Missing Values). The dataset does not consist of null values.
- id and date column is removed from the dataset.
- Checking for duplicate rows, there are no duplicate rows present.
- Only sqft_lot, floors, bedrooms, bathrooms features are used to train a model.
- Standard Scaler is used for standardizing Numerical feature values.

The dataset is split into train and test set using sklearn train_test_split() in the ratio 80:20.

## Features Description

**Target Variable**  
price - Sale price

**Independent Variables**  
sqft_lot - Size of the lot in square feet  
floors - Number of floors  
bedrooms - Number of bedrooms  
bathrooms - Number of bathrooms  

**Datatype**

| variable     | Data Type | Typical Values               |
|--------------|-----------|------------------------------|
|    sqft_lot  |  integer  |   1,000-5,000, 100,000 sq ft        |
|    floors    |  float    |   1.0, 1.5, 2.0, 2.5, 3.0, 3.5        |
|    bedrooms  |  integer  |   1 - 33                             |
|    bathrooms |  float    |   0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 2.75   |


## Model Training

The sklearn LinearRegression model is used.

## Model Evaluation

To assess the model's performance, calculated the Mean Squared Error (MSE) on the test set.

## Predicting Results

For a given user query consisting of lot area, number of floors, bedrooms, bathrooms, house price is predcited.

## Web Application

The web application is built using Flask. Users can input house features such as number of bedrooms, bathrooms, floors, and lot size through a simple web form. Upon submission, the application processes the input, feeds it into a trained machine learning model, and returns the predicted house price in USD.

**Usage**

Run app.py file as:     

    python app.py

## API Structure & Endpoints

**Base-URL**  
http://127.0.0.1:port_no

**Base Endpoint**  
This endpoint displays the form for user input.

**Request**  
Method: GET  
Path: Base-URL

**Response**  
Type: None  
Purpose: Displays the prediction form to the user.  
Trigger: Accessing the page directly via URL (e.g., clicking a "Predict" button or visiting /predict)

**House Price Prediction Endpoint**  
This endpoint provides the predicted house price.

**Request**  
Method: POST  
Path: Base-URL/predict

**Response**  
Type: HTML (text/html)  
Status Code: 200 OK (if successful)  
Content Returned: Renders the predict.html template with the predicted price embedded.

## Demo Run

- The following image shows GET request.

<kbd>![](/README_images/PART1.png)</kbd>

- The following image shows the predicted house price based on user input.

<kbd>![](/README_images/PART2.png)</kbd>


### References

- https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

---


