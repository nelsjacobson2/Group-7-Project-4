# Group-7-Project-4

# House Price Prediction Project

## Introduction
Welcome to the House Price Prediction Project! This project aims to predict house prices using a machine learning model. The dataset used for this project contains various features of houses and their corresponding sale prices.

## Requirements
To run the project, you will need the following:
- Python 3.x
- Apache Spark
- pyspark library
- pandas library
- scikit-learn library
- Jupyter Notebook

## Installation
1. Install Python 3.x from the official website: https://www.python.org/downloads/
2. Install Apache Spark. You can download it from the official website: https://spark.apache.org/downloads.html
3. Install required Python libraries using pip:

pip install pyspark pandas scikit-learn


## Data
The dataset used for this project is available in the `data` folder. It contains two CSV files: `train.csv` for training the model and `test.csv` for testing the model.

## Code
The main Jupyter Notebook for this project is `house_price_prediction.ipynb`. This notebook performs the following steps:
1. Load the data from CSV files into a Spark DataFrame.
2. Explore the data and handle missing values.
3. Convert categorical features into numerical using OneHotEncoder.
4. Normalize and standardize the numerical features.
5. Split the data into training and testing sets.
6. Train a regression model using Linear Regression.
7. Evaluate the model using Root Mean Squared Error (RMSE) and R-squared.

## Running the Project
To run the project, follow these steps:
1. Make sure you have installed all the required dependencies as mentioned in the "Installation" section.
2. Open Jupyter Notebook and navigate to the project directory.
3. Open the `house_price_prediction.ipynb` notebook.
4. Run the notebook cells to execute the code.

## Model Optimization
The model optimization process involved iterative changes to the model and analyzing the resulting changes in model performance. The final model achieved an RMSE of 31037.94 and an R-squared value of 0.834.

## GitHub Repository
The project's GitHub repository contains all the necessary files and folders. Unnecessary files have been removed, and a .gitignore file is in use to exclude unwanted files from version control.

## Group Presentation
The project presentation covers all the relevant aspects, including data preprocessing, model training, optimization, and evaluation. All group members participated in the presentation.

Feel free to explore the code and data to gain insights into house price prediction. If you have any questions or need further assistance, don't hesitate to reach out.

Happy coding!
