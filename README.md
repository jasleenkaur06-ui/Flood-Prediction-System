# 🌊 Flood Prediction System Using Machine Learning

## 1. Project Overview

The Flood Prediction System is a machine learning-based application that estimates flood probability scores using environmental, geographical, infrastructure, and human-related factors.

The application uses a trained Linear Regression model and provides an interactive dashboard developed using Streamlit.

## 2. Objectives

- Estimate flood probability scores using machine learning.
- Analyse different factors associated with flood prediction.
- Provide an interactive interface for entering input values.
- Display model performance metrics.
- Maintain and download prediction history.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 4. Dataset

**Dataset:** Flood Prediction Dataset  
**Source:** Kaggle  
**Author:** naiyakhalid

The dataset contains 20 input features and one target variable, `FloodProbability`.

The input features include:
- Monsoon Intensity
- Topography Drainage
- River Management
- Deforestation
- Urbanization
- Climate Change
- Dams Quality
- Siltation
- Agricultural Practices
- Encroachments
- Disaster Preparedness
- Drainage Systems
- Coastal Vulnerability
- Landslides
- Watersheds
- Deteriorating Infrastructure
- Population Score
- Wetland Loss
- Inadequate Planning
- Political Factors

## 5. Machine Learning Model

Linear Regression is used to predict the continuous flood probability score.

The dataset is divided into training and testing sets using an 80:20 split.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | 0.01579 |
| RMSE | 0.02008 |
| R² Score | 0.84488 |

## 6. Application Features

- Interactive Streamlit dashboard
- 20 input parameters
- Flood probability score prediction
- Input factor visualization
- Feature influence analysis
- Model performance analysis
- Prediction history
- Individual prediction deletion
- CSV history download

## 7. Project Structure

```text
FloodPrediction/
│
├── app.py
├── model.py
├── explore_data.py
├── train.csv
├── flood_model.pkl
├── features.pkl
├── random_forest_model.pkl
├── prediction_history.csv
├── requirements.txt
└── README.md
```

## 8. Installation and Execution

### Step 1: Install Python

Install Python on your system.

### Step 2: Install Required Libraries

Open the terminal inside the project folder and execute:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

Execute:

```bash
python -m streamlit run app.py
```

### Step 4: Open the Dashboard

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 9. Limitations

- Predictions depend on the dataset and the trained model.
- Input values represent dataset-based scores, not direct real-world measurements.
- The output is an estimated dataset-based probability score, not an official flood warning.
- Real-world flood forecasting requires additional real-time environmental and geographical data.

## 10. Conclusion

This project demonstrates the application of machine learning in estimating flood probability scores. It combines a trained regression model with an interactive dashboard to make predictions and model analysis accessible to users.

**Project Type:** Machine Learning – Regression  
**Interface:** Streamlit  
**Model:** Linear Regressiondone
