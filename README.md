# Telco Customer Churn Prediction

This repository contains the project for **Customer Churn Prediction** using the Telco Customer Churn dataset from Kaggle.  
The objective is to analyze customer behaviors and subscription patterns to predict churn and provide actionable insights for business strategies.

## Table of Contents
1. [Link to Notebook](#link-to-notebook)
2. [Project Overview](#project-overview)
3. [Project Objectives](#project-objectives)
4. [Relation to Capstone Project](#relation-to-capstone-project)
   - [How This EDA Supports the Capstone](#how-this-eda-supports-the-capstone)
5. [Methodology](#methodology)
6. [Findings and Recommendations](#findings-and-recommendations)
   - [Key Findings](#key-findings)
   - [Recommendations](#recommendations)
7. [Next Steps](#next-steps)
8. [Repository Structure](#repository-structure)
9. [How to Run](#how-to-run)
10. [Conclusion](#conclusion)

## Link to Notebook
You can view the detailed analysis and results in the [EDA and Model Preparation Notebook](./notebook/eda_and_preparation.ipynb).

## Project Overview
This project aims to predict customer churn for a telecommunications company, enabling proactive retention strategies and improved customer satisfaction. The dataset includes demographics, services, and subscription details, making it a rich source for churn prediction analysis.

## Project Objectives
1. **EDA and Feature Analysis**: Explore and understand the key drivers of churn through visualizations and statistical analysis.
2. **Baseline Model**: Set a benchmark performance using a `DummyClassifier`.
3. **Model Building and Comparison**: Train and evaluate multiple machine learning models, such as Logistic Regression, Random Forest, SVM, and more.
4. **Actionable Insights**: Identify factors influencing churn and provide recommendations for reducing churn rates.

## Relation to Capstone Project

This exploratory data analysis (EDA) and initial report serve as the foundation for the final capstone project: **Federated Learning for Customer Churn Prediction**. The capstone explores how Federated Learning and AutoML can improve churn prediction while preserving data privacy in a decentralized environment.

### How This EDA Supports the Capstone
1. **Data Understanding**: Identifies key patterns and relationships in the Telco Customer Churn dataset that influence customer behavior.
2. **Baseline Models**: Establishes performance benchmarks for centralized models, which will be compared to federated and AutoML approaches in later stages.
3. **Preparation for Decentralized Setup**: Prepares the dataset for partitioning and simulating a decentralized federated learning environment.

This initial analysis ensures a comprehensive understanding of the dataset and provides a strong starting point for implementing Federated Learning and AutoML strategies in the final project.

## Methodology
1. **Data Preprocessing**:
   - Handled missing values and irrelevant features (e.g., dropped `customerID`).
   - Encoded categorical features and scaled numerical features for model training.
2. **EDA**:
   - Used descriptive statistics and visualizations (e.g., histograms, boxplots, heatmaps) to explore relationships between features and churn.
   - Examined correlations between numerical variables and churn.
3. **Modeling**:
   - Established a baseline model.
   - Trained and tuned various models, including Logistic Regression, Random Forest, SVM, and others.
   - Evaluated models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Findings and Recommendations
### Key Findings
- **Key Drivers of Churn**:
  - Customers with month-to-month contracts and electronic check payments are more likely to churn.
  - Short tenure and higher monthly charges are associated with churn.
- **Model Performance**:
  - Logistic Regression had the highest ROC-AUC (0.84), showcasing strong class discrimination.
  - Random Forest achieved the best accuracy (0.79), providing balanced predictions.

### Recommendations
1. **Customer Retention Strategies**:
   - Target customers on month-to-month contracts with incentives to switch to long-term contracts.
   - Identify customers using electronic check payments and encourage them to switch to automated payment methods.
2. **Focus on High-Churn Risk Groups**:
   - Implement loyalty programs or discounts for customers with higher monthly charges.
   - Offer personalized retention campaigns for customers with shorter tenures.

## Next Steps
1. **Hyperparameter Tuning**: Further optimize the best-performing models to improve predictive performance.
2. **Feature Engineering**: Explore interaction terms and non-linear transformations for better model accuracy.
3. **Model Deployment**: Deploy the Random Forest model and monitor its performance in a live environment.

## Repository Structure
- `data/`: Contains the Telco Customer Churn dataset.
- `models/`: Stores trained models and result files (e.g., `Random Forest.joblib`).
- `notebook/`: Includes the Jupyter Notebook with all EDA and model preparation steps.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python libraries.
- `utils.py`: Helper functions for data processing and modeling.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/stirelli/bank-marketing-model-comparison.git

2. Install the necessary packages:

   ```bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook ./notebook/eda_and_preparation.ipynb

## Conclusion
This project highlights the importance of data-driven decision-making for customer retention. By identifying key drivers of churn and leveraging predictive models, the telecom company can proactively address churn risks, improve customer satisfaction, and enhance revenue retention.