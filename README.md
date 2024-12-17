# Federated Learning for Customer Churn Prediction: A Comparative Analysis with Centralized Models  

This project focuses on predicting customer churn using the **Telco Customer Churn** dataset. The aim is to analyze customer behaviors and subscription patterns, predict churn, and compare performance between centralized and federated learning setups. Below is the structured overview of the project's objectives, methodology, and findings.  

## Table of Contents  

1. [Project Overview](#project-overview)  
2. [Project Objectives](#project-objectives)  
3. [Methodology](#methodology)  
   1. [Data Preprocessing](#data-preprocessing)  
   2. [EDA](#eda)  
   3. [Modeling](#modeling)  
4. [Findings and Recommendations](#findings-and-recommendations)  
   1. [Key Findings](#key-findings)  
   2. [Recommendations](#recommendations)  
5. [Federated Learning Setup](#federated-learning-setup)  
6. [Methodology for Federated Learning](#methodology-for-federated-learning)  
   1. [Client Training](#client-training)  
   2. [Server Aggregation](#server-aggregation)  
   3. [Iterative Process](#iterative-process)  
   4. [Evaluation](#evaluation)  
   5. [Global Model Testing](#global-model-testing)  
7. [Results and Key Findings](#results-and-key-findings)  
   1. [Convergence](#convergence)  
   2. [Performance Comparison](#performance-comparison)  
   3. [Key Observations](#key-observations)  
8. [Next Steps](#next-steps)  
   1. [EDA-Related Steps](#eda-related-steps)  
   2. [Federated Learning-Related Steps](#federated-learning-related-steps)  
9. [Repository Structure](#repository-structure)  
   1. [Data Directory](#data-directory)  
   2. [Federated Learning Directory](#federated-learning-directory)  
   3. [Notebook Directory](#notebook-directory)  
   4. [Project Files](#project-files)  
10. [How to Run the Project](#how-to-run-the-project)  
11. [Conclusion](#conclusion)  

## Project Overview  
This project explores customer churn prediction through **Logistic Regression** and evaluates the benefits of federated learning in preserving data privacy while achieving effective performance. The study also compares results from centralized models with federated learning frameworks to draw meaningful insights.  

## Project Objectives  
1. Perform exploratory data analysis (EDA) to understand key drivers of churn.  
2. Build and optimize a baseline **Logistic Regression model** using centralized data.  
3. Implement a federated learning framework with two clients and a central server.  
4. Compare federated learning outcomes with centralized model performance.  

## Methodology  

### Data Preprocessing  
- Handled missing values and irrelevant features (e.g., dropped `customerID`).  
- Encoded categorical features and scaled numerical features for model training.  

### EDA  
- Used descriptive statistics and visualizations (e.g., histograms, boxplots, heatmaps) to explore relationships between features and churn.  
- Examined correlations between numerical variables and churn.  

### Modeling  
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

## Federated Learning Setup  
1. **Data Partitioning**:  
   - The dataset was divided into two subsets, each representing a unique client.  
2. **Server Configuration**:  
   - A central server was set up to coordinate learning and aggregate model updates.  
3. **Federated Framework**:  
   - Each client trains a local **Logistic Regression model** and sends its coefficients to the server.  
   - The server aggregates coefficients through weighted averaging and redistributes the updated model to clients.  

## Methodology for Federated Learning  

### Client Training  
- Clients train models locally on their respective datasets and share coefficients with the server.  

### Server Aggregation  
- The server computes the weighted average of the coefficients and updates the global model.  

### Iterative Process  
- This process is repeated for 10 rounds, with metrics monitored at each step.  

### Evaluation  
- The federated model's performance was compared against the centralized baseline.  

### Global Model Testing  
- The global model generated by the federated learning process is saved as `federated_learning/server/models/global_model.pkl`.  
- A dedicated notebook, `federated_learning/server/notebook/testing_global_model.ipynb`, was created to validate the global model using test data.  

## Results and Key Findings  

### Convergence  
- The federated model improved over 10 rounds, achieving stable metrics by round 7.  

### Performance Comparison  
- The federated model's performance was slightly below the centralized model:  
  - **Precision**: Federated: 0.82 | Centralized: 0.84  
  - **Recall**: Federated: 0.79 | Centralized: 0.81  
  - **F1-Score**: Federated: 0.80 | Centralized: 0.82  
  - **ROC-AUC**: Federated: 0.84 | Centralized: 0.86  

### Key Observations  
- Federated learning preserved privacy while maintaining strong performance.  
- Data partitioning introduced variability, slightly impacting metrics.  
- Weighted aggregation effectively balanced contributions from clients.  

## Next Steps  

### EDA-Related Steps  
1. **Hyperparameter Tuning**:  
   - Further optimize the best-performing models to improve predictive performance.  
2. **Feature Engineering**:  
   - Explore interaction terms and non-linear transformations for better model accuracy.  

### Federated Learning-Related Steps  
3. **Decentralized Hyperparameter Optimization**:  
   - Implement decentralized hyperparameter tuning across clients to enhance model performance while maintaining privacy.  
4. **Advanced Aggregation Techniques**:  
   - Explore more sophisticated methods to better handle imbalanced or non-iid data.  
5. **Scaling Federated Learning**:  
   - Expand the setup to include additional clients and study its impact on performance and convergence.  

## Repository Structure
- **data/**: Contains the Telco Customer Churn dataset.
- **models/**: Stores trained models and result files (e.g., `Random Forest.joblib`).
- **notebook/**: Includes the Jupyter Notebook with all EDA and model preparation steps.
- `README`: Project documentation.
- `requirements.txt`: List of required Python libraries.

### Federated Learning Directory  
- **client/**:  
  - `data_splitter.py`: Splits data into client-specific subsets.  
  - `federated_client_sklearn_lr.py`: Script to run the federated client with Logistic Regression.  
  - **data/**: Contains `client_0_data.csv` and `client_1_data.csv`.  
  - **metadata/**: Contains `encoder.pkl` and `scaler.pkl` for preprocessing.  
- **server/**:  
  - `federated_server.py`: Runs the federated learning server.  
  - `preprocess_server.py`: Preprocesses data and generates metadata.  
  - **metrics/**: Contains `metrics.json`, which logs the loss, accuracy, precision, recall, F1-score, and other relevant metrics for each round of federated learning. This file is essential for tracking the progress and convergence of the global model across training rounds.
  - **models/**: Stores `global_model.pkl`, the serialized global model generated after federated learning. This file contains the trained Logistic Regression model coefficients aggregated across all federated learning rounds. The global model can be used for further validation or deployment on unseen data to evaluate its generalization.
  - **notebook/**: Contains `testing_global_model.ipynb`, a Jupyter Notebook used to validate the global model's performance. This notebook loads `global_model.pkl` and applies it to a validation dataset to assess accuracy, precision, recall, F1-score, and other key metrics. It ensures that the federated training process was successful and the model is ready for practical applications.

### Notebook Directory  
- `eda_and_preparation.ipynb`: Notebook for EDA and model preparation.  

### Project Files  
- `.gitignore`: Specifies files to ignore in version control.  
- `.python-version`: Python version used in the project.  
- `README.md`: Project documentation.  
- `requirements.txt`: Required Python libraries.  

## How to Run the Project  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/stirelli/federated-churn-ai.git  

2. **Install the necessary packages**:
    ```bash 
    pip install -r requirements.txt

3. **Run the Jupyter Notebook**:
    ```bash 
    jupyter notebook ./notebook/eda_and_preparation.ipynb

4. **Divide the dataset into client-specific datasets**:
    ```bash 
    python federated_learning/client/data_splitter.py

5. **Generate server preprocessing metadata**:
    ```bash 
    python federated_learning/server/preprocess_server.py

6. **Start the Federated Learning server**:
    ```bash 
    python federated_learning/server/federated_server.py

7. **Connect the clients to the server**:
    ```bash 
    python federated_learning/client/federated_client_sklearn_lr.py --client_id 0
    python federated_learning/client/federated_client_sklearn_lr.py --client_id 1

## Conclusion
This project underscores the critical role of data-driven decision-making in improving customer retention. By identifying key drivers of churn and utilizing predictive models, the telecom company can proactively mitigate churn risks, enhance customer satisfaction, and boost revenue retention.

Additionally, the project highlights the potential of federated learning as a privacy-preserving approach to machine learning. Federated models demonstrated performance comparable to centralized methods, providing a practical solution for decentralized data environments.

Future improvements, such as decentralized hyperparameter optimization and advanced aggregation strategies, have the potential to enhance the scalability and performance of the framework further.