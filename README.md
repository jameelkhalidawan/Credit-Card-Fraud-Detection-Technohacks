# Credit-Card-Fraud-Detection-Technohacks

**Dataset**
The dataset used is named creditcard.csv.
It contains transactions made by credit cards, including both legitimate and fraudulent transactions.
The dataset features include time, various transaction features, and a binary target variable 'Class' (1 for fraudulent, 0 for legitimate transactions).
you can download the dataset using following link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Prerequisites**

To run this script, you'll need the following libraries installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn (for SMOTE)
xgboost
You can install these libraries using pip:


**Copy code**
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
**Usage**

Clone the repository to your local machine:


Copy code
git clone <repository-url>

**Navigate to the project directory:**


Copy code
cd credit-card-fraud-detection

Place the creditcard.csv dataset in the project directory.

Run the script:


Copy code
python credit_card_fraud_detection.py
**Workflow**

**Data Preprocessing:**

Load the dataset and perform basic data preprocessing steps.
Normalize the feature values using StandardScaler.
Split the data into training and testing sets.
Handling Class Imbalance:

Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes in the training data.
**Model Training:**

Train an XGBoost classifier using RandomizedSearchCV for hyperparameter tuning.
**Model Evaluation:**

Evaluate the model's performance on the test dataset.
Display a classification report, confusion matrix, and ROC-AUC score.
Plot a Precision-Recall Curve and calculate the Average Precision Score.
Plot an ROC Curve.
Plot a Learning Curve to visualize model performance with varying training data sizes.
Results
The script provides insights into the performance of the credit card fraud detection model, including classification metrics, ROC-AUC curve, precision-recall curve, and learning curve. These visualizations and metrics help assess the model's ability to identify fraudulent transactions accurately.

