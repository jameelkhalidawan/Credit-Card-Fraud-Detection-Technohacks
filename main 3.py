import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Data preprocessing
X = data.iloc[:, 1:-1].values  # Exclude 'Time' column and 'Class' column
y = data['Class'].values       # Target variable

# Data exploration
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
plt.show()

# Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handling class imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Model training with RandomizedSearchCV for hyperparameter tuning
param_dist = {
    'n_estimators': np.arange(50, 300, 10),
    'max_depth': np.arange(3, 10),
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4]
}

xgb = XGBClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=50, cv=StratifiedKFold(n_splits=5),
                                   n_jobs=-1, verbose=2, scoring='f1_macro', random_state=42)
random_search.fit(X_resampled, y_resampled)

best_xgb = random_search.best_estimator_

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(best_xgb, X_resampled, y_resampled, cv=StratifiedKFold(n_splits=5), n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

# Model evaluation
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]

print("Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Precision-Recall Curve and Average Precision Score
precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(12, 6))

# Plot Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Avg Precision = {:.2f})'.format(avg_precision))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()

# Learning curve plot
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-o', label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'g-o', label='Validation Accuracy')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
