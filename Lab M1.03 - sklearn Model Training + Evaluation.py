"""
# Breast Cancer Prediction with KNN
## Author: Javier Romero
## Description: Predict whether a breast tumor is malignant or benign using KNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

### Step 1 - Data Loading and Exploration
# Load the breast cancer dataset
print("Loading breast cancer dataset...")
cancer_data = load_breast_cancer()

# The dataset is a Bunch object with 'data', 'target', 'feature_names', etc.
print(f"\nDataset type: {type(cancer_data)}")
print(f"Number of samples: {len(cancer_data.data)}")
print(f"Number of features: {len(cancer_data.feature_names)}")
print(f"Target classes: {cancer_data.target_names}")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget distribution:")
print(df['target'].value_counts())
print(f"Malignant (1): {(df['target'] == 1).sum()}")
print(f"Benign (0): {(df['target'] == 0).sum()}")

# Basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

### Step 2 - Data Splitting
# Visualize feature distributions (select a few key features)
print("\n" + "="*50)
print("FEATURE DISTRIBUTIONS")
print("="*50)

# Select a few representative features to visualize
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    axes[idx].hist(df[df['target'] == 0][feature], alpha=0.5, label='Benign', bins=30)
    axes[idx].hist(df[df['target'] == 1][feature], alpha=0.5, label='Malignant', bins=30)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {feature}')
    axes[idx].legend()

### Had to comment this out to prevent potential timeouts during execution
#plt.tight_layout()
#plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
#print("Saved visualization to 'feature_distributions.png'")
#plt.show()


### Step 3 - Splitting the Data
# Separate features and target
X = df.drop('target', axis=1)  # All columns except 'target'
y = df['target']  # Target column

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into training and testing sets
# random_state ensures reproducibility
# stratify=y ensures both sets have similar class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("\n" + "="*50)
print("DATA SPLIT")
print("="*50)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Verify class distribution in both sets
print("\nTraining set target distribution:")
print(y_train.value_counts())
print(f"  Benign (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

print("\nTest set target distribution:")
print(y_test.value_counts())
print(f"  Benign (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

### Step 4 - Training the KNN Model

# Create KNN classifier
# n_neighbors=5 means the model will look at the 5 nearest neighbors to make a prediction

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")

# Make predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f"\nTraining predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")


### Step 5 - Making predictiions

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Correct': y_test.values == y_test_pred
})
print("DATA COMPARISON:")
print(comparison)

correct = (y_test.values == y_test_pred).sum()
total = len(y_test)
print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
print("\nFirst 10 predictions comparison:")
print(comparison.head(10))


### Step 6 - Evaluating Model Performance


# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("=== Model Performance ===")
print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\nTest Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

print("\n=== Confusion Matrix ===")
print("                Predicted")
print("              Benign  Malignant")
print(f"Actual Benign    {test_confusion[0,0]:4d}      {test_confusion[0,1]:4d}")
print(f"      Malignant  {test_confusion[1,0]:4d}      {test_confusion[1,1]:4d}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred, target_names=cancer_data.target_names))

### Step 7 - Experimenting with Different Values

# Experiment with different K values
print("\n" + "="*50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    # Create and train model
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    
    # Make predictions
    y_pred_temp = knn_temp.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    
    results.append({
        'K': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec
    })
    
    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

# Find best K
results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df['Accuracy'].idxmax(), 'K']
print(f"\nBest K value: {best_k} (Accuracy: {results_df['Accuracy'].max():.4f})")

# Visualize results
#plt.figure(figsize=(10, 6))
#plt.plot(results_df['K'], results_df['Accuracy'], marker='o', label='Accuracy')
#plt.plot(results_df['K'], results_df['Precision'], marker='s', label='Precision')
#plt.plot(results_df['K'], results_df['Recall'], marker='^', label='Recall')
#plt.xlabel('K (Number of Neighbors)')
#plt.ylabel('Score')
#plt.title('KNN Performance vs K Value')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.savefig('knn_k_comparison.png', dpi=150, bbox_inches='tight')
#print("\nSaved visualization to 'knn_k_comparison.png'")
#plt.show()


################################################################################
# PART 2: Telco Customer Churn Prediction with KNN
################################################################################
"""
## Description: Predict whether a customer will churn using KNN
## Dataset: Telco Customer Churn
"""

from sklearn.preprocessing import LabelEncoder, StandardScaler

### Step 1 - Data Loading and Exploration
print("\n" + "="*60)
print("PART 2: TELCO CUSTOMER CHURN PREDICTION")
print("="*60)

print("\n" + "="*50)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*50)

# Load the Telco Customer Churn dataset
print("\nLoading Telco Customer Churn dataset...")
df_churn = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"\nDataset shape: {df_churn.shape}")
print(f"Number of samples: {len(df_churn)}")
print(f"Number of features: {len(df_churn.columns)}")

print("\nColumn names:")
print(df_churn.columns.tolist())

print("\nFirst few rows:")
print(df_churn.head())

print("\nDataset info:")
print(df_churn.info())

print("\nData types:")
print(df_churn.dtypes)

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing_churn = df_churn.isnull().sum()
if missing_churn.sum() == 0:
    print("No null values found in standard check")
else:
    print(missing_churn[missing_churn > 0])

# Check for empty strings or whitespace (common in CSV files)
print("\nChecking for empty strings or whitespace...")
for col in df_churn.columns:
    empty_count = (df_churn[col].astype(str).str.strip() == '').sum()
    if empty_count > 0:
        print(f"  {col}: {empty_count} empty values")

# Check target distribution
print("\n" + "="*50)
print("TARGET DISTRIBUTION (Churn)")
print("="*50)
print(df_churn['Churn'].value_counts())
print(f"\nNo (Stayed): {(df_churn['Churn'] == 'No').sum()} ({(df_churn['Churn'] == 'No').mean()*100:.1f}%)")
print(f"Yes (Churned): {(df_churn['Churn'] == 'Yes').sum()} ({(df_churn['Churn'] == 'Yes').mean()*100:.1f}%)")

# Basic statistics for numeric columns
print("\n" + "="*50)
print("BASIC STATISTICS (Numeric Columns)")
print("="*50)
print(df_churn.describe())

# Visualize key features
print("\n" + "="*50)
print("FEATURE DISTRIBUTIONS")
print("="*50)

key_features_churn = ['tenure', 'MonthlyCharges']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, feature in enumerate(key_features_churn):
    data_no_churn = pd.to_numeric(df_churn[df_churn['Churn'] == 'No'][feature], errors='coerce')
    data_churn = pd.to_numeric(df_churn[df_churn['Churn'] == 'Yes'][feature], errors='coerce')

    axes[idx].hist(data_no_churn.dropna(), alpha=0.5, label='No Churn', bins=30)
    axes[idx].hist(data_churn.dropna(), alpha=0.5, label='Churn', bins=30)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {feature} by Churn')
    axes[idx].legend()

#commented out to prevent potential timeouts during execution
#plt.tight_layout()
#plt.savefig('churn_feature_distributions.png', dpi=150, bbox_inches='tight')
#print("Saved visualization to 'churn_feature_distributions.png'")
#plt.show()


### Step 2 - Data Preprocessing
print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING")
print("="*50)

# Create a copy for preprocessing
df_processed = df_churn.copy()

# Drop customerID (not useful for prediction)
print("\nDropping customerID column...")
df_processed = df_processed.drop('customerID', axis=1)

# Handle TotalCharges - convert to numeric (contains some whitespace values)
print("\nConverting TotalCharges to numeric...")
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# Check for missing values after conversion
missing_total = df_processed['TotalCharges'].isnull().sum()
print(f"Missing values in TotalCharges after conversion: {missing_total}")

# Fill missing TotalCharges with median
if missing_total > 0:
    median_total = df_processed['TotalCharges'].median()
    df_processed['TotalCharges'].fillna(median_total, inplace=True)
    print(f"Filled missing values with median: {median_total:.2f}")

# Identify categorical columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Remove target from list
print(f"\nCategorical columns to encode: {categorical_cols}")

# Encode categorical variables using LabelEncoder
print("\nEncoding categorical variables...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

# Convert target variable to binary (0/1)
print("\nConverting target variable (Churn) to binary...")
df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
print(f"  No -> 0, Yes -> 1")

# Drop any remaining rows with NaN values
print("\nChecking for remaining NaN values...")
nan_count_before = df_processed.isnull().sum().sum()
print(f"Total NaN values before dropping: {nan_count_before}")
df_processed = df_processed.dropna()
print(f"Rows after dropping NaN: {len(df_processed)}")

print("\nProcessed dataset info:")
print(df_processed.info())

print("\nFirst few rows after preprocessing:")
print(df_processed.head())


### Step 3 - Splitting the Data
print("\n" + "="*50)
print("STEP 3: SPLITTING THE DATA")
print("="*50)

# Separate features and target
X_churn = df_processed.drop('Churn', axis=1)
y_churn = df_processed['Churn']

print(f"Features shape: {X_churn.shape}")
print(f"Target shape: {y_churn.shape}")
print(f"\nFeature columns: {X_churn.columns.tolist()}")

# Feature scaling (important for KNN)
print("\nApplying feature scaling (StandardScaler)...")
scaler = StandardScaler()
X_churn_scaled = scaler.fit_transform(X_churn)  # Returns numpy array

print("Scaling complete - features now have mean=0 and std=1")

# Convert target to numpy array to ensure alignment
y_churn_array = y_churn.values

# Split into training and testing sets (using numpy arrays)
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn_scaled, y_churn_array,
    test_size=0.2,
    random_state=42,
    stratify=y_churn_array
)

print("\n" + "="*50)
print("DATA SPLIT")
print("="*50)
print(f"Training set size: {X_train_churn.shape[0]} samples")
print(f"Test set size: {X_test_churn.shape[0]} samples")
print(f"Training features: {X_train_churn.shape[1]}")
print(f"Test features: {X_test_churn.shape[1]}")

# Verify class distribution in both sets
print("\nTraining set target distribution:")
print(f"  No Churn (0): {(y_train_churn == 0).sum()} ({(y_train_churn == 0).mean()*100:.1f}%)")
print(f"  Churn (1): {(y_train_churn == 1).sum()} ({(y_train_churn == 1).mean()*100:.1f}%)")

print("\nTest set target distribution:")
print(f"  No Churn (0): {(y_test_churn == 0).sum()} ({(y_test_churn == 0).mean()*100:.1f}%)")
print(f"  Churn (1): {(y_test_churn == 1).sum()} ({(y_test_churn == 1).mean()*100:.1f}%)")


### Step 4 - Training the KNN Model
print("\n" + "="*50)
print("STEP 4: TRAINING THE KNN MODEL")
print("="*50)

# Create KNN classifier with k=5
knn_churn = KNeighborsClassifier(n_neighbors=5)
knn_churn.fit(X_train_churn, y_train_churn)

print("KNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn_churn.n_neighbors}")

# Make predictions
y_train_pred_churn = knn_churn.predict(X_train_churn)
y_test_pred_churn = knn_churn.predict(X_test_churn)

print(f"\nTraining predictions: {len(y_train_pred_churn)}")
print(f"Test predictions: {len(y_test_pred_churn)}")


### Step 5 - Making Predictions and Evaluating
print("\n" + "="*50)
print("STEP 5: PREDICTIONS AND EVALUATION")
print("="*50)

# Compare actual vs predicted
comparison_churn = pd.DataFrame({
    'Actual': y_test_churn,
    'Predicted': y_test_pred_churn,
    'Correct': y_test_churn == y_test_pred_churn
})

correct_churn = (y_test_churn == y_test_pred_churn).sum()
total_churn = len(y_test_churn)
print(f"\nCorrect predictions: {correct_churn}/{total_churn} ({correct_churn/total_churn*100:.1f}%)")

print("\nFirst 10 predictions comparison:")
print(comparison_churn.head(10))

# Calculate metrics
train_accuracy_churn = accuracy_score(y_train_churn, y_train_pred_churn)
test_accuracy_churn = accuracy_score(y_test_churn, y_test_pred_churn)
test_precision_churn = precision_score(y_test_churn, y_test_pred_churn)
test_recall_churn = recall_score(y_test_churn, y_test_pred_churn)
test_confusion_churn = confusion_matrix(y_test_churn, y_test_pred_churn)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"\nTraining Accuracy: {train_accuracy_churn:.4f} ({train_accuracy_churn*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy_churn:.4f} ({test_accuracy_churn*100:.2f}%)")
print(f"\nTest Precision: {test_precision_churn:.4f}")
print(f"Test Recall: {test_recall_churn:.4f}")

print("\n=== Confusion Matrix ===")
print("                Predicted")
print("              No Churn  Churn")
print(f"Actual No Churn  {test_confusion_churn[0,0]:4d}      {test_confusion_churn[0,1]:4d}")
print(f"       Churn     {test_confusion_churn[1,0]:4d}      {test_confusion_churn[1,1]:4d}")

print("\n=== Interpretation ===")
print(f"True Negatives (correctly predicted No Churn): {test_confusion_churn[0,0]}")
print(f"False Positives (incorrectly predicted Churn): {test_confusion_churn[0,1]}")
print(f"False Negatives (missed Churners): {test_confusion_churn[1,0]}")
print(f"True Positives (correctly predicted Churn): {test_confusion_churn[1,1]}")

print("\n=== Classification Report ===")
print(classification_report(y_test_churn, y_test_pred_churn, target_names=['No Churn', 'Churn']))


### Step 6 - Experimenting with Different K Values
print("\n" + "="*50)
print("STEP 6: EXPERIMENTING WITH DIFFERENT K VALUES")
print("="*50)

k_values_churn = [1, 3, 5, 7, 9, 11, 15]
results_churn = []

for k in k_values_churn:
    # Create and train model
    knn_temp_churn = KNeighborsClassifier(n_neighbors=k)
    knn_temp_churn.fit(X_train_churn, y_train_churn)

    # Make predictions
    y_pred_temp_churn = knn_temp_churn.predict(X_test_churn)

    # Calculate metrics
    acc = accuracy_score(y_test_churn, y_pred_temp_churn)
    prec = precision_score(y_test_churn, y_pred_temp_churn)
    rec = recall_score(y_test_churn, y_pred_temp_churn)

    results_churn.append({
        'K': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec
    })

    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

# Find best K based on different metrics
results_df_churn = pd.DataFrame(results_churn)
best_k_acc_churn = results_df_churn.loc[results_df_churn['Accuracy'].idxmax(), 'K']
best_k_recall_churn = results_df_churn.loc[results_df_churn['Recall'].idxmax(), 'K']

print(f"\nBest K for Accuracy: {int(best_k_acc_churn)} (Accuracy: {results_df_churn['Accuracy'].max():.4f})")
print(f"Best K for Recall: {int(best_k_recall_churn)} (Recall: {results_df_churn['Recall'].max():.4f})")

# Visualize results
#plt.figure(figsize=(10, 6))
#plt.plot(results_df_churn['K'], results_df_churn['Accuracy'], marker='o', label='Accuracy', linewidth=2)
#plt.plot(results_df_churn['K'], results_df_churn['Precision'], marker='s', label='Precision', linewidth=2)
#plt.plot(results_df_churn['K'], results_df_churn['Recall'], marker='^', label='Recall', linewidth=2)
#plt.xlabel('K (Number of Neighbors)')
#plt.ylabel('Score')
#plt.title('KNN Performance vs K Value - Churn Prediction')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.xticks(k_values_churn)
#plt.savefig('churn_knn_k_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to 'churn_knn_k_comparison.png'")
#plt.show()


### Step 7 - Analysis and Recommendations
print("\n" + "="*50)
print("STEP 7: ANALYSIS AND RECOMMENDATIONS")
print("="*50)

# Retrain with best K
best_k_final = int(best_k_acc_churn)
knn_best_churn = KNeighborsClassifier(n_neighbors=best_k_final)
knn_best_churn.fit(X_train_churn, y_train_churn)
y_pred_best_churn = knn_best_churn.predict(X_test_churn)

final_accuracy_churn = accuracy_score(y_test_churn, y_pred_best_churn)
final_precision_churn = precision_score(y_test_churn, y_pred_best_churn)
final_recall_churn = recall_score(y_test_churn, y_pred_best_churn)

print(f"\n=== FINAL MODEL PERFORMANCE (K={best_k_final}) ===")
print(f"Accuracy: {final_accuracy_churn:.4f} ({final_accuracy_churn*100:.2f}%)")
print(f"Precision: {final_precision_churn:.4f}")
print(f"Recall: {final_recall_churn:.4f}")

print("\n=== KEY FINDINGS ===")
print(f"""
1. MODEL ACCURACY: {final_accuracy_churn*100:.1f}%
   - The model correctly predicts customer churn about {final_accuracy_churn*100:.0f}% of the time.

2. PRECISION: {final_precision_churn*100:.1f}%
   - When the model predicts a customer will churn, it's correct {final_precision_churn*100:.0f}% of the time.
   - This helps avoid wasting resources on false alarms.

3. RECALL: {final_recall_churn*100:.1f}%
   - The model catches {final_recall_churn*100:.0f}% of actual churners.
   - This is important because missing a churner means losing a customer.

4. CLASS IMBALANCE:
   - The dataset has more non-churners than churners (imbalanced).
   - This can affect model performance on the minority class (churners).
""")

print("\n=== FEATURE ANALYSIS ===")
print("""
Based on exploratory analysis, key factors potentially influencing churn:

1. TENURE:
   - Customers with shorter tenure tend to churn more.
   - New customers need extra attention and engagement.

2. MONTHLY CHARGES:
   - Higher monthly charges may correlate with higher churn.
   - Consider pricing strategies for at-risk customers.

3. CONTRACT TYPE:
   - Month-to-month contracts typically have higher churn.
   - Incentivize longer-term contracts.

4. INTERNET SERVICE:
   - Fiber optic customers may have different churn patterns.
   - Quality of service matters.
""")

print("\n=== RECOMMENDATIONS FOR THE COMPANY ===")
print("""
1. CUSTOMER RETENTION STRATEGY:
   - Focus on customers in their first year (low tenure).
   - Offer loyalty programs or discounts for staying.

2. PRICING REVIEW:
   - Analyze if high monthly charges drive churn.
   - Consider competitive pricing or value-added services.

3. CONTRACT INCENTIVES:
   - Encourage annual/two-year contracts with benefits.
   - Reduce month-to-month churn through special offers.

4. EARLY WARNING SYSTEM:
   - Use this model to identify at-risk customers.
   - Proactive outreach before they churn.

5. SERVICE QUALITY:
   - Ensure technical support and service quality.
   - Address issues before they lead to cancellation.
""")

print("\n=== MODEL LIMITATIONS ===")
print(f"""
1. IMBALANCED DATA:
   - {(y_churn_array == 0).sum()} non-churners vs {(y_churn_array == 1).sum()} churners.
   - Model may be biased toward predicting non-churn.

2. FEATURE ENGINEERING:
   - Simple encoding may lose information.
   - More sophisticated feature engineering could help.

3. KNN LIMITATIONS:
   - Computationally expensive for large datasets.
   - Sensitive to irrelevant features.
   - Doesn't provide feature importance directly.

4. TEMPORAL ASPECTS:
   - Model doesn't capture time-series patterns.
   - Customer behavior changes over time.

5. POSSIBLE IMPROVEMENTS:
   - Try other algorithms (Random Forest, XGBoost, Neural Networks).
   - Use SMOTE or other techniques for class imbalance.
   - Cross-validation for more robust evaluation.
   - Feature selection to identify most important predictors.
""")

print("\n" + "="*60)
print("PART 2 ANALYSIS COMPLETE")
print("="*60)
