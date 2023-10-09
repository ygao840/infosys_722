# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 01 - BU
"""
Describe the business objectives.
"""

# 02 - DU
"""
Load your data.
Explore the data.
Visializations on raw data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the XPT file
df1 = pd.read_sas('LLCP2021.XPT')
df = df1

# Read the CSV file into a DataFrame
df_cvd = pd.read_csv('CVD_cleaned.csv')
df_heart = pd.read_csv('heart.csv')

# Cross-tabulation between CVDINFR4 and INCOME3
ct = pd.crosstab(df['INCOME3'], df['CVDINFR4'])

# Plot 1
ct.plot(kind='bar', stacked=True)
plt.title('Relationship between CVDINFR4 and INCOME3')
plt.ylabel('Count')
plt.xlabel('Income Category')
plt.xticks(rotation=0)
plt.show()

# Box plot to visualize relationship
sns.boxplot(x='Heart_Disease', y='FriedPotato_Consumption', data=df_cvd)
plt.title('Boxplot of FriedPotato_Consumption by Heart_Disease Status')
plt.ylabel('FriedPotato_Consumption')
plt.xlabel('Heart_Disease Status')
plt.show()

sns.boxplot(x='Heart_Disease', y='Weight_(kg)', data=df_cvd)
plt.title('Boxplot of Weight_(kg) by Heart_Disease Status')
plt.ylabel('Weight_(kg)')
plt.xlabel('Heart_Disease Status')
plt.show()

# Cross-tabulation between cp and target
ct = pd.crosstab(df_heart['cp'], df_heart['target'])

# Plot
ct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Relationship between Chest Pain Type and Heart Disease Status')
plt.ylabel('Count')
plt.xlabel('Chest Pain Type')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Asymptomatic', 'Atypical Angina', 'Non-anginal Pain', 'Typical Angina'], rotation=0)
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

def verify_data_integrity(df):
    total_values = df.size
    # For missing values
    missing_values_count = df.isnull().sum()
    missing_values_percentage = (missing_values_count / len(df)) * 100
    total_missing_values_count = missing_values_count.sum()
    total_missing_values_percentage = (total_missing_values_count / total_values) * 100
    print(f"Percentage of Missing Values in the Entire Dataset: {total_missing_values_percentage:.2f}%")
    print(f"Percentage of Data integrity in the Entire Dataset: {100 - total_missing_values_percentage:.2f}%")

    # For outliers using IQR
    total_outliers = 0
    outliers = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers[column] = df[outlier_mask].shape[0]
        total_outliers += df[outlier_mask].shape[0]
    outliers_percentage = {column: (count/len(df))*100 for column, count in outliers.items()}
    total_outliers_percentage = (total_outliers / total_values) * 100
    print(f"Percentage of Extreme Values in the Entire Dataset: {total_outliers_percentage:.2f}%")

    # For extreme values using IQR
    total_extreme_values = 0
    extreme_values = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.15)
        Q3 = df[column].quantile(0.85)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        extreme_values[column] = df[outlier_mask].shape[0]
        total_extreme_values += df[outlier_mask].shape[0]
    extreme_values_percentage = {column: (count/len(df))*100 for column, count in extreme_values.items()}
    total_extreme_values_percentage = (total_extreme_values / total_values) * 100
    print(f"Percentage of Extreme Values in the Entire Dataset: {total_extreme_values_percentage:.2f}%")
    
    # Combine results into one DataFrame
    results = pd.DataFrame({
        'Missing Values': missing_values_count,
        'Missing Values (%)': missing_values_percentage,
        'Outliers': pd.Series(outliers),
        'Outliers (%)': pd.Series(outliers_percentage),
        'Extreme Values': pd.Series(extreme_values),
        'Extreme Values (%)': pd.Series(extreme_values_percentage)
    })
    
    return results

df_results = verify_data_integrity(df)
df_des = df.describe()
df_corr = round(df.corr().loc['CVDINFR4'],2)
df_corr.describe()

df_cvd_results = verify_data_integrity(df_cvd)
df_cvd_des = df_cvd.describe()

df_heart_results = verify_data_integrity(df_heart)
df_heart_results_des = df_heart_results.describe()
df_heart_results_corr = round(df_heart.corr().loc['target'],2)
df_heart_results_corr.describe()

df_des[['HEIGHT3','WEIGHT2','FLSHTMY3']]

# 03 - DP
"""
Preprocessing of the data.
"""
# List of columns to drop
columns_to_drop = [
    'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 
    'SEQNO', '_PSU', 'CTELENM1', 'PVTRESD1', 'COLGHOUS', 'STATERE1', 'CELPHON1'
]
# Drop the columns
df = df.drop(columns=columns_to_drop)

# Calculate the percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100
# Identify columns to drop (those with more than 29% missing values)
columns_to_drop = missing_percentage[missing_percentage > 29].index.tolist()
# Drop the columns
df.drop(columns=columns_to_drop, inplace=True)

df_des1 = df.describe()

# List of columns to drop
columns_to_drop = [
    '_STATE', 'SAFETIME', 'CTELNUM1', 'CELLFON5', 'CADULT1'
]
# Drop the columns
df = df.drop(columns=columns_to_drop)

df_des2 = df.describe()
df_results1 = verify_data_integrity(df)

df2 = df

df3 = df2
df3.replace([88, 555, 888], 0, inplace=True)

# Calculate Q1, Q3, and IQR for each column
Q1 = df.quantile(0)
Q3 = df.quantile(0.85)
IQR = Q3 - Q1

# Define bounds for extreme values
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter rows without extreme values
df3 = df[~((df > upper_bound)).any(axis=1)]
df_des3 = df3.describe()
df3_results = verify_data_integrity(df3)

# Remove columns where number of unique values is 1
df3 = df3.loc[:, df3.nunique() != 1]
df_des3 = df3.describe()

df3_results = verify_data_integrity(df3)

# Calculate the percentage of missing values for each column
missing_percentage = df3.isnull().mean() * 100
# Identify columns to drop (those with more than 5% missing values)
columns_to_drop = missing_percentage[missing_percentage > 5].index.tolist()
# Drop the columns
df3.drop(columns=columns_to_drop, inplace=True)

df3_results = verify_data_integrity(df3)

# Drop rows where any of the specified columns have missing values
df3 = df3.dropna(subset=['CHOLMED3', 'TOLDHI3', '_RFCHOL3'])


for column in df3.columns:
    missing = df3[column].isnull()
    num_missing = missing.sum()
    
    if num_missing > 0:
        # Sample from the existing values to fill the NaNs
        samples = df3[column].dropna().sample(num_missing, replace=True)
        samples.index = df3[missing].index
        df3.loc[missing, column] = samples

df3_results = verify_data_integrity(df3)


df3_type = df3.dtypes
df3 = df3.astype(int)
df3_type = df3.dtypes

# Plot the histogram
df3['_AGE80'].hist(bins=20, edgecolor='black', alpha=0.7)

# Title and labels
plt.title('Distribution of _AGE80')
plt.xlabel('Age')
plt.ylabel('Number of People')

# Display the plot
plt.show()

# Define the age bins
bins = [18, 30, 45, 60, 75, float('inf')]
labels = ['young', 'matured young people', 'middle-aged people', 'old people', 'grand people']
# Create a new column 'age_group' with classifications based on '_AGE80'
df3['age_group'] = pd.cut(df3['_AGE80'], bins=bins, labels=labels, right=False)


# Count occurrences of each age group
age_group_counts = df3['age_group'].value_counts()

# Create the bar plot
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)

# Title and labels
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of People')
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()


# Define the mapping
age_group_mapping = {
    'young': 1,
    'matured young people': 2,
    'middle-aged people': 3,
    'old people': 4,
    'grand people': 5
}

# Apply the mapping to the 'age_group' column
df3['age_group'] = df3['age_group'].map(age_group_mapping)

df3 = df3.astype(int)





# 1. Add an 'ID' column to df
df['ID'] = range(1, len(df) + 1)

# 2. Split df into df_a and df_b
cols_a = ['ID'] + list(df.columns[:-1])[:200000]  # ID + first 200,000 columns
cols_b = ['ID'] + list(df.columns[:-1])[200000:] # ID + remaining columns

df_a = df[cols_a]
df_b = df[cols_b]

# 3. Merge df_a and df_b on 'ID' to get df_s
df_s = pd.merge(df_a, df_b, on='ID')

# Drop 'ID' from df and df_s for comparison
df.drop(columns='ID', inplace=True)
df_s.drop(columns='ID', inplace=True)

# 4. Validate if df_s is the same as df
if df.equals(df_s):
    print("df_s is the same as df")
else:
    print("df_s and df are different")


# 04 - DT
"""
Removing variables, projection of variables.
"""
from scipy.stats import pearsonr

# Get the list of all features excluding the target variable
features = df3.columns.tolist()
features.remove('CVDINFR4')

# Create an empty list to store the results
results = []

# Loop through all the features and calculate the Pearson correlation and p-value
for feature in features:
    # Skip non-numeric columns
    if pd.api.types.is_numeric_dtype(df3[feature]):
        corr, p_value = pearsonr(df3[feature].dropna(), df3['CVDINFR4'].dropna())
        results.append({'feature': feature, 'correlation': corr, 'p_value': p_value})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Display the results sorted by the absolute correlation values in descending order
results_df['abs_correlation'] = results_df['correlation'].abs()
results_df.sort_values(by='abs_correlation', ascending=False, inplace=True)

# Select features with p-value <= 0.15
selected_features = results_df[results_df['p_value'] <= 0.15]['feature'].tolist()

# Don't forget to include the target column 'CVDINFR4'
selected_features.append('CVDINFR4')

# Filter df3 using selected features
df4 = df3[selected_features]

df4_results = verify_data_integrity(df4)


unique_values = df4['CVDINFR4'].unique()

# Print the unique values
print(unique_values)









# 05 - DMM
"""
Identify the DM method and describe how it aligns with the business objectives.
"""

# 06 - DMA
"""
Choose the relevant algorithm.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Set up feature matrix X and target vector y
X = df4.drop('CVDINFR4', axis=1)
y = df4['CVDINFR4']

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=10000)  # max_iter is set high to ensure convergence

# Record the start time
start_time = time.time()

# Train the model
logreg.fit(X, y)

# Calculate the time taken
time_taken = time.time() - start_time

print(f"Time taken to train the model: {time_taken:.2f} seconds")

# Predict on the entire dataset
y_pred = logreg.predict(X)

# Calculate and print the training accuracy
training_accuracy = accuracy_score(y, y_pred)
print(f"Training Accuracy: {training_accuracy * 100:.2f}%")



# Set up feature matrix X and target vector y
X = df4.drop('CVDINFR4', axis=1)
y = df4['CVDINFR4']

# Initialize the logistic regression model
logreg = LogisticRegression(
    penalty='l2',
    C=1.0, 
    solver='liblinear', 
    max_iter=1000,
    fit_intercept=True
)
# Record the start time
start_time = time.time()

# Train the model
logreg.fit(X, y)

# Calculate the time taken
time_taken = time.time() - start_time

print(f"Time taken to train the model: {time_taken:.2f} seconds")

# Predict on the entire dataset
y_pred = logreg.predict(X)

# Calculate and print the training accuracy
training_accuracy = accuracy_score(y, y_pred)
print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y, y_pred)

print("Confusion Matrix:")
print(cm)






from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()

# Record the start time
start_time = time.time()

# Train the model
tree_clf.fit(X, y)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the entire dataset
y_pred = tree_clf.predict(X)

# Calculate and print the training accuracy
training_accuracy = accuracy_score(y, y_pred)

print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
print(f"Time taken to train the model: {time_taken:.2f} seconds")



import numpy as np
from sklearn.model_selection import train_test_split

X = df4.drop('CVDINFR4', axis=1)
y = df4['CVDINFR4']

# Split the data into training and test sets (e.g., 80% / 20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the range of values
min_samples_splits = np.arange(2, 25, 2)  # Example range: 2, 4, ... 48
min_samples_leaves = np.arange(1, 25, 2)  # Example range: 1, 3, ... 49

train_results = []
test_results = []

for split in min_samples_splits:
    for leaf in min_samples_leaves:
        tree = DecisionTreeClassifier(min_samples_split=split, min_samples_leaf=leaf)
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_results.append(train_accuracy)
        
        test_pred = tree.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_results.append(test_accuracy)

# Now, let's plot
X, Y = np.meshgrid(min_samples_splits, min_samples_leaves)
Z_train = np.array(train_results).reshape(X.shape)
Z_test = np.array(test_results).reshape(X.shape)

plt.figure(figsize=(12, 6))
cp = plt.contourf(X, Y, Z_train, cmap='viridis')
plt.colorbar(cp, label="Training Accuracy")
plt.title('Training Accuracy for different min_samples_split and min_samples_leaf')
plt.xlabel('Min Samples Split')
plt.ylabel('Min Samples Leaf')
plt.show()

plt.figure(figsize=(12, 6))
cp = plt.contourf(X, Y, Z_test, cmap='viridis')
plt.colorbar(cp, label="Test Accuracy")
plt.title('Test Accuracy for different min_samples_split and min_samples_leaf')
plt.xlabel('Min Samples Split')
plt.ylabel('Min Samples Leaf')
plt.show()





# Assuming df4 is your dataset and 'CVDINFR4' is the target column
X = df4.drop('CVDINFR4', axis=1)
y = df4['CVDINFR4']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a range of depth values
max_depths = np.arange(1, 25)  # Example range: 1 to 49

train_accuracies = []
test_accuracies = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    
    train_pred = tree.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_accuracy)
    
    test_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_accuracies.append(test_accuracy)

# Now, let's plot
plt.figure(figsize=(12, 6))
plt.plot(max_depths, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(max_depths, test_accuracies, label='Test Accuracy', marker='o')
plt.title('Accuracy vs. Tree Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



tree_clf = DecisionTreeClassifier(criterion='gini',
    max_depth=4,
    min_samples_split=12,
    min_samples_leaf=15,)

# Record the start time
start_time = time.time()

# Train the model
tree_clf.fit(X, y)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the entire dataset
y_pred = tree_clf.predict(X)

# Calculate and print the training accuracy
training_accuracy = accuracy_score(y, y_pred)

print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
print(f"Time taken to train the model: {time_taken:.2f} seconds")



# from sklearn.svm import SVC

# svm_clf = SVC()

# # Record the start time
# start_time = time.time()

# # Train the model
# svm_clf.fit(X, y)

# # Calculate the time taken
# time_taken = time.time() - start_time

# # Predict on the entire dataset
# y_pred = svm_clf.predict(X)

# # Calculate and print the training accuracy
# training_accuracy = accuracy_score(y, y_pred)

# print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
# print(f"Time taken to train the model: {time_taken:.2f} seconds")


from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(hidden_layer_sizes=(100,))

# Record the start time
start_time = time.time()

# Train the model
nn_clf.fit(X, y)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the entire dataset
y_pred = nn_clf.predict(X)

# Calculate the training accuracy
training_accuracy = accuracy_score(y, y_pred)

print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
print(f"Time taken to train the model: {time_taken:.2f} seconds")



# nn_clf = MLPClassifier(hidden_layer_sizes=(100, 100),
#     activation='relu',
#     solver='adam',
#     batch_size=200,
#     learning_rate='adaptive',
#     max_iter=300,
#     )

# # Record the start time 
# start_time = time.time()

# # Train the model
# nn_clf.fit(X, y)

# # Calculate the time taken
# time_taken = time.time() - start_time

# # Predict on the entire dataset
# y_pred = nn_clf.predict(X)

# # Calculate the training accuracy
# training_accuracy = accuracy_score(y, y_pred)

# print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
# print(f"Time taken to train the model: {time_taken:.2f} seconds")

# # Calculate the confusion matrix
# cm = confusion_matrix(y, y_pred)

# print("Confusion Matrix:")
# print(cm)














# 07 - DM
"""
Execute the DM task.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

X = df4.drop('CVDINFR4', axis=1)
y = df4['CVDINFR4']

# Split the data into training and test sets (80% / 20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Initialize the logistic regression model
logreg = LogisticRegression(
    penalty='l2',
    C=1.0, 
    solver='liblinear', 
    max_iter=1000,
    fit_intercept=True
)

# Record the start time
start_time = time.time()

# Training the logistic regression model
logreg.fit(X_train, y_train)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(f"Time taken to train the model: {time_taken:.2f} seconds")
logreg_cr = classification_report(y_test, y_pred)
print(logreg_cr)
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()


# Extract coefficients
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Calculate odds ratios
odds_ratios = np.exp(coefficients)

# Create the dataframe
result_logreg = pd.DataFrame({
    'Feature': X_train.columns,
    'B (Coefficient)': coefficients,
    'Exp(B) (Odds Ratio)': odds_ratios
})


# Sort the dataframe by 'B (Coefficient)'
sorted_result = result_logreg.sort_values(by='B (Coefficient)', ascending=False)

# Extract top 6 and bottom 6 (to later exclude the very top and bottom)
top_6 = sorted_result.head(6)
bottom_6 = sorted_result.tail(6)

# Exclude the very top and bottom
top_5_except_1st = top_6.iloc[1:]
bottom_5_except_last = bottom_6.iloc[:-1]

# Plotting top 5
plt.figure(figsize=(10,6))
sns.barplot(x='B (Coefficient)', y='Feature', data=top_5_except_1st)
plt.title('Top 5 Features According to B')
plt.show()

# Plotting bottom 5
plt.figure(figsize=(10,6))
sns.barplot(x='B (Coefficient)', y='Feature', data=bottom_5_except_last)
plt.title('Bottom 5 Features According to B')
plt.show()














tree_clf = DecisionTreeClassifier(criterion='gini',
    max_depth=4,
    min_samples_split=12,
    min_samples_leaf=15,
    )

# Record the start time
start_time = time.time()

# Train the model
tree_clf.fit(X_train, y_train)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the test set
y_pred = tree_clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(f"Time taken to train the model: {time_taken:.2f} seconds")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 15))
plot_tree(tree_clf, filled=True, feature_names=X.columns, class_names=str(tree_clf.classes_), rounded=True, fontsize=10)
plt.show()







nn_clf = MLPClassifier(hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    batch_size=200,
    learning_rate='adaptive',
    max_iter=300,
    )


# Record the start time
start_time = time.time()

# Train the model
nn_clf.fit(X_train, y_train)

# Calculate the time taken
time_taken = time.time() - start_time

# Predict on the entire dataset
y_pred = nn_clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(f"Time taken to train the model: {time_taken:.2f} seconds")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()



print(classification_report(y_test, y_pred))


from sklearn.inspection import permutation_importance

# Compute the permutation importance
result = permutation_importance(nn_clf, X_train, y_train, n_repeats=30, random_state=42, n_jobs=-1)

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': result.importances_mean,
    'Standard Deviation': result.importances_std
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)


# Sort the dataframe by 'Importance'
sorted_importance = importance_df.sort_values(by='Importance', ascending=False)

top_10_except_1st = sorted_importance.head(11).iloc[1:]

# Plotting top 10 excluding the 1st
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_10_except_1st, ci="Standard Deviation")
plt.title('Top 10 Features According to Importance')
plt.show()




plt.figure(figsize=(12, 8))
plt.plot(nn_clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()





# 08 - INT
"""
Summarise your result.
"""




# 09 - ACT
"""
Describe the action plan to implement the results.
"""
