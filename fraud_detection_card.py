
import numpy as np 
import pandas as pd
import joblib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import re
import os
import unicodedata
import itertools

# Library for file manipulation
import pandas as pd
import numpy as np
import pandas

# Data visualization
import plotly
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import pyplot as plt
from IPython.display import SVG

# Configuration for graphs width and layout of graphs
sns.set_theme(style='whitegrid')
palette='viridis'

# Warnings remove warnings
import warnings
warnings.filterwarnings("ignore")

# Python version
from platform import python_version
print('Python version in this Jupyter Notebook:', python_version())

# Load library versions
import watermark

# Database 
data = pd.read_csv("creditcard.csv")
data

# Data info
data.info()

# Function to convert the binary variable "Class" to string
def fraude(dado):
    if dado == 0:
        return "Normal"
    else:
        return "Fraude"

# Copy the data
dados_tmp = data.copy()
dados_tmp['Class'] = dados_tmp['Class'].apply(fraude)

# Figure size
plt.figure(figsize=(8, 6))

# Bar plot with counts on bars and different colors
ax = sns.countplot(data=dados_tmp, x='Class', palette='Blues')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')

# Adding counts on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.grid(False)
plt.show()

del dados_tmp

# Import necessary modules from sklearn for scaling
from sklearn.preprocessing import StandardScaler, RobustScaler

# Initialize the RobustScaler
robust_scaler = RobustScaler()

# This transformation helps in reducing the impact of outliers by scaling the data based on the interquartile range (IQR)
data['Amount'] = robust_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Similarly, the 'Time' data is scaled to manage potential outliers and make the data more suitable for modeling
data['Time'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Display the first few rows of the transformed dataset
data.head()

# Separate the dataset into features (X) and target variable (Y)
X = data.drop(['Class'], axis=1)  
Y = data['Class']  

# Import necessary modules for cross-validation
from sklearn.model_selection import KFold, StratifiedKFold

# StratifiedKFold ensures that each fold of the cross-validation process has the same proportion of each class
strat_kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# Split the data using StratifiedKFold and iterate through each fold
for indice_treino, indice_teste in strat_kfold.split(X, Y):
    print("Train Indices:", indice_treino, "Test Indices:", indice_teste)
    X_treino, X_teste = X.iloc[indice_treino], X.iloc[indice_teste]  
    Y_treino, Y_teste = Y.iloc[indice_treino], Y.iloc[indice_teste]  

# Convert the training and testing datasets to NumPy arrays
X_treino = X_treino.values
X_teste = X_teste.values
Y_treino = Y_treino.values
Y_teste = Y_teste.values

# Calculate the unique class labels and their respective counts in the training and testing sets
label_treino_unico, label_contagem_treino = np.unique(Y_treino, return_counts=True)
label_teste_unico, label_contagem_teste = np.unique(Y_teste, return_counts=True)

# Print the distribution of classes in the training and testing datasets
print('-' * 100)
print('Class Distribution: \n')
print('-' * 100)
print(label_contagem_treino / len(Y_treino))  # Proportion of each class in the training set
print('-' * 100)
print(label_contagem_teste / len(Y_teste))  # Proportion of each class in the testing set

# Randomly shuffle the entire dataset
dados = data.sample(frac=1)

# Determine the number of fraudulent transactions
tamanho_fraude = len(dados[dados['Class'] == 1])

# Extract all rows corresponding to fraudulent transactions
dados_fraude = dados.loc[dados['Class'] == 1]

# This is done to balance the dataset by randomly selecting the same number of non-fraudulent transactions
dados_semfraude = dados.loc[dados['Class'] == 0][:tamanho_fraude]

# Combine the fraudulent and the randomly selected non-fraudulent transactions to form a new balanced dataset
dados_novos = pd.concat([dados_fraude, dados_semfraude])

# Shuffle the new balanced dataset to ensure random order
dados_novos = dados_novos.sample(frac=1, random_state=42)

# Display the first few rows of the new balanced dataset
dados_novos.head()

# Copy the balanced dataset and convert the 'Class' values to string labels
dados_tmp = dados_novos.copy()
dados_tmp['Class'] = dados_tmp['Class'].apply(fraude)

# Set the figure size
plt.figure(figsize=(8, 6))

# Create a bar plot with customized colors
ax = sns.countplot(data=dados_tmp, x='Class', palette='coolwarm')

# Add labels and title
plt.xlabel('Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Normal and Fraudulent Transactions', fontsize=16)

# Add the count numbers on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=12)

# Customize grid and background
plt.grid(False)

# Show the plot
plt.show()

# Clean up the temporary dataframe
del dados_tmp

# Calculate the correlation matrix for both the unbalanced and balanced datasets
correlacao_desbalanceada = dados.corr() 
correlacao_balanceada = dados_novos.corr()  

# Set up the figure with two subplots
plt.figure(figsize=(20, 10))

# Plot the correlation matrix for the unbalanced dataset without annotations
plt.subplot(1, 2, 1)
sns.heatmap(correlacao_desbalanceada, annot=False, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix - Unbalanced Data", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Plot the correlation matrix for the balanced dataset without annotations
plt.subplot(1, 2, 2)
sns.heatmap(correlacao_balanceada, annot=False, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix - Balanced Data", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

f, ax = plt.subplots(2,2, figsize=(20,8))

sns.boxplot(x="Class", y="V3", data=dados_novos,ax=ax[0][0])
sns.boxplot(x="Class", y="V10", data=dados_novos,ax=ax[0][1])
sns.boxplot(x="Class", y="V12", data=dados_novos,ax=ax[1][0])
sns.boxplot(x="Class", y="V14", data=dados_novos,ax=ax[1][1])

f, ax = plt.subplots(1,3, figsize=(20,4))

sns.boxplot(x="Class", y="V2", data=dados_novos,ax=ax[0])
sns.boxplot(x="Class", y="V4", data=dados_novos,ax=ax[1])
sns.boxplot(x="Class", y="V11", data=dados_novos,ax=ax[2])

from scipy.stats import norm

# Create a figure with 4x2 subplots
f, ax = plt.subplots(4, 2, figsize=(20, 24))

# Plot distributions for V3
sns.distplot(dados_novos['V3'].loc[dados_novos['Class'] == 0].values, ax=ax[0][0], fit=norm)
ax[0][0].set_title('Distribuição de V3 \n (Transações Normais)', fontsize=14)
sns.distplot(dados_novos['V3'].loc[dados_novos['Class'] == 1].values, ax=ax[0][1], fit=norm)
ax[0][1].set_title('Distribuição de V3 \n (Transações com Fraude)', fontsize=14)

# Plot distributions for V10
sns.distplot(dados_novos['V10'].loc[dados_novos['Class'] == 0].values, ax=ax[1][0], fit=norm)
ax[1][0].set_title('Distribuição de V10 \n (Transações Normais)', fontsize=14)
sns.distplot(dados_novos['V10'].loc[dados_novos['Class'] == 1].values, ax=ax[1][1], fit=norm)
ax[1][1].set_title('Distribuição de V10 \n (Transações com Fraude)', fontsize=14)

# Plot distributions for V12
sns.distplot(dados_novos['V12'].loc[dados_novos['Class'] == 0].values, ax=ax[2][0], fit=norm)
ax[2][0].set_title('Distribuição de V12 \n (Transações Normais)', fontsize=14)
sns.distplot(dados_novos['V12'].loc[dados_novos['Class'] == 1].values, ax=ax[2][1], fit=norm)
ax[2][1].set_title('Distribuição de V12 \n (Transações com Fraude)', fontsize=14)

# Plot distributions for V14
sns.distplot(dados_novos['V14'].loc[dados_novos['Class'] == 0].values, ax=ax[3][0], fit=norm)
ax[3][0].set_title('Distribuição de V14 \n (Transações Normais)', fontsize=14)
sns.distplot(dados_novos['V14'].loc[dados_novos['Class'] == 1].values, ax=ax[3][1], fit=norm)
ax[3][1].set_title('Distribuição de V14 \n (Transações com Fraude)', fontsize=14)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Extract the values of V3 for fraudulent transactions
v3_fraude = dados_novos['V3'].loc[dados_novos['Class'] == 1].values

# Calculate the 25th and 75th percentiles (Q1 and Q3) for V3
q25, q75 = np.percentile(v3_fraude, 25), np.percentile(v3_fraude, 75)

# Compute the Interquartile Range (IQR) for V3
v3_iqr = q75 - q25

# Define the cutoff for identifying outliers, using 1.5 times the IQR
v3_cut_off = v3_iqr * 1.5
v3_inferior, v3_superior = q25 - v3_cut_off, q75 + v3_cut_off

# Identify the outliers in V3 based on the calculated cutoff
outliers = [x for x in v3_fraude if x < v3_inferior or x > v3_superior]

# Remove the outliers from the dataset for V3
dados_novos = dados_novos.drop(dados_novos[(dados_novos['V3'] > v3_superior) | (dados_novos['V3'] < v3_inferior)].index)

# 2. Process for Variable V10 (same steps as above)
v10_fraude = dados_novos['V10'].loc[dados_novos['Class'] == 1].values
q25, q75 = np.percentile(v10_fraude, 25), np.percentile(v10_fraude, 75)
v10_iqr = q75 - q25
v10_cut_off = v10_iqr * 1.5
v10_inferior, v10_superior = q25 - v10_cut_off, q75 + v10_cut_off
outliers = [x for x in v10_fraude if x < v10_inferior or x > v10_superior]
dados_novos = dados_novos.drop(dados_novos[(dados_novos['V10'] > v10_superior) | (dados_novos['V10'] < v10_inferior)].index)

# 3. Process for Variable V12 (same steps as above)
v12_fraude = dados_novos['V12'].loc[dados_novos['Class'] == 1].values
q25, q75 = np.percentile(v12_fraude, 25), np.percentile(v12_fraude, 75)
v12_iqr = q75 - q25
v12_cut_off = v12_iqr * 1.5
v12_inferior, v12_superior = q25 - v12_cut_off, q75 + v12_cut_off
outliers = [x for x in v12_fraude if x < v12_inferior or x > v12_superior]
dados_novos = dados_novos.drop(dados_novos[(dados_novos['V12'] > v12_superior) | (dados_novos['V12'] < v12_inferior)].index)

# 4. Process for Variable V14 (same steps as above)
v14_fraude = dados_novos['V14'].loc[dados_novos['Class'] == 1].values
q25, q75 = np.percentile(v14_fraude, 25), np.percentile(v14_fraude, 75)
v14_iqr = q75 - q25
v14_cut_off = v14_iqr * 1.5
v14_inferior, v14_superior = q25 - v14_cut_off, q75 + v14_cut_off
outliers = [x for x in v14_fraude if x < v14_inferior or x > v14_superior]
dados_novos = dados_novos.drop(dados_novos[(dados_novos['V14'] > v14_superior) | (dados_novos['V14'] < v14_inferior)].index)

# Display the first few rows of the updated dataset
dados_novos.head()

f, ax = plt.subplots(2, 2, figsize=(20, 12))

# Box plot for V3
sns.boxplot(x="Class", y="V3", data=dados_novos, ax=ax[0][0], palette="Set3")
ax[0][0].set_title("V3 with Outliers Removed", fontsize=16)
ax[0][0].set_xlabel("Class", fontsize=14)
ax[0][0].set_ylabel("V3", fontsize=14)
ax[0][0].tick_params(axis='both', which='major', labelsize=12)
ax[0][0].grid(False)

# Box plot for V10
sns.boxplot(x="Class", y="V10", data=dados_novos, ax=ax[0][1], palette="Set3")
ax[0][1].set_title("V10 with Outliers Removed", fontsize=16)
ax[0][1].set_xlabel("Class", fontsize=14)
ax[0][1].set_ylabel("V10", fontsize=14)
ax[0][1].tick_params(axis='both', which='major', labelsize=12)
ax[0][1].grid(False)

# Box plot for V12
sns.boxplot(x="Class", y="V12", data=dados_novos, ax=ax[1][0], palette="Set3")
ax[1][0].set_title("V12 with Outliers Removed", fontsize=16)
ax[1][0].set_xlabel("Class", fontsize=14)
ax[1][0].set_ylabel("V12", fontsize=14)
ax[1][0].tick_params(axis='both', which='major', labelsize=12)
ax[1][0].grid(False)

# Box plot for V14
sns.boxplot(x="Class", y="V14", data=dados_novos, ax=ax[1][1], palette="Set3")
ax[1][1].set_title("V14 with Outliers Removed", fontsize=16)
ax[1][1].set_xlabel("Class", fontsize=14)
ax[1][1].set_ylabel("V14", fontsize=14)
ax[1][1].tick_params(axis='both', which='major', labelsize=12)
ax[1][1].grid(False)

# Adjust layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()

# The remaining columns in the dataset are the features used for prediction
X = dados_novos.drop('Class', axis=1).values

# Extract the target variable (Y) which represents the class labels
Y = dados_novos['Class'].values

# Import necessary libraries for dimensionality reduction and visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

X_reduzido_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)

# PCA is a linear dimensionality reduction technique that identifies the axes of maximum variance in the data
X_reduzido_pca = PCA(n_components=2, random_state=42).fit_transform(X)

# Dimensionality Reduction with Truncated SVD
X_reduzido_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X)

# Custom color patches for legend
azul = mpatches.Patch(color='#0A0AFF', label='Sem fraude')
vermelho = mpatches.Patch(color='#AF0000', label='Fraude')

# Create subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

# Plot for t-SNE
ax1.scatter(X_reduzido_tsne[Y == 0, 0], X_reduzido_tsne[Y == 0, 1], c='#0A0AFF', label='Sem fraude', alpha=0.6, linewidths=2)
ax1.scatter(X_reduzido_tsne[Y == 1, 0], X_reduzido_tsne[Y == 1, 1], c='#AF0000', label='Fraude', alpha=0.6, linewidths=2)
ax1.set_title('t-SNE', fontsize=18)
ax1.set_xlabel('Componente 1', fontsize=14)
ax1.set_ylabel('Componente 2', fontsize=14)
ax1.grid(False)
ax1.legend(handles=[azul, vermelho], fontsize=12)

# Plot for PCA
ax2.scatter(X_reduzido_pca[Y == 0, 0], X_reduzido_pca[Y == 0, 1], c='#0A0AFF', label='Sem fraude', alpha=0.6, linewidths=2)
ax2.scatter(X_reduzido_pca[Y == 1, 0], X_reduzido_pca[Y == 1, 1], c='#AF0000', label='Fraude', alpha=0.6, linewidths=2)
ax2.set_title('PCA', fontsize=18)
ax2.set_xlabel('Componente 1', fontsize=14)
ax2.set_ylabel('Componente 2', fontsize=14)
ax2.grid(False)
ax2.legend(handles=[azul, vermelho], fontsize=12)

# Plot for SVD
ax3.scatter(X_reduzido_svd[Y == 0, 0], X_reduzido_svd[Y == 0, 1], c='#0A0AFF', label='Sem fraude', alpha=0.6, linewidths=2)
ax3.scatter(X_reduzido_svd[Y == 1, 0], X_reduzido_svd[Y == 1, 1], c='#AF0000', label='Fraude', alpha=0.6, linewidths=2)
ax3.set_title('SVD', fontsize=18)
ax3.set_xlabel('Componente 1', fontsize=14)
ax3.set_ylabel('Componente 2', fontsize=14)
ax3.grid(False)
ax3.legend(handles=[azul, vermelho], fontsize=12)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X,
                                                        Y,
                                                        test_size=0.2,
                                                        random_state=42)
                                                        
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Model Definitions
modelos = {
           "Random Forest Classifier": RandomForestClassifier()
           }

# Cross-Validation for All Models
for nome, modelo in modelos.items():
    modelo.fit(X_treino, Y_treino)
    score_treino = cross_val_score(modelo, X_treino, Y_treino, cv=5)
    print("Model: ", nome, "has accuracy", round(score_treino.mean(), 2) * 100, "%")
    print('-' * 100)

# Hyperparameter tuning for Random Forest Classifier
rfc_params = {"criterion": ["gini", "entropy"], 
              "max_depth": list(range(2, 10, 1)), 
              "min_samples_leaf": list(range(5, 10, 1))}
grid_rfc = GridSearchCV(RandomForestClassifier(), rfc_params, n_jobs=-1)
grid_rfc.fit(X_treino, Y_treino)
rfc_clf = grid_rfc.best_estimator_
rfc_score = cross_val_score(rfc_clf, X_treino, Y_treino, cv=5)


feature_names = [f'feature_{i+1}' for i in range(30)]  
def plot_feature_importance(model, model_name, feature_names):
    feature_importances = model.feature_importances_
    
    if len(feature_importances) != len(feature_names):
        print(f"Error: Mismatch between feature importances and feature names lengths")
        print(f"Feature Importances Length: {len(feature_importances)}")
        print(f"Feature Names Length: {len(feature_names)}")
        return

    indices = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance - {model_name}")
    plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
    plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.show()

# Plotting feature importance for RandomForestClassifier
plot_feature_importance(rfc_clf, "Random Forest", feature_names)

X_subamostra = dados.drop('Class', axis=1)  # All features except the target variable 'Class'
Y_subamostra = dados['Class']  # The target variable 'Class'

# Using StratifiedKFold for cross-validation to ensure that each fold has the same proportion of each class
for indice_treino, indice_teste in strat_kfold.split(X_subamostra, Y_subamostra):
    # Printing the indices of the training and testing sets for each fold
    print("Training indices:", indice_treino, "Testing indices:", indice_teste)
    
    # Creating the training and testing subsets based on the indices generated by StratifiedKFold
    X_treino_subamostra, X_teste_subamostra = X_subamostra.iloc[indice_treino], X_subamostra.iloc[indice_teste]
    Y_treino_subamostra, Y_teste_subamostra = Y_subamostra.iloc[indice_treino], Y_subamostra.iloc[indice_teste]
    
# Converting the training and testing sets to numpy arrays for further processing
X_treino_subamostra = X_treino_subamostra.values
X_teste_subamostra = X_teste_subamostra.values
Y_treino_subamostra = Y_treino_subamostra.values
Y_teste_subamostra = Y_teste_subamostra.values

# Import necessary libraries for handling imbalanced data and evaluating model performance
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from imblearn.under_sampling import NearMiss
from collections import Counter

# Create an instance of the NearMiss under-sampling technique
nm = NearMiss()

# Apply the NearMiss technique to balance the dataset by under-sampling the majority class
X_nearmiss, Y_nearmiss = nm.fit_resample(X_subamostra.values, Y_subamostra.values)

# Count the number of occurrences of each class after applying NearMiss to verify the balance
contagem_nearmiss = Counter(Y_nearmiss)
print('NearMiss Count: {}'.format(contagem_nearmiss))

# Visualizing the results (this step is simply to display the NearMiss object, no further action taken)
nm

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Initialize lists to store metrics for subsampled data
accuracy_subsample = []
precision_subsample = []
recall_subsample = []
f1_subsample = []
auc_subsample = []

# Reducing the number of splits for faster computation
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

# Generate predictions using cross-validation for calculating AUC and ROC
previsao_rfc = cross_val_predict(rfc_clf, X_treino, Y_treino, cv=5)

print('Random Forest Classifier: ', roc_auc_score(Y_treino, previsao_rfc))

rfc_fpr, rfc_tpr, rfc_threshold = roc_curve(Y_treino, previsao_rfc)

# Function to plot multiple ROC curves
def graph_roc_curve_multiple(rfc_fpr, rfc_tpr):
    plt.figure(figsize=(18,10))
    plt.title('ROC Curve \n All Classifiers', fontsize=18)
    plt.plot(rfc_fpr, rfc_tpr, label='RFC Score: {:.4f}'.format(roc_auc_score(Y_treino, previsao_rfc)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(False)
    plt.legend()

# Calling the function to plot the ROC curves of all models
graph_roc_curve_multiple(rfc_fpr, rfc_tpr)

from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score, accuracy_score

# Calculate precision-recall curves for all models
precision_rfc, recall_rfc, threshold_rfc = precision_recall_curve(Y_treino, previsao_rfc)

# Predict the target variable for each model using the training data
Y_pred_rfc = rfc_clf.predict(X_treino)

# Function to print performance metrics for each model
def print_scores(model_name, Y_true, Y_pred):
    print(f'{model_name}:')
    print('Performance on Training Data: \n')
    print('Recall Score: {:.2f}'.format(recall_score(Y_true, Y_pred)))
    print('Precision Score: {:.2f}'.format(precision_score(Y_true, Y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(Y_true, Y_pred)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(Y_true, Y_pred)))
    print()

# Print performance metrics for all models
print_scores('Random Forest', Y_treino, Y_pred_rfc)

# Setting the limits for the y-axis and x-axis
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

from imblearn.over_sampling import SMOTE

# Create an instance of the SMOTE class
sm = SMOTE(sampling_strategy='minority', random_state=42)

# Apply the fit_resample method to balance the training data
Xsmote_treino, Ysmote_treino = sm.fit_resample(X_treino, Y_treino)

from sklearn.metrics import confusion_matrix

# Predictions for the models on the test set
Y_pred_rfc = rfc_clf.predict(X_teste)

# Calculating the confusion matrices
cm_rfc = confusion_matrix(Y_teste, Y_pred_rfc)

# Plotting the confusion matrices
fig, ax = plt.subplots(5, 2, figsize=(22, 30))

sns.heatmap(cm_rfc, ax=ax[2, 0], annot=True, cmap="Paired")
ax[2, 0].set_title("Random Forest \n Confusion Matrix", fontsize=14)

plt.show()

from sklearn.metrics import classification_report

# Print classification report for Random Forest model
print('Random Forest:')
print(classification_report(Y_teste, Y_pred_rfc))

from sklearn.metrics import accuracy_score, confusion_matrix

# Predictions for the models on the test set
Y_pred_rfc = rfc_clf.predict(X_teste)

# Scores for undersampling (model trained without SMOTE)
score_subamostra_rfc = accuracy_score(Y_teste, Y_pred_rfc)

# Plotting confusion matrices and comparing scores
fig, ax = plt.subplots(5, 2, figsize=(22, 30))

sns.heatmap(confusion_matrix(Y_teste, Y_pred_rfc), ax=ax[2, 0], annot=True, cmap="Paired")
ax[2, 0].set_title(f"Random Forest \n Confusion Matrix\n Undersampling: {score_subamostra_rfc:.2f}, Oversampling: {score_subamostra_rfc:.2f}", fontsize=14)
ax[2, 0].set_xticklabels(["No Fraud", "Fraud"], fontsize=12, rotation=0)
ax[2, 0].set_yticklabels(["No Fraud", "Fraud"], fontsize=12, rotation=0)

plt.show()

# Dados que geram o DataFrame final
data = {
    'Model': ['Random Forest'] * 2,
    'Technique': ['Random Undersampling'] * 10 + ['Oversampling (SMOTE)'] * 10,
    'Score': [score_subamostra_rfc]
}

# Certificando que todas as listas têm o mesmo comprimento
max_len = max(len(value) for value in data.values())
for key, value in data.items():
    if len(value) < max_len:
        data[key].extend([None] * (max_len - len(value)))

# Criando o DataFrame após o ajuste
final_df = pd.DataFrame(data)


# Function to highlight the highest and lowest scores
def highlight_max_min(val):
    color = ''
    if val == final_df['Score'].max():
        color = 'background-color: lightgreen'
    elif val == final_df['Score'].min():
        color = 'background-color: lightcoral'
    return color

# Applying the style to highlight the highest and lowest scores
styled_df = final_df.style.applymap(highlight_max_min, subset=['Score'])

# Displaying the styled DataFrame
styled_df

# Saave trained model
joblib.dump(modelo, 'rf_model.pkl')