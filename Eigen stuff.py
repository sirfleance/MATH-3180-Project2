import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (assuming it's in CSV format)
data = pd.read_csv("COVID-19_Data.csv")

# Select relevant features: race/ethnicity, age, sex, medical conditions, hospitalization
selected_features = ['race_ethnicity_combined', 'age_group', 'sex', 'medcond_yn', 'hosp_yn']
data = data[selected_features]

# Handle missing values
data.dropna(inplace=True)

# Reduce the size of the dataset (e.g., taking a random sample)
sample_size = 1000  # Adjust the sample size as needed
data = data.sample(n=sample_size, random_state=42)  # Random sample with a fixed random state for reproducibility

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['race_ethnicity_combined', 'age_group', 'sex'])

# Encode hospitalization and medical condition as numerical (1 for 'Yes', 0 for 'No')
data['hosp_yn'] = data['hosp_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers
data['medcond_yn'] = data['medcond_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

# Plot eigenvalues
plt.figure(figsize=(8, 6))
plt.scatter(range(len(eigenvalues)), eigenvalues, color='skyblue')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Correlation Matrix')
plt.grid(True)
plt.show()

