import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
#Incremental PCA allows for data to be analyzed and aggregated in chunks, imperferct but a must for a large dataset

import matplotlib.pyplot as plt
import warnings

# Helps me to analyze large amounts of data (8.3million cases)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#The Data Sorted out using SQL
data = pd.read_csv("COVID-19_Data.csv")

# Rewriting dates to a reradable format
data['cdc_report_dt'] = pd.to_datetime(data['cdc_report_dt'])

# Filter the dataset for dates between January 1, 2020, and December 31, 2020 for an initial study
start_date = '2020-01-01'
end_date = '2020-12-31'
filtered_data = data[(data['cdc_report_dt'] >= start_date) & (data['cdc_report_dt'] <= end_date)]

# Select relevant features: race/ethnicity, age, sex, medical conditions, hospitalization
selected_features = ['race_ethnicity_combined', 'age_group', 'sex', 'medcond_yn', 'hosp_yn']
data = filtered_data[selected_features]

# Handle missing values (Basically get them out hehe)
data.dropna(inplace=True)

# One-hot encode categorical variables for their usability (is that a word?) Strings -> Floats
data = pd.get_dummies(data, columns=['race_ethnicity_combined', 'age_group', 'sex'])

# Encode hospitalization and medical condition as numerical (1 for 'Yes', 0 for 'No')
data['hosp_yn'] = data['hosp_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers
data['medcond_yn'] = data['medcond_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers

# Scale numerical variables
scaler = StandardScaler()
ipca = IncrementalPCA(n_components=2, batch_size=500)  # Adjust batch_size as needed

# Iterate over chunks of data and fit PCA
iteration_count = 0
for chunk in pd.read_csv("COVID-19_Data.csv", chunksize=5000):
    # Filter chunk for the selected features
    chunk = chunk[selected_features]
    # Handle missing values
    chunk.dropna(inplace=True)
    # One-hot encode categorical variables
    chunk = pd.get_dummies(chunk, columns=['race_ethnicity_combined', 'age_group', 'sex'])
    # Encode 'hosp_yn' and 'medcond_yn' columns as integers
    chunk['hosp_yn'] = chunk['hosp_yn'].map({'Yes': 1, 'No': 0})
    chunk['medcond_yn'] = chunk['medcond_yn'].map({'Yes': 1, 'No': 0})
    # Align columns with the original dataset
    aligned_chunk = chunk.reindex(columns=data.columns, fill_value=0)
    # Print summary statistics of aligned_chunk
    print("Summary statistics of aligned_chunk:")
    print(aligned_chunk.describe())
    # Scale numerical variables
    scaled_data = scaler.fit_transform(aligned_chunk.drop(['hosp_yn'], axis=1))
    # Check for NaNs and infinite values
    if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        print("Warning: NaNs or infinite values detected. Skipping partial fit for this chunk.")
        continue
    # Fit IncrementalPCA
    ipca.partial_fit(scaled_data)
    # Increment the iteration counter
    iteration_count += 1
    # Check if reached the desired number of iterations
    if iteration_count >= 50:
        break

# Transform the entire dataset using fitted PCA
pca_data = ipca.transform(scaled_data)

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(len(ipca.explained_variance_ratio_)), ipca.explained_variance_ratio_, color='skyblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.show()

# Create a scatter plot with legend
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=aligned_chunk['hosp_yn'], cmap='coolwarm', alpha=0.5, label=['No Hospitalization', 'Hospitalization'])  # Using the last chunk for coloring
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Scatter Plot')
plt.colorbar(label='Hospitalization')
plt.grid(True)
plt.legend()
plt.show()

