import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans  # Import KMeans
import matplotlib.pyplot as plt
import warnings

# Filter out the warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the dataset (assuming it's in CSV format)
data = pd.read_csv("COVID-19_Data.csv")

# Convert the date column to datetime format if it's not already
data['cdc_report_dt'] = pd.to_datetime(data['cdc_report_dt'])

# Filter the dataset for dates between January 1, 2020, and December 31, 2020
start_date = '2020-01-01'
end_date = '2020-12-31'
filtered_data = data[(data['cdc_report_dt'] >= start_date) & (data['cdc_report_dt'] <= end_date)]

# Select relevant features: race/ethnicity, age, sex, medical conditions, hospitalization
selected_features = ['race_ethnicity_combined', 'age_group', 'sex', 'medcond_yn', 'hosp_yn']
data = filtered_data[selected_features]

# Handle missing values
data.dropna(inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['race_ethnicity_combined', 'age_group', 'sex'])

# Encode hospitalization and medical condition as numerical (1 for 'Yes', 0 for 'No')
data['hosp_yn'] = data['hosp_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers
data['medcond_yn'] = data['medcond_yn'].map({'Yes': 1, 'No': 0})  # Encoding as integers

# Scale numerical variables
scaler = StandardScaler()
ipca = IncrementalPCA(n_components=2, batch_size=500)  # Adjust batch_size as needed

# Fit IncrementalPCA
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

# Get explained variance for PC1 and PC2
explained_variance_pc1 = ipca.explained_variance_ratio_[0]
explained_variance_pc2 = ipca.explained_variance_ratio_[1]

# Transform the entire dataset using fitted PCA
pca_data = ipca.transform(scaled_data)

# Initialize k-means with desired number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed

# Fit k-means to the data
kmeans.fit(scaled_data)

# Get cluster labels
cluster_labels = kmeans.labels_

# Create a scatter plot with legend
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.xlabel('PC1 (Explained Variance: {:.2f}%)'.format(explained_variance_pc1 * 100))
plt.ylabel('PC2 (Explained Variance: {:.2f}%)'.format(explained_variance_pc2 * 100))
plt.title('PCA Scatter Plot with K-Means Clustering')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
