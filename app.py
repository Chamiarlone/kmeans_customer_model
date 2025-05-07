import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Load model
with open('kmeans_customer_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Get number of clusters and features from the model
num_clusters = loaded_model.n_clusters
n_features = loaded_model.cluster_centers_.shape[1]

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering Viewer", layout="centered")
st.title("📊 K-Means Clustering Visualizer")

# Sidebar options
st.sidebar.header("🔧 Options")

# Modify slider to have more adjustable levels (with a step value)
selected_cluster = st.sidebar.slider(
    "Highlight Cluster", 
    min_value=0, 
    max_value=num_clusters - 1, 
    value=0, 
    step=1  # You can adjust the step size if you want a finer or coarser control
)

# Generate synthetic data with correct number of features
X, _ = make_blobs(n_samples=300, centers=num_clusters, n_features=n_features, cluster_std=0.60, random_state=0)

# Predict clusters
y_kmeans = loaded_model.predict(X)

# Build color list for highlighting selected cluster
colors = ['orange' if cluster == selected_cluster else 'gray' for cluster in y_kmeans]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot all points, with larger size and transparency for the highlighted cluster
scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, label='Data Points')

# Highlight the selected cluster points with a different style (e.g., bigger points, border)
highlighted_points = np.where(y_kmeans == selected_cluster)[0]
ax.scatter(X[highlighted_points, 0], X[highlighted_points, 1], c='orange', s=200, edgecolors='black', label=f'Cluster {selected_cluster}')

# Plot centroids
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1],
           s=300, c='red', marker='X', label='Centroids')

# Set labels and title
ax.set_xlabel("Age")
ax.set_ylabel("Spending Score")
ax.set_title(f"Cluster View: Highlighting Cluster {selected_cluster}")
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)

# Optional note if the model has more than 2 features
if n_features > 2:
    st.warning(f"Note: Your model was trained on {n_features} features. Only the first 2 features are visualized here.")
