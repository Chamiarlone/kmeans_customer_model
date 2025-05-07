import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Load model
with open('kmeans_customer_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Infer required feature count from cluster centers
num_clusters = loaded_model.n_clusters
n_features = loaded_model.cluster_centers_.shape[1]

# Set page config
st.set_page_config(page_title="K-Means Clustering Viewer", layout="centered")
st.title("📊 K-Means Clustering Visualizer")

# Sidebar for highlighting a cluster
st.sidebar.header("🔧 Options")
selected_cluster = st.sidebar.slider("Highlight cluster", min_value=0, max_value=num_clusters - 1, value=0)

# Generate synthetic data with correct number of features
X, _ = make_blobs(n_samples=300, centers=num_clusters, n_features=n_features, cluster_std=0.60, random_state=0)

# Predict
y_kmeans = loaded_model.predict(X)

# Plot
fig, ax = plt.subplots()
colors = np.array(['gray'] * len(X))
colors[y_kmeans == selected_cluster] = 'orange'

# Only plot first 2 dimensions for visualization
ax.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7, label='Data Points')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')

ax.set_title(f"Cluster View: Highlighting Cluster {selected_cluster}")
ax.legend()
st.pyplot(fig)
