# kmeans_visualizer_app.py

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Load pretrained model
with open('/mnt/data/kmeans_customer_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Get number of clusters from the model
num_clusters = loaded_model.n_clusters

# Set Streamlit page config
st.set_page_config(page_title="K-Means Clustering Viewer", layout="centered")

# Page title
st.title("📊 K-Means Clustering Visualizer")

# Sidebar
st.sidebar.header("🔧 Options")
selected_cluster = st.sidebar.slider("Highlight cluster", min_value=0, max_value=num_clusters - 1, value=0)

# Generate synthetic dataset (optional - if original X not saved in model)
X, _ = make_blobs(n_samples=300, centers=num_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Plot
fig, ax = plt.subplots()
colors = np.array(['gray'] * len(X))
colors[y_kmeans == selected_cluster] = 'orange'

# Plot data points
ax.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7, label='Data Points')
# Plot cluster centers
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')

ax.set_title(f"Cluster View: Highlighting Cluster {selected_cluster}")
ax.legend()
st.pyplot(fig)
