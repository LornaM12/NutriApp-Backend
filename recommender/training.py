import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def train_food_clustering_model():
    df = pd.read_csv("data/FoodData_Cleaned.csv")

    # Features for clustering
    features = [
        'Calories', 'Carbohydrates', 'Protein', 'Fat',
        'Fiber Content', 'Glycemic Index'
    ]
    x = df[features]

    # Scale the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train the KMeans clustering model
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(x_scaled)

    # Save the model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    df['Cluster'] = kmeans.labels_

    # Food Clusters Visualization
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    df['PCA1'] = x_pca[:, 0]
    df['PCA2'] = x_pca[:, 1]

    plt.figure(figsize=(10, 6))
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', alpha=0.6)

    plt.title("Food Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/Cluster_Visualization.png")

    print("Clustering model and scaler saved successfully.")

    # Cluster Cardinality (number of foods per cluster)
    cluster_counts = df['Cluster'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title("Cluster Cardinality (Number of Foods per Cluster)")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Foods")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/Cluster_Cardinality.png")


if __name__ == "__main__":
    train_food_clustering_model()
