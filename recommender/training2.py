import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

def find_optimal_clusters(X_scaled, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    # Create subplots for both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (WCSS)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette score
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Different k')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("data/cluster_optimization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Recommended optimal k based on silhouette score: {optimal_k}")
    print(f"Silhouette scores: {dict(zip(K_range, silhouette_scores))}")
    
    return optimal_k

def analyze_clusters(df, features):
    """Analyze cluster characteristics"""
    cluster_analysis = []
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        analysis = {
            'cluster': cluster,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100
        }
        
        # Calculate mean values for key nutritional features
        key_features = ['calories', 'protein_g', 'fat_g', 'carbohydrates_g', 'fiber_g']
        for feature in key_features:
            if feature in cluster_data.columns:
                analysis[f'avg_{feature}'] = cluster_data[feature].mean()
        
        cluster_analysis.append(analysis)
    
    cluster_df = pd.DataFrame(cluster_analysis)
    print("\nCluster Analysis:")
    print(cluster_df.round(3))
    
    # Save cluster analysis
    cluster_df.to_csv("data/cluster_analysis.csv", index=False)
    
    return cluster_df

def train_food_clustering_model():
    """Train KMeans clustering model for Kenyan food dataset"""
    
    # Load cleaned dataset
    print("Loading dataset...")
    df = pd.read_csv("data/KenyaFoodData_clean.csv")
    
    # Keep food information for reference
    food_info = df[['food_code', 'food_name']].copy()
    
    # Select numeric features for clustering
    features = df.drop(columns=['food_code', 'food_name'])
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    X = features[numeric_cols]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features for clustering: {len(numeric_cols)}")
    print(f"Features: {list(numeric_cols)}")
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k = find_optimal_clusters(X_scaled, max_k=10)
    
    # Allow manual override
    user_k = input(f"\nUse recommended k={optimal_k}? Press Enter or type different k: ").strip()
    if user_k.isdigit():
        optimal_k = int(user_k)
    
    print(f"Training KMeans with k={optimal_k}...")
    
    # Train the KMeans clustering model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate final silhouette score
    final_silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"Final silhouette score: {final_silhouette:.3f}")
    
    # Add cluster labels to dataset
    df['cluster'] = cluster_labels
    
    # Save models and data
    print("Saving models and data...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    
    # Save feature names for later use
    feature_names = list(numeric_cols)
    joblib.dump(feature_names, "models/feature_names.pkl")
    
    # Save clustered data
    df.to_csv("data/clustered_food_data.csv", index=False)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df, numeric_cols)
    
    # Visualizations
    print("Creating visualizations...")
    
    # 1. PCA Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    for i, cluster in enumerate(sorted(df['cluster'].unique())):
        cluster_data = df[df['cluster'] == cluster]
        cluster_pca = X_pca[df['cluster'] == cluster]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], 
                   c=[colors[i]], label=f'Cluster {cluster} (n={len(cluster_data)})', 
                   alpha=0.7, s=50)
    
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'Food Clusters Visualization (k={optimal_k})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/cluster_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Cluster Cardinality
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                   color=colors[:len(cluster_counts)], alpha=0.8)
    plt.title('Cluster Cardinality (Number of Foods per Cluster)')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Foods')
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/cluster_cardinality.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature Importance (based on cluster centers)
    feature_importance = np.std(kmeans.cluster_centers_, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top 15 Features by Clustering Importance')
    plt.xlabel('Standard Deviation of Cluster Centers')
    plt.tight_layout()
    plt.savefig("data/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print sample foods from each cluster
    print("\nSample foods from each cluster:")
    for cluster in sorted(df['cluster'].unique()):
        cluster_foods = df[df['cluster'] == cluster]['food_name'].head(5).tolist()
        print(f"Cluster {cluster}: {', '.join(cluster_foods)}")
    
    print(f"\nKenyan Food Dataset Clustering complete!")
    print(f"- Model saved to: models/kenya_food_kmeans.pkl")
    print(f"- Scaler saved to: models/kenya_food_scaler.pkl")
    print(f"- Feature names saved to: models/kenya_food_features.pkl")
    print(f"- Clustered data saved to: data/kenya_food_clustered.csv")
    print(f"- Visualizations saved to: data/ (with kenya_food_ prefix)")
    print(f"- Final silhouette score: {final_silhouette:.3f}")
    print(f"\nYour original models remain untouched!")

if __name__ == "__main__":
    train_food_clustering_model()