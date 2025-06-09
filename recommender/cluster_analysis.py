import pandas as pd
import joblib
import os

# Load the cleaned food data
food_df_cleaned = pd.read_csv("data/FoodData_Cleaned.csv")

# Load the trained scaler and KMeans model
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")

clustering_features = [
    'Calories', 'Carbohydrates', 'Protein', 'Fat',
    'Fiber Content', 'Glycemic Index'
]


x_scaled = scaler.transform(food_df_cleaned[clustering_features])

# Assign clusters to the cleaned data
food_df_cleaned['Cluster'] = kmeans.predict(x_scaled)

# --- Analyze Cluster Characteristics ---
print("\n--- Nutritional Averages for Each Cluster (StandardScaled Values) ---")
# The cluster centers are in the scaled space
cluster_centers_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=clustering_features)
print(cluster_centers_scaled)

print("\n--- Nutritional Averages for Each Cluster (Original MinMaxScaled Values) ---")
# To get averages in the original MinMaxScaled range (0-1),
# you can group the assigned dataframe and calculate means.
cluster_means_minmax = food_df_cleaned.groupby('Cluster')[clustering_features].mean()
print(cluster_means_minmax)

print("\n--- Food Type Distribution within Each Cluster ---")
cluster_food_type_distribution = food_df_cleaned.groupby('Cluster')['Food Type'].value_counts(normalize=True).unstack(
    fill_value=0)
print(cluster_food_type_distribution)

print("\n--- 'Suitable for Diabetes' Distribution within Each Cluster ---")
cluster_suitable_distribution = food_df_cleaned.groupby('Cluster')['Suitable for Diabetes'].value_counts(
    normalize=True).unstack(fill_value=0)
print(cluster_suitable_distribution)
