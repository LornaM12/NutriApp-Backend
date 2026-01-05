import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# --- CONFIGURATION ---
DATA_PATH = "data/KenyaFoodCompositionsClean.csv"
MODELS_DIR = "models/"
NUM_CLUSTERS = 5 # Group foods into 5 distinct "Metabolic Types"

def train_model():
    print("--- Step 2: Training the AI Model ---")
    
    # 1. Load Clean Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Could not find {DATA_PATH}. Did you run preprocess.py?")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} foods for training.")

    # 2. Select Features for AI Analysis
    # We include the new SCORES so the AI understands "Health Value" + "Hydration"
    features = [
        'calories', 
        'protein_g', 
        'fat_g', 
        'carbohydrates_g', 
        'fiber_g', 
        'diabetes_score',   # The Sugar Control metric
        'hydration_score'   # The Water/Fluid metric
    ]
    
    # Ensure only numeric data goes into the AI
    X = df[features].fillna(0)
    
    if X.empty:
        print("❌ Error: No valid data found for training.")
        return

    # 3. Scale the Data (0-1 Range)
    # AI works best when Calories (300) and Fiber (5) are on the same scale
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train K-Means
    print(f"Training K-Means with {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 5. Analyze Results (For your info)
    df['Cluster'] = kmeans.labels_
    
    print("\n--- AI Cluster Insights ---")
    print("The AI has grouped your foods into these patterns:")
    summary = df.groupby('Cluster')[['carbohydrates_g', 'protein_g', 'hydration_score']].mean().round(1)
    print(summary)
    
    # 6. Save the "Brain"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    
    # Update the CSV with Cluster IDs (Optional, but good for debugging)
    df.to_csv(DATA_PATH, index=False)
    
    print(f"\n✅ Success! Model saved to '{MODELS_DIR}'.")
    print(f"✅ Data updated with Cluster labels.")

if __name__ == "__main__":
    train_model()