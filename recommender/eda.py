import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("data/KenyaFoodData_clean.csv")

# Distribution of Key Nutrients
nutrients = ["calories", "carbohydrates_g", "protein_g", "fat_g", "fiber_g"]

for nutrient in nutrients:
    if nutrient in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[nutrient], bins=30, kde=True, color="skyblue")
        plt.title(f"Distribution of {nutrient}")
        plt.xlabel(nutrient)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", cbar=True)
plt.title("Nutrient Correlation Heatmap")
plt.tight_layout()
plt.show()

# Carbohydrates vs Fiber (scatter) 
if "carbohydrates_g" in df.columns and "fiber_g" in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="carbohydrates_g", y="fiber_g")
    plt.title("Carbohydrates vs Fiber")
    plt.xlabel("Carbohydrates (g)")
    plt.ylabel("Fiber (g)")
    plt.tight_layout()
    plt.show()

#  Sodium-Potassium Ratio Distribution 
if "sodium_potassium_ratio" in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df["sodium_potassium_ratio"], bins=30, kde=True, color="salmon")
    plt.title("Distribution of Sodium-Potassium Ratio")
    plt.xlabel("Sodium / Potassium Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

#  Top 10 High-Fiber Foods
if "fiber_g" in df.columns:
    top_fiber = df.nlargest(10, "fiber_g")[["food_name", "fiber_g"]]
    plt.figure(figsize=(8,5))
    sns.barplot(data=top_fiber, x="fiber_g", y="food_name", palette="viridis")
    plt.title("Top 10 Foods by Fiber Content")
    plt.xlabel("Fiber (g)")
    plt.ylabel("Food")
    plt.tight_layout()
    plt.show()
