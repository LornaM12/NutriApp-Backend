import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_food_data (food_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Kenyan food dataset:
    1. Clean column names
    2. Standardize units (all nutrients in grams/100g, calories in kcal)
    3. Handle missing values
    4. Keep diabetes-relevant features
    5. Engineer useful features (ratios, densities, etc.)
    """

    # Clean Column names
    food_df.columns = (
        food_df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"\(g\)", "_g", regex=True)
        .str.replace(r"\(mg\)", "_mg", regex=True)
        .str.replace(r"\(mcg\)", "_mcg", regex=True)
        .str.replace(r"\(.*\)", "", regex=True)  # remove leftover brackets
        .str.lower()
        .str.replace("__", "_")
    )

    print("Columns after cleaning:", food_df.columns.tolist())



    #Standardize feature units to grams (mg → g, mcg → g)
    mg_columns = ["cholesterol_mg", "sodium_mg", "potassium_mg", 
                  "calcium_mg", "iron_mg", "magnesium_mg", "vitamin_c_mg"]
    mcg_columns = ["thiamin_mcg", "riboflavin_mcg", "vitamin_a_mcg_rae"]

    for col in mg_columns:
        if col in food_df.columns:
            food_df[col.replace("_mg", "_g")] = food_df[col] / 1000
            food_df.drop(columns=[col], inplace=True)

    for col in mcg_columns:
        if col in food_df.columns:
            food_df[col.replace("_mcg", "_g")] = food_df[col] / 1_000_000
            food_df.drop(columns=[col], inplace=True)

    # Convert numeric columns
    for col in food_df.columns:
        if col not in ['food_code', 'food_name']:
            food_df[col] = pd.to_numeric(food_df[col], errors='coerce')

    # Fill missing values
    food_df = food_df.fillna(food_df.median(numeric_only=True))


    # Feature engineering
    if 'carbohydrates_g' in food_df.columns and 'fiber_g' in food_df.columns:
        food_df['carb_fiber_ratio'] = food_df['carbohydrates_g'] / (food_df['fiber_g'] + 1)

    if 'protein_g' in food_df.columns and 'calories' in food_df.columns:
        food_df['protein_density'] = food_df['protein_g'] / (food_df['calories'] + 1)

    if 'fat_g' in food_df.columns and 'calories' in food_df.columns:
        food_df['fat_energy_ratio'] = (food_df['fat_g'] * 9) / (food_df['calories'] + 1)

    if 'sodium_g' in food_df.columns and 'potassium_g' in food_df.columns:
        food_df['sodium_potassium_ratio'] = food_df['sodium_g'] / (food_df['potassium_g'] + 1e-6)


    # Scale numeric values (MinMaxScaler)
    scaler = MinMaxScaler()
    numeric_cols = [col for col in food_df.columns if col not in ['food_code', 'food_name']]

    food_df[numeric_cols] = scaler.fit_transform(food_df[numeric_cols])

    return food_df


if __name__ == "__main__":

    df = pd.read_csv("data/KenyaFoodData.csv", encoding = "latin1")
    clean_df = preprocess_food_data(df)
    clean_df.to_csv("data/KenyaFoodData_clean.csv", index=False)

    print("\nNumber of columns after preprocessing:", clean_df.shape[1])
    print("Columns after preprocessing:", clean_df.columns.tolist())
    print("\nSample rows:")
    print(clean_df.head())