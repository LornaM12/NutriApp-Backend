import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
RAW_FILE_PATH = "data/KenyaFoodCompositions.xlsx"
OUTPUT_PATH = "data/KenyaFoodCompositionsClean.csv"

def clean_food_name(name):
    """Removes brackets and cleans up the name."""
    if not isinstance(name, str): return "Unknown"
    name = name.split('(')[0] # Remove (text)
    parts = name.split(',')
    return parts[0].strip()

def apply_strict_rules(row):
    """
    Implements the USER'S STRICT 15 RULES based on Group_Name.
    Returns the valid Category (Protein, Starch, Veggie, Fruit) or 'Avoid'.
    """
    # Normalize text for easy matching
    name = str(row['food_name']).lower()
    group = str(row['Group_Name']).lower().strip()
    
    # --- RULE 1: BEVERAGES ---
    if 'beverages' in group:
        if 'wine' in name: return 'Avoid'
        return 'Beverage' # Or 'Avoid' if you don't want beverages recommended as meals

    # --- RULE 2: CEREALS ---
    if 'cereal' in group:
        # Block list
        if any(x in name for x in ['cornflour', 'wheat', 'millet', 'raw']):
            return 'Avoid'
        return 'Starch'

    # --- RULE 3: CONDIMENTS ---
    if 'condiment' in group or 'spice' in group:
        return 'Avoid'

    # --- RULE 4: FISH ---
    if 'fish' in group or 'sea food' in group:
        # Allow list ONLY
        allowed = ['tilapia', 'prawn', 'tuna']
        if any(x in name for x in allowed):
            return 'Protein'
        return 'Avoid'

    # --- RULE 5: FRUITS ---
    if 'fruit' in group:
        return 'Fruit'

    # --- RULE 6: INSECTS ---
    if 'insect' in group:
        return 'Avoid'

    # --- RULE 7: LEGUMES ---
    if 'legume' in group or 'pulse' in group:
        # Allow list ONLY
        allowed = ['bean', 'chickpea', 'green gram', 'cowpea', 'lentil', 'soybean']
        if any(x in name for x in allowed):
            return 'Protein'
        return 'Avoid'

    # --- RULE 8: MEATS ---
    if 'meat' in group or 'poultry' in group or 'egg' in group:
        # Block list
        if any(x in name for x in ['duck', 'blood', 'raw']):
            return 'Avoid'
        return 'Protein'

    # --- RULE 9: DAIRY ---
    if 'milk' in group or 'dairy' in group:
        # Allow list ONLY
        if 'milk' in name:
            return 'Protein' # Milk counts as protein/drink
        return 'Avoid'

    # --- RULE 10: NUTS ---
    if 'nut' in group or 'seed' in group:
        return 'Avoid'

    # --- RULE 11: OILS ---
    if 'oil' in group or 'fat' in group:
        return 'Avoid'

    # --- RULE 12: TUBERS (STARCHY ROOTS) ---
    if 'root' in group or 'tuber' in group:
        # Block list
        if any(x in name for x in ['turnip', 'taro', 'beetroot']):
            return 'Avoid'
        return 'Starch'

    # --- RULE 13: SUGAR ---
    if 'sugar' in group or 'sweet' in group:
        if any(x in name for x in ['sugar', 'sugarcane']):
            return 'Avoid'
        return 'Sweet' # Likely filtered out later for diabetics anyway

    # --- RULE 14: VEGETABLES ---
    if 'vegetable' in group:
        return 'Vegetable'

    # --- RULE 15: MIXED DISHES ---
    if 'mixed' in group:
        # Simple Macro Check to decide if it's Starch or Protein dish
        if row['protein_g'] > 8: return 'Protein'
        if row['carbohydrates_g'] > 15: return 'Starch'
        return 'Other'

    # Fallback for unknown groups
    return 'Avoid'

def calculate_holistic_scores(row):
    net_carbs = max(0, row['carbohydrates_g'] - row['fiber_g'])
    
    # Diabetic Score
    d_raw = (
        (row['fiber_g'] * 8) + 
        (row['protein_g'] * 1.5) - 
        (net_carbs * 0.5) - 
        (row['calories'] * 0.01)
    )
    
    # Hydration Score (g water)
    h_raw = row['water_g']
    
    return pd.Series([d_raw, h_raw])

def load_and_process():
    print("--- Starting Strict Data Preprocessing ---")
    
    try:
        # Load Data (Header 2 for Food Data, Header 1 for Groups)
        df = pd.read_excel(RAW_FILE_PATH, sheet_name="Food Data", header=2)
        df_groups = pd.read_excel(RAW_FILE_PATH, sheet_name="Food Groups", header=1)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Clean Headers
    df.columns = [str(c).strip() for c in df.columns]
    df_groups.columns = [str(c).strip() for c in df_groups.columns]
    
    # Handle Duplicates
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    # Rename
    col_map = {
        'Code': 'food_code', 'Food name': 'food_name', 'Energy': 'energy_kj', 'Energy.1': 'calories',
        'Water': 'water_g', 'Protein': 'protein_g', 'Fat': 'fat_g', 
        'Carbohydrate available': 'carbohydrates_g', 'Fibre': 'fiber_g',
        'Na': 'sodium_mg', 'K': 'potassium_mg', 'Cholesterol': 'cholesterol_mg'
    }
    df.rename(columns=col_map, inplace=True)
    if 'calories' not in df.columns and 'energy_kj' in df.columns: df['calories'] = df['energy_kj'] * 0.239

    # Clean Rows & Numerics
    df = df[df['food_name'].notna()]
    df = df[~df['food_name'].str.isupper()] # Remove header repeats
    df = df[~df['food_name'].astype(str).str.contains("SD|min|max", regex=True)]
    
    numeric_cols = ['calories', 'water_g', 'protein_g', 'fat_g', 'carbohydrates_g', 'fiber_g', 'sodium_mg', 'potassium_mg', 'cholesterol_mg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[', '').str.replace(']', ''), errors='coerce').fillna(0)

    # MAP GROUPS
    try:
        group_map = dict(zip(df_groups['CODE'], df_groups['FOOD GROUPS']))
    except:
        group_map = dict(zip(df_groups.iloc[:, 0], df_groups.iloc[:, 1]))

    df['Group_Name'] = df['food_code'].apply(lambda x: group_map.get(int(x)//1000, "Unknown") if pd.notnull(x) and str(x).isdigit() else "Unknown")

    # --- APPLY STRICT RULES ---
    df['Category'] = df.apply(apply_strict_rules, axis=1)
    
    # Filter out 'Avoid' immediately
    print(f"Total items before filter: {len(df)}")
    df = df[df['Category'] != 'Avoid']
    print(f"Total items after strict filter: {len(df)}")

    # Clean Name
    df['clean_name'] = df['food_name'].apply(clean_food_name)

    # Merge Duplicates
    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict['Category'] = 'first'
    agg_dict['Group_Name'] = 'first'
    final_df = df.groupby('clean_name', as_index=False).agg(agg_dict)
    final_df.rename(columns={'clean_name': 'food_name'}, inplace=True)

    # Scores & Tags
    scores = final_df.apply(calculate_holistic_scores, axis=1)
    final_df['diabetes_score'] = scores[0]
    final_df['hydration_score'] = scores[1]
    
    # Normalize
    for col in ['diabetes_score', 'hydration_score']:
        mn, mx = final_df[col].min(), final_df[col].max()
        final_df[col] = ((final_df[col] - mn) / (mx - mn)) * 100

    def generate_tags(row):
        tags = [row['Category']]
        if row['protein_g'] > 10: tags.append('High Protein')
        if row['fiber_g'] > 3: tags.append('High Fiber')
        if row['hydration_score'] > 70: tags.append('Hydrating')
        if row['sodium_mg'] > 400: tags.append('High Sodium')
        if row['potassium_mg'] > 250: tags.append('High Potassium')
        if row.get('cholesterol_mg', 0) == 0: tags.append('Vegetarian')
        return ", ".join(tags)

    final_df['tags'] = final_df.apply(generate_tags, axis=1)

    # Save
    cols_to_save = ['food_name', 'Category', 'Group_Name', 'diabetes_score', 'hydration_score', 'tags'] + numeric_cols
    final_df[cols_to_save].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ SUCCESS! Processed {len(final_df)} valid foods. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_process()