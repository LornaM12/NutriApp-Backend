import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Food Data Preprocessing
def load_food_data():
    fooddata_df = pd.read_csv("data/FoodData.csv")

    # 1: Data Cleaning
    # Check for missing values
    print("Missing Values:\n", fooddata_df.isnull().sum())

    # Drop rows with missing critical nutrients
    nutrient_cols = [
        'Glycemic Index', 'Calories', 'Carbohydrates', 'Protein', 'Fat',
        'Sodium Content', 'Potassium Content', 'Magnesium Content',
        'Calcium Content', 'Fiber Content'
    ]
    fooddata_df.dropna(subset=nutrient_cols, inplace=True)

    # Check Data Types in the columns
    print("\nData Types:\n", fooddata_df.dtypes)

    # Convert mismatching columns to the desired dtypes
    fooddata_df['Suitable for Diabetes'] = pd.to_numeric(
        fooddata_df['Suitable for Diabetes'], errors='coerce'
    )

    # Check for any NaNs introduced
    missing_after_conversion = fooddata_df['Suitable for Diabetes'].isnull().sum()
    print(f"\nMissing values in 'Suitable for Diabetes' after conversion: {missing_after_conversion}\n")

    # Drop rows where conversion failed
    fooddata_df.dropna(subset=['Suitable for Diabetes'], inplace=True)

    # Convert to int for consistency (if all values are whole numbers)
    fooddata_df['Suitable for Diabetes'] = fooddata_df['Suitable for Diabetes'].astype(int)

    # Feature Engineering: Classify Foods
    def classify_food(row):
        if row['Protein'] >= 5 and row['Carbohydrates'] < 10:
            return 'Protein'
        elif row['Carbohydrates'] >= 15:
            return 'Carb'
        elif row['Calories'] < 60 and row['Fiber Content'] >= 2 and row['Carbohydrates'] <= 15:
            return 'Vegetable'
        else:
            return 'Other'

    # Apply the classification
    fooddata_df['Food Type'] = fooddata_df.apply(classify_food, axis=1)

    # Show how many foods were assigned to each category
    print("\nFood Type Distribution:\n", fooddata_df['Food Type'].value_counts())

    # Normalize numeric nutrient columns
    cols_to_normalize = [
        'Glycemic Index', 'Calories', 'Carbohydrates', 'Protein', 'Fat',
        'Sodium Content', 'Potassium Content', 'Magnesium Content',
        'Calcium Content', 'Fiber Content'
    ]

    scaler = MinMaxScaler()
    fooddata_df[cols_to_normalize] = scaler.fit_transform(fooddata_df[cols_to_normalize])

    print("\nNutrient columns normalized.")

    # Save cleaned and preprocessed data
    fooddata_df.to_csv("data/FoodData_Cleaned.csv", index=False)
    print("\nCleaned dataset saved as 'FoodData_Cleaned.csv'")

    return fooddata_df


if __name__ == "__main__":
    fooddata_df = load_food_data()


# Patient Data preprocessing
def load_patient_data():
    patient_data_df = pd.read_csv("data/PatientData.csv")

    # Check for missing values
    print("Missing values:\n", patient_data_df.isnull().sum())

    # Drop missing values
    patient_data_df.dropna(inplace=True)

    # Data Type validation
    print("Data types:\n", patient_data_df.dtypes)

    # Encode Gender Column
    gender_encoded = pd.get_dummies(patient_data_df['Gender'], prefix='Gender')
    patient_data = pd.concat([patient_data_df, gender_encoded], axis=1)

    # Convert sugar readings from mmol/L to mg/dL
    patient_data['FBS'] = patient_data['FBS'] * 18
    patient_data['RBS'] = patient_data['RBS'] * 18

    # Add New Column 'Sugar Status' to classify patients
    def classify_sugar_status(row):
        if row['HbA1c'] >= 6.5 or row['FBS'] >= 126 or row['RBS'] >= 200:
            return 'Diabetic'
        elif 5.7 <= row['HbA1c'] < 6.5 or 100 <= row['FBS'] < 126 or 140 <= row['RBS'] < 200:
            return 'Pre-diabetic'
        else:
            return 'Normal'

    patient_data_df['SugarStatus'] = patient_data_df.apply(classify_sugar_status, axis=1)

    # Save cleaned Patient Data
    patient_data_df.to_csv("data/PatientData_Cleaned.csv", index=False)

    return patient_data_df


if __name__ == "__main__":
    patient_data_df = load_patient_data()
    print(patient_data_df.head(5))
