import os
import sys
from recommender import DiabetesDietRecommender

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def run_cli():
    print("---------------------------------------")
    print("  Diabetes Diet Recommender (CLI)      ")
    print("---------------------------------------")

    recommender = None

    try:
        recommender = DiabetesDietRecommender(
            food_data_path="data/FoodData_Cleaned.csv",
            models_dir="models"
        )
        print("Recommender engine initialized successfully.")

    except Exception as e:
        print(f"Error initializing recommender: {e}")
        print("Please ensure 'preprocess.py' and 'training.py' have been run and data/models are in place.")
        return  # Exit if recommender fails to initialize

    while True:
        print("\n--- Enter Patient Data for Recommendation ---")
        fbs_input = input("Enter Fasting Blood Sugar (FBS) level (mg/dL) [optional, press Enter to skip]: ")
        rbs_input = input("Enter Random Blood Sugar (RBS) level (mg/dL) [optional, press Enter to skip]: ")

        fbs_level = float(fbs_input) if fbs_input else None
        rbs_level = float(rbs_input) if rbs_input else None

        # Basic input validation for blood sugar levels
        if fbs_level is None and rbs_level is None:
            print("Error: You must provide at least one blood sugar level (FBS or RBS).")
            continue

        meal_type = input("Enter Meal Type (breakfast, lunch, dinner, snack) [default: lunch]: ").strip().lower()
        if meal_type not in ['breakfast', 'lunch', 'dinner', 'snack']:
            if meal_type == "":
                meal_type = "lunch"  # Default if empty
            else:
                print("Invalid meal type. Using 'lunch' as default.")
                meal_type = "lunch"

        num_alternatives_input = input("Number of alternatives per slot [default: 1]: ").strip()
        num_alternatives_per_slot = 1
        try:
            if num_alternatives_input:
                num_alternatives_per_slot = int(num_alternatives_input)
                if num_alternatives_per_slot < 1:
                    print("Number of alternatives must be at least 1. Using 1.")
                    num_alternatives_per_slot = 1
        except ValueError:
            print("Invalid number for alternatives. Using 1.")

        try:
            recommendations = recommender.recommend_diet(
                fbs_level=fbs_level,
                rbs_level=rbs_level,
                meal_type=meal_type,
                num_alternatives_per_slot=num_alternatives_per_slot
            )

            print("\n--- Your Personalized Meal Recommendation ---")
            for item in recommendations:
                print(f"- {item}")
            print("--------------------------------------------")

        except ValueError as ve:
            print(f"\nRecommendation Error: {ve}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

        again = input("\nGet another recommendation? (yes/no): ").strip().lower()
        if again != 'yes':
            print("Exiting the Recommender CLI. Goodbye!")
            break


if __name__ == "__main__":
    run_cli()
