from .recommender_engine import DiabetesDietRecommender
import os


def recommend_diet(patient_profile, meal_type, num_alternatives_per_slot=1):

    # Instantiate the recommender.
    recommender = DiabetesDietRecommender(
        food_data_path=os.path.join(os.path.dirname(__file__), "data/clustered_food_data.csv"),
        models_dir=os.path.join(os.path.dirname(__file__), "models/")
    )

    fbs_level = patient_profile.get('FBS')
    rbs_level = patient_profile.get('RBS')

    if fbs_level is None and rbs_level is None:
        print("Error: Blood sugar readings (RBS or FBS) are essential for recommendations.")
        print("Please ensure your patient profile includes 'RBS' or 'FBS' with a valid numeric value.")
        return ["Cannot generate recommendations: Missing crucial blood sugar readings."]

    current_sugar_display = []
    if fbs_level is not None:
        current_sugar_display.append(f"FBS: {fbs_level} mg/dL")
    if rbs_level is not None:
        current_sugar_display.append(f"RBS: {rbs_level} mg/dL")

    print(f"\nPatient Sugar Readings: {', '.join(current_sugar_display)}")

    recommendations = recommender.recommend_diet(
        fbs_level=fbs_level,
        rbs_level=rbs_level,
        meal_type=meal_type,
        num_alternatives_per_slot=num_alternatives_per_slot
    )
    return recommendations


if __name__ == "__main__":

    patient_test_profile = {

        "FBS": 90,
        "RBS": 90
    }

    print("\n" + "=" * 70)
    print("Patient Meal Plan")
    print("Patient Profile:", patient_test_profile)
    print("=" * 70)

    print("\nBreakfast Recommendation")
    breakfast_recs = recommend_diet(patient_test_profile, meal_type="breakfast")
    print(breakfast_recs)

    print("\nLunch Recommendation")
    lunch_recs = recommend_diet(patient_test_profile, meal_type="lunch")
    print(lunch_recs)

    print("\nDinner Recommendation")
    dinner_recs = recommend_diet(patient_test_profile, meal_type="dinner")
    print(dinner_recs)
