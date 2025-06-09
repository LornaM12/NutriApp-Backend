import pandas as pd
import joblib
import os
import numpy as np


class DiabetesDietRecommender:
    def __init__(self, food_data_path="data/FoodData_Cleaned.csv", models_dir="models/"):
        # Load Food data
        try:
            self.food_df = pd.read_csv(food_data_path)
            if self.food_df.empty:
                raise ValueError(
                    f"Food data file '{food_data_path}' is empty.")
        except FileNotFoundError:
            print(f"Error: Food data file '{food_data_path}' not found.")
            print("Please ensure preprocess.py has been run successfully to generate this file.")
            raise
        except Exception as e:
            print(f"Error loading food data: {e}")
            raise

        # Load scaler and model
        try:
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            self.kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
        except FileNotFoundError:
            print(f"Error: Model files not found in {models_dir}.")
            print("Please ensure training.py has been run successfully to generate these models.")
            raise
        except Exception as e:
            print(f"Error loading model files: {e}")
            raise

        self.clustering_features = [
            'Calories', 'Carbohydrates', 'Protein', 'Fat',
            'Fiber Content', 'Glycemic Index'
        ]

        scaled_features_for_clustering = self.scaler.transform(self.food_df[self.clustering_features])
        self.food_df['Cluster'] = self.kmeans_model.predict(scaled_features_for_clustering)

        # Blood sugar thresholds - based on standard guidelines
        self.FBS_HYPO = 70
        self.FBS_NORMAL_MIN = 70
        self.FBS_NORMAL_MAX = 99
        self.FBS_HYPER = 180

        self.RBS_HYPO = 70
        self.RBS_NORMAL_MIN = 70
        self.RBS_NORMAL_MAX = 139
        self.RBS_HYPER = 200

        # Cluster characteristics (from cluster_analysis.py)
        self.cluster_characteristics = {
            0: {'name': 'Moderate Carb, High GI', 'GI_risk': 'high', 'fiber_impact': 'low', 'protein_impact': 'low',
                'focus': ['Carb', 'Other'], 'suitable_for_diabetes_prop': 0.76},
            1: {'name': 'Low Everything (Light Foods/Drinks)', 'GI_risk': 'very_low', 'fiber_impact': 'low',
                'protein_impact': 'low', 'focus': ['Vegetable', 'Other', 'Carb', 'Protein'],
                'suitable_for_diabetes_prop': 1.00},
            2: {'name': 'High Calorie/Fat, Low GI', 'GI_risk': 'very_low', 'fiber_impact': 'moderate_high',
                'protein_impact': 'moderate', 'focus': ['Carb', 'Other', 'Protein'],
                'suitable_for_diabetes_prop': 1.00},
            3: {'name': 'Pure High Carb, Moderate-High GI', 'GI_risk': 'high', 'fiber_impact': 'very_low',
                'protein_impact': 'very_low', 'focus': ['Carb'], 'suitable_for_diabetes_prop': 0.93},
            4: {'name': 'High Protein, Very Low GI', 'GI_risk': 'very_low', 'fiber_impact': 'very_low',
                'protein_impact': 'high', 'focus': ['Protein', 'Other'], 'suitable_for_diabetes_prop': 1.00}
        }

        # Defining preferred clusters for different blood sugar states for primary filtering
        self.preferred_clusters_by_sugar_state = {
            'hypoglycemic_fast': [0, 3],
            'high_sugar': [1, 2, 4],
            'normal_stable': [1, 2, 4, 0, 3],
            'unstable': [1, 2, 4]
        }

        # Defining meal components based on Food Type for balanced meals
        self.meal_composition = {
            'breakfast': ['Protein', 'Carb', 'Vegetable'],
            'lunch': ['Protein', 'Carb', 'Vegetable', 'Vegetable'],
            'dinner': ['Protein', 'Carb', 'Vegetable', 'Vegetable'],
            'snack': ['Protein', 'Vegetable', 'Carb']
        }

    # Categorize patients depending on their sugar states
    def get_blood_sugar_state(self, fbs_level=None, rbs_level=None):

        if (fbs_level is not None and fbs_level < self.FBS_HYPO) or \
                (rbs_level is not None and rbs_level < self.RBS_HYPO):
            return 'hypoglycemic'

        is_fbs_high = (fbs_level is not None and fbs_level >= self.FBS_HYPER)
        is_rbs_high = (rbs_level is not None and rbs_level >= self.RBS_HYPER)

        if is_fbs_high or is_rbs_high:
            return 'high_sugar'

        is_fbs_normal = (fbs_level is not None and self.FBS_NORMAL_MIN <= fbs_level <= self.FBS_NORMAL_MAX)
        is_rbs_normal = (rbs_level is not None and self.RBS_NORMAL_MIN <= rbs_level <= self.RBS_NORMAL_MAX)

        if (fbs_level is not None and rbs_level is not None and is_fbs_normal and is_rbs_normal) or \
                (fbs_level is not None and rbs_level is None and is_fbs_normal) or \
                (rbs_level is not None and fbs_level is None and is_rbs_normal):
            return 'normal_stable'

        if fbs_level is not None or rbs_level is not None:
            return 'unstable'

        return 'unknown'

    def recommend_diet(self, fbs_level=None, rbs_level=None, meal_type="any", num_alternatives_per_slot=1):

        sugar_state = self.get_blood_sugar_state(fbs_level=fbs_level, rbs_level=rbs_level)

        current_sugar_display = []
        if fbs_level is not None:
            current_sugar_display.append(f"FBS: {fbs_level} mg/dL")
        if rbs_level is not None:
            current_sugar_display.append(f"RBS: {rbs_level} mg/dL")

        print(
            f"\n Patient Sugar state: {sugar_state}")
        print(f"Meal Type: {meal_type.capitalize()}")

        recommendations = []
        eligible_foods = self.food_df[self.food_df['Suitable for Diabetes'] == 1].copy()

        if sugar_state == 'hypoglycemic':
            print("Action: Hypoglycemia detected. Recommending fast-acting carb.")
            fast_carb_options = eligible_foods[
                eligible_foods['Cluster'].isin(self.preferred_clusters_by_sugar_state['hypoglycemic_fast']) &
                (eligible_foods['Food Type'] == 'Carb')
                ]
            if not fast_carb_options.empty:
                chosen_carb = fast_carb_options.sort_values(by=['Glycemic Index', 'Fiber Content'],
                                                            ascending=[False, True]).head(1)
                recommendations.append(
                    f"Immediate Action: {chosen_carb['Food Name'].iloc[0]} consume immediately to raise sugar)")

            stabilizing_recs = self.get_stabilizing_meal(meal_type, num_alternatives_per_slot = 1)
            if stabilizing_recs:
                recommendations.append("Follow-up meal:")
                recommendations.extend(stabilizing_recs)
            return recommendations

        target_clusters_for_state = self.preferred_clusters_by_sugar_state.get(sugar_state,
                                                                               self.preferred_clusters_by_sugar_state[
                                                                                   'normal_stable'])

        if sugar_state in ['high_sugar', 'unstable']:
            eligible_foods = eligible_foods[eligible_foods['Cluster'].isin(target_clusters_for_state)]
            print(
                f"Filtered by sugar state '{sugar_state}': Only considering foods from clusters {target_clusters_for_state}.")
        elif sugar_state == 'normal_stable':
            print(
                f"Blood sugar is normal and stable. Considering a broad range of diabetes-suitable foods for a "
                f"balanced meal.")

        if eligible_foods.empty:
            return [
                "No suitable recommendations found based on your criteria and available food items. Please broaden "
                "your criteria or consult a dietitian."]

        meal_types_needed = self.meal_composition.get(meal_type.lower(), ['Protein', 'Carb', 'Vegetable'])

        meal_recommendations = []

        for food_type_needed in meal_types_needed:
            possible_items = []
            type_options = eligible_foods[
                (eligible_foods['Food Type'] == food_type_needed)
            ].copy()

            if type_options.empty:
                print(f"Warning: No eligible food found for '{food_type_needed}' within remaining filters.")
                continue

            if sugar_state == 'high_sugar' or sugar_state == 'unstable':
                priority_clusters_for_current_state = [
                    cluster_id for cluster_id in [1, 2, 4]
                    if cluster_id in type_options['Cluster'].unique()
                ]

                if priority_clusters_for_current_state:
                    priority_options = type_options[type_options['Cluster'].isin(priority_clusters_for_current_state)]
                    if not priority_options.empty:
                        possible_items.extend(priority_options['Food Name'].tolist())

                remaining_options = type_options[~type_options['Cluster'].isin(priority_clusters_for_current_state)]
                if not remaining_options.empty:
                    possible_items.extend(remaining_options['Food Name'].tolist())

            elif sugar_state == 'normal_stable':
                ordered_clusters = sorted(
                    type_options['Cluster'].unique(),
                    key=lambda c: (
                        0 if self.cluster_characteristics[c]['GI_risk'] == 'very_low' else
                        1 if self.cluster_characteristics[c]['GI_risk'] == 'low' else
                        2 if self.cluster_characteristics[c]['GI_risk'] == 'moderate' else
                        3 if self.cluster_characteristics[c]['GI_risk'] == 'high' else 4,
                        - (0 if self.cluster_characteristics[c]['fiber_impact'] == 'very_low' else
                           1 if self.cluster_characteristics[c]['fiber_impact'] == 'low' else
                           2 if self.cluster_characteristics[c]['fiber_impact'] == 'moderate' else
                           3 if self.cluster_characteristics[c]['fiber_impact'] == 'moderate_high' else
                           4)
                    )
                )

                for cluster_id in ordered_clusters:
                    cluster_options = type_options[type_options['Cluster'] == cluster_id]
                    if not cluster_options.empty:
                        possible_items.extend(cluster_options['Food Name'].tolist())

            unique_possible_items = list(dict.fromkeys(possible_items))

            if unique_possible_items:
                num_to_sample = min(num_alternatives_per_slot, len(unique_possible_items))

                if num_to_sample > 0:
                    selected_items = np.random.choice(unique_possible_items, num_to_sample, replace=False).tolist()
                    meal_recommendations.append(f"{food_type_needed}: {', '.join(selected_items)}")
                    eligible_foods = eligible_foods[~eligible_foods['Food Name'].isin(selected_items)]
            else:
                meal_recommendations.append(f"Could not find {food_type_needed} options.")

        if not meal_recommendations or all("Could not find" in rec for rec in meal_recommendations):
            print("Falling back to general recommendations as specific meal components were hard to find.")
            fallback_foods = self.food_df[self.food_df['Suitable for Diabetes'] == 1].sample(min(3, len(self.food_df)))
            recommendations.extend(fallback_foods['Food Name'].tolist())
        else:
            recommendations.extend(meal_recommendations)

        return recommendations

    def get_stabilizing_meal(self, meal_type, num_alternatives_per_slot):
        # This function will recommend a balanced meal after a hypo event i.e - when fbs/rbs is < 70
        print("Recommending a stabilizing meal for hypoglycemia.")
        eligible_foods = self.food_df[self.food_df['Suitable for Diabetes'] == 1].copy()

        target_clusters_stabilizing = self.preferred_clusters_by_sugar_state['normal_stable']
        eligible_foods = eligible_foods[eligible_foods['Cluster'].isin(target_clusters_stabilizing)]

        if eligible_foods.empty:
            return ["No suitable stabilizing meal found based on criteria. Please consult a dietitian."]

        meal_types_needed = self.meal_composition.get(meal_type.lower(), ['Protein', 'Carb', 'Vegetable'])
        stabilizing_recs = []
        chosen_food_names = set()

        for food_type_needed in meal_types_needed:
            possible_items = []
            type_options = eligible_foods[
                (eligible_foods['Food Type'] == food_type_needed)
            ].copy()

            if type_options.empty:
                stabilizing_recs.append(f"Could not find {food_type_needed} options for stabilization.")
                continue

            ordered_clusters = sorted(
                type_options['Cluster'].unique(),
                key=lambda c: (
                    0 if self.cluster_characteristics[c]['GI_risk'] == 'very_low' else
                    1 if self.cluster_characteristics[c]['GI_risk'] == 'low' else
                    2 if self.cluster_characteristics[c]['GI_risk'] == 'moderate' else
                    3 if self.cluster_characteristics[c]['GI_risk'] == 'high' else 4,
                    - (0 if self.cluster_characteristics[c]['fiber_impact'] == 'very_low' else
                       1 if self.cluster_characteristics[c]['fiber_impact'] == 'low' else
                       2 if self.cluster_characteristics[c]['fiber_impact'] == 'moderate' else
                       3 if self.cluster_characteristics[c]['fiber_impact'] == 'moderate_high' else
                       4)
                )
            )

            for cluster_id in ordered_clusters:
                cluster_options = type_options[type_options['Cluster'] == cluster_id]
                if not cluster_options.empty:
                    possible_items.extend(cluster_options['Food Name'].tolist())

            unique_possible_items = list(dict.fromkeys(possible_items))

            if unique_possible_items:
                num_to_sample = min(num_alternatives_per_slot, len(unique_possible_items))
                if num_to_sample > 0:
                    selected_items = np.random.choice(unique_possible_items, num_to_sample, replace=False).tolist()
                    stabilizing_recs.append(f"{food_type_needed}: {', '.join(selected_items)}")
                    eligible_foods = eligible_foods[~eligible_foods['Food Name'].isin(selected_items)]
            else:
                stabilizing_recs.append(f"Could not find {food_type_needed} options for stabilization.")

        return stabilizing_recs


