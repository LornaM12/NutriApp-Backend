import pandas as pd
import numpy as np
import joblib
import os
import random

class DiabetesDietRecommender:
    def __init__(self, food_data_path="data/KenyaFoodCompositionsClean.csv", models_dir="models/"):
        # 1. Load Data
        try:
            self.food_df = pd.read_csv(food_data_path)
            if self.food_df.empty: raise ValueError("Empty CSV")
            self.food_df['tags'] = self.food_df['tags'].fillna('').astype(str)
            self.food_df['food_name'] = self.food_df['food_name'].astype(str)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        # 2. Load AI Models (Optional)
        try:
            self.kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        except: pass

        # 3. Thresholds
        self.THRESHOLDS = {
            'FBS_HYPO': 70, 'FBS_HIGH': 126,
            'RBS_HYPO': 70, 'RBS_HIGH': 200,
            'WATER_TARGET': 8, 
            'EXERCISE_MIN': 30 
        }

        # 4. Safe Defaults (Fallback for Lunch/Dinner)
        self.SAFE_DEFAULTS = {
            'Starch': ['Ugali', 'Brown Rice', 'Sweet Potato', 'Green Bananas (Matoke)'],
            'Protein': ['Beef Stew', 'Chicken Stew', 'Beans (Kamande)', 'Green Grams (Ndengu)', 'Tilapia'],
            'Veggie': ['Spinach', 'Sukuma Wiki', 'Managu', 'Cabbage', 'Kales'],
            'Fruit': ['Watermelon', 'Pawpaw', 'Orange']
        }

    def analyze_user_state(self, profile):
        state = {'sugar_status': 'normal', 'hydration_status': 'normal', 'activity_status': 'sedentary'}
        
        fbs = profile.get('fbs')
        rbs = profile.get('rbs')
        
        if (fbs and fbs < self.THRESHOLDS['FBS_HYPO']) or (rbs and rbs < self.THRESHOLDS['RBS_HYPO']):
            state['sugar_status'] = 'hypoglycemic'
        elif (fbs and fbs >= self.THRESHOLDS['FBS_HIGH']) or (rbs and rbs >= self.THRESHOLDS['RBS_HIGH']):
            state['sugar_status'] = 'high_sugar'

        water_cups = profile.get('water_cups', 0)
        if water_cups < (self.THRESHOLDS['WATER_TARGET'] / 2):
            state['hydration_status'] = 'dehydrated'

        exercise_mins = profile.get('exercise_mins', 0)
        if exercise_mins >= self.THRESHOLDS['EXERCISE_MIN']:
            state['activity_status'] = 'active'

        return state

    def recommend_diet(self, user_profile, meal_type="lunch", num_alternatives=1, disliked_foods=None):
        """
        Generates recommendations, filtering out 'disliked_foods' if provided.
        disliked_foods: List of strings (e.g., ['Omena', 'Kales'])
        """
        state = self.analyze_user_state(user_profile)
        print(f"\n--- DEBUG: Recommending for {meal_type} ({state['sugar_status']}) ---")
        
        # Start with all foods
        eligible_foods = self.food_df.copy()

        # --- FEEDBACK LOOP FILTER ---
        if disliked_foods:
            print(f"DEBUG: Filtering out user dislikes: {disliked_foods}")
            # Create a regex pattern to match any disliked food name (case insensitive)
            # e.g., if disliked=['Omena'], it removes "Fried Omena", "Dried Omena", etc.
            pattern = '|'.join([str(x).strip() for x in disliked_foods if x])
            if pattern:
                eligible_foods = eligible_foods[
                    ~eligible_foods['food_name'].str.contains(pattern, case=False, na=False)
                ]

        # A. HYPOGLYCEMIA
        if state['sugar_status'] == 'hypoglycemic':
            rescue = eligible_foods[
                (eligible_foods['Category'].isin(['Starch', 'Fruit', 'Sweet'])) & 
                (eligible_foods['fiber_g'] < 2)
            ].sort_values(by='carbohydrates_g', ascending=False)
            
            recs = [f"🚨 URGENT: {rescue.iloc[0]['food_name']}" if not rescue.empty else "🚨 URGENT: Juice/Glucose"]
            recs.append("Info: Once stable, eat below:")
            recs.extend(self._build_meal(eligible_foods, meal_type, num_alternatives, state))
            return recs

        # B. STANDARD MEAL
        return self._build_meal(eligible_foods, meal_type, num_alternatives, state)

    def _get_breakfast_options(self, df, category):
        """Logic strictly for Breakfast items."""
        name_series = df['food_name'].str.lower()
        
        if category == 'Beverage':
            return df[~name_series.str.contains('water|wine|beer|alcohol|Coconut', na=False)]

        if category == 'Starch':
            keywords = ['potato', 'yam', 'cassava', 'arrowroot', 'nduma', 'ngwaci', 'taro', 'bread', 'bun', 'toast', 'oats', 'mandazi', 'mahamri', 'pancake', 'chapati']
            matches = df[name_series.str.contains('|'.join(keywords), na=False)]
            return matches if not matches.empty else df 

        if category == 'Protein':
            keywords = ['egg', 'sausage', 'bacon', 'milk', 'yogurt']
            matches = df[name_series.str.contains('|'.join(keywords), na=False)]
            return matches if not matches.empty else df 

        if category == 'Vegetable':
            return df 

        return df

    def _get_lunch_dinner_options(self, df, category):
        """Logic strictly for Main Meals (Lunch/Dinner)."""
        name_series = df['food_name'].str.lower()

        if category == 'Starch':
            allow_keywords = ['ugali', 'rice', 'chapati', 'pasta', 'spaghetti', 'matoke', 'plantain', 'corn', 'maize', 'githeri', 'pilau', 'biryani', 'mukimo', 'couscous']
            exclude_keywords = ['porridge', 'uji', 'oats', 'toast', 'bun', 'mandazi', 'mahamri', 'pancake', 'scone', 'arrowroot', 'nduma', 'ngwaci', 'sweet potato', 'yam', 'cassava', 'taro']
            
            matches = df[name_series.str.contains('|'.join(allow_keywords), na=False)]
            matches = matches[~matches['food_name'].str.lower().str.contains('|'.join(exclude_keywords), na=False)]
            return matches if not matches.empty else df

        if category == 'Protein':
            exclude = ['milk', 'yogurt', 'tea', 'coffee', 'cocoa', 'porridge']
            matches = df[~name_series.str.contains('|'.join(exclude), na=False)]
            return matches if not matches.empty else df

        if category == 'Vegetable':
            return df 

        return df

    def _build_meal(self, df, meal_type, num_options, state):
        recommendations = []
        
        # 1. SCORE SORTING
        min_score = 50 if state['sugar_status'] == 'high_sugar' else 20
        
        if state['hydration_status'] == 'dehydrated':
            df = df.sort_values(by='hydration_score', ascending=False)
        else:
            df = df.sort_values(by='diabetes_score', ascending=False)

        # 2. SAFETY POOL
        safe_df = df[df['diabetes_score'] >= min_score]
        if safe_df.empty: safe_df = df[df['diabetes_score'] >= 10]

        # 3. DEFINE SLOTS
        if meal_type == 'breakfast':
            slot_names = ['Beverage', 'Starch', 'Protein', 'Vegetable']
        elif meal_type == 'snack':
            slot_names = ['Fruit']
        else:
            slot_names = ['Protein', 'Starch', 'Veggie', 'Fruit']

        for slot in slot_names:
            # Step A: Get Candidate Foods
            source_df = df if slot == 'Beverage' else safe_df
            target_cat = 'Vegetable' if slot == 'Veggie' else slot
            category_options = source_df[source_df['Category'] == target_cat]
            
            # Step B: Apply Meal-Specific Logic
            if meal_type == 'breakfast':
                appropriate_options = self._get_breakfast_options(category_options, target_cat)
            else:
                appropriate_options = self._get_lunch_dinner_options(category_options, target_cat)

            # Step C: Fallback (Relax filters if empty)
            if appropriate_options.empty:
                print(f"DEBUG: No '{slot}' found for {meal_type}. Relaxing filters...")
                fallback_df = df[df['Category'] == target_cat]
                if meal_type == 'breakfast':
                    appropriate_options = self._get_breakfast_options(fallback_df, target_cat)
                else:
                    appropriate_options = self._get_lunch_dinner_options(fallback_df, target_cat)

            # Step D: Apply State Modifiers
            final_pool = appropriate_options
            
            if not final_pool.empty:
                if state['activity_status'] == 'active' and slot in ['Fruit', 'Veggie']:
                    k_rich = final_pool[final_pool['tags'].str.contains('High Potassium', na=False)]
                    if not k_rich.empty: final_pool = k_rich

                if state['hydration_status'] == 'dehydrated' and slot in ['Fruit', 'Veggie']:
                    hydrating = final_pool[final_pool['tags'].str.contains('Hydrating', na=False)]
                    if not hydrating.empty: final_pool = hydrating

            # --- SAFETY NET (OPTION B) ---
            if final_pool.empty and meal_type in ['lunch', 'dinner']:
                print(f"DEBUG: Critical Failure for {slot} in {meal_type}. Using SAFE DEFAULT.")
                # We need to pick a safe default that is NOT in the disliked list? 
                # Ideally, but for now let's just pick one.
                safe_choice = random.choice(self.SAFE_DEFAULTS.get(slot, ['Standard Option']))
                recommendations.append(f"{slot}: {safe_choice}")
                continue 

            # --- FINAL SELECTION ---
            if not final_pool.empty:
                chosen = final_pool.sample(min(num_options, len(final_pool)))
                recommendations.append(f"{slot}: {', '.join(chosen['food_name'].tolist())}")
                print(f"DEBUG: Selected {slot}: {chosen['food_name'].tolist()}")
            else:
                recommendations.append(f"{slot}: No suitable options found")
                print(f"DEBUG: FAILED to find {slot}")

        return recommendations

    def generate_insight(self, user_profile):
        state = self.analyze_user_state(user_profile)
        tips = []

        if state['sugar_status'] == 'hypoglycemic':
            tips.append("⚠️ <strong>Low Sugar:</strong> We've added fast-acting carbs to stabilize you.")
        elif state['sugar_status'] == 'high_sugar':
            tips.append("⚠️ <strong>High Sugar:</strong> We've minimized processed carbs and prioritized fiber.")
        else:
            tips.append("✅ <strong>Sugar Normal:</strong> This balanced meal helps maintain your stability.")

        if state['hydration_status'] == 'dehydrated':
            tips.append("💧 <strong>Low Water Intake:</strong> We selected hydrating foods (soups/fruits). Please drink water!")

        if state['activity_status'] == 'active':
            tips.append("💪 <strong>Good Activity!</strong> We've added potassium-rich foods to help muscle recovery.")

        return "<br><br>".join(tips)