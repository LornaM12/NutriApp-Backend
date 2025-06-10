from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import os
import sys
from .recommender_engine import DiabetesDietRecommender
from fastapi.middleware.cors import CORSMiddleware

#  Path for the Recommender Engine
current_script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_script_dir)

# project root directory.
project_root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

# Initialize FastAPI
app = FastAPI(
    title="Diet Recommender API",
    description="API for personalized diet based on blood sugars",
    version="1.0.0"
)

# CORS Configuration - For the frontend

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",  # running port
    "https://nutrition-app-ivory.vercel.app",
    "http://localhost:63342",  #
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:5500",
    "http://127.0.0.1:5500",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow for cookies to be included in cross-origin requests
    allow_methods=["*"],  # HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # HTTP headers in cross-origin requests
)

try:

    food_data_full_path = os.path.join(current_script_dir, "data", "FoodData_Cleaned.csv")
    models_full_dir = os.path.join(current_script_dir, "models")

    recommender_instance = DiabetesDietRecommender(
        food_data_path=food_data_full_path,
        models_dir=models_full_dir
    )
    print("Diabetes Diet Recommender Initialized Successfully")

except Exception as e:
    print(f"Failed to initialize Diabetes Diet Recommender: {e}")
    print("Check file locations and ensure they are correct")
    recommender_instance = None


# API Endpoint - GET Request
@app.get("/recommend_meal", response_model=List[str])
async def recommend_meal(
        fbs_level: Optional[float] = Query(None, description="Fasting Blood Sugar level in mg/dL"),
        rbs_level: Optional[float] = Query(None, description="Random Blood Sugar level in mg/dL"),
        meal_type: str = Query("lunch", description="Type of meal: 'breakfast', 'lunch', 'dinner', 'snack'"),
        num_alternatives_per_slot: int = Query(1, ge=1,
                                               description="Number of alternative food items per meal component")
):
    if recommender_instance is None:
        raise HTTPException(status_code=500, detail="Recommender instance is not initialized.")

    if fbs_level is None and rbs_level is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of the two inputs must be provided"
        )
    try:
        recommendations = recommender_instance.recommend_diet(
            fbs_level=fbs_level,
            rbs_level=rbs_level,
            meal_type=meal_type,
            num_alternatives_per_slot=num_alternatives_per_slot
        )
        return recommendations
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during recommendation: {e}")


@app.get("/")
async def read_root():
    return {"message": "Diabetes Diet Recommender API is running!"}
