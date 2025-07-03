from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import os
import sys
from .recommender_engine import DiabetesDietRecommender
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv # Still imported, but feedback will go to DB
from datetime import datetime

# NEW IMPORTS FOR DATABASE
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import urllib.parse # For parsing DB URL
import logging # For better logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a Pydantic model for feedback data
class Feedback(BaseModel):
    rating: int
    feedback_text: Optional[str] = None
    contact_email: Optional[str] = None

# --- DATABASE CONFIGURATION (NEW) ---
# Get database URL from environment variable (Render will provide this)
# For local testing, you can set this in your environment or directly here for now.
# Example: DATABASE_URL = "postgresql://user:password@localhost/dbname"
DATABASE_URL = "postgresql://feedback_db_aval_user:I9Hn5kvMMtrlv0hCsTtHxmuk3kMX9Cxi@dpg-d1j6mnidbo4c73c7j8o0-a.oregon-postgres.render.com/feedback_db_aval"

if not DATABASE_URL:
    # Fallback for local development if not set as env var
    # REPLACE THIS WITH YOUR ACTUAL RENDER INTERNAL DATABASE URL FOR LOCAL TESTING
    # Example: "postgresql://render_user:render_password@dpg-xxxxxx.oregon-postgres.render.com/nutriapp_db_name"
    logger.warning("DATABASE_URL environment variable not set. Using a placeholder for local testing.")
    DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/your_database_name"
    # IMPORTANT: For Render deployment, ensure DATABASE_URL is set as an environment variable
    # in your Render service settings.

# Parse the database URL to handle special characters in password if any
# This is especially important if your password contains characters like '#' or '@'
parsed_url = urllib.parse.urlparse(DATABASE_URL)
db_user = parsed_url.username
db_password = urllib.parse.quote_plus(parsed_url.password) if parsed_url.password else ''
db_host = parsed_url.hostname
db_port = parsed_url.port if parsed_url.port else 5432
db_name = parsed_url.path[1:] # Remove leading slash

# Reconstruct the URL with properly quoted password
DB_CONNECTION_STRING = (
    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# SQLAlchemy setup
Engine = create_engine(DB_CONNECTION_STRING)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Base = declarative_base()

# Define the Feedback table (NEW)
class FeedbackDB(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    rating = Column(Integer, nullable=False)
    feedback_text = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)

# Create tables in the database (if they don't exist)
def create_db_tables():
    try:
        Base.metadata.create_all(bind=Engine)
        logger.info("Database tables created successfully or already exist.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        # Optionally, raise the exception if database is critical for app startup
        # raise e

# Call this function when the app starts up
create_db_tables()


# Path for the Recommender Engine
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

# project root directory. (This might not be strictly needed with absolute imports for recommender_engine)
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
    "https://nutrition-app-ivory.vercel.app", # Your deployed frontend URL
    "http://localhost:63342",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    food_data_full_path = os.path.join(current_script_dir, "data", "FoodData_Cleaned.csv")
    models_full_dir = os.path.join(current_script_dir, "models")

    recommender_instance = DiabetesDietRecommender(
        food_data_path=food_data_full_path,
        models_dir=models_full_dir
    )
    logger.info("Diabetes Diet Recommender Initialized Successfully")

except Exception as e:
    logger.error(f"Failed to initialize Diabetes Diet Recommender: {e}")
    logger.error("Check file locations and ensure they are correct")
    recommender_instance = None


# REMOVED: CSV file path and initialization (no longer needed for feedback)
# FEEDBACK_CSV_PATH = os.path.join(current_script_dir, "data", "feedback.csv")
# os.makedirs(os.path.dirname(FEEDBACK_CSV_PATH), exist_ok=True)
# if not os.path.exists(FEEDBACK_CSV_PATH):
#     with open(FEEDBACK_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['timestamp', 'rating', 'feedback_text', 'contact_email'])
#     logger.info(f"Created new feedback CSV file at: {FEEDBACK_CSV_PATH}")


# API Endpoint - GET Request for recommendations (Existing)
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


# API Endpoint - POST Request for feedback (MODIFIED for DB)
@app.post("/submit_feedback")
async def submit_feedback(feedback: Feedback):
    db = SessionLocal() # Get a new database session
    try:
        new_feedback = FeedbackDB(
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            contact_email=feedback.contact_email,
            timestamp=datetime.utcnow() # Use UTC for consistency
        )
        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback) # Refresh to get the generated ID
        logger.info(f"Feedback submitted to DB: {new_feedback.id}")
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        db.rollback() # Rollback on error
        logger.error(f"Error saving feedback to database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")
    finally:
        db.close() # Always close the session


@app.get("/")
async def read_root():
    return {"message": "Diabetes Diet Recommender API is running!"}