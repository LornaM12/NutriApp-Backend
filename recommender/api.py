from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import os
import sys
from .recommender_engine import DiabetesDietRecommender
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from dotenv import load_dotenv
import bcrypt
import uuid

# --- Load environment variables from .env file ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for data
class Feedback(BaseModel):
    rating: int
    feedback_text: Optional[str] = None
    contact_email: Optional[str] = None


class UserSignup(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


# --- GOOGLE SHEETS CONFIGURATION (using environment variables) ---
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
CREDENTIALS_FILE_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH")

if GOOGLE_SHEET_ID is None:
    logging.error("Environment variable GOOGLE_SHEET_ID is not set.")
    sys.exit(1)

if CREDENTIALS_FILE_PATH is None:
    logging.error("Environment variable GOOGLE_CREDENTIALS_PATH is not set.")
    sys.exit(1)


# --- GOOGLE SHEETS INITIALIZATION (updated to handle multiple sheets) ---
def initialize_worksheet(worksheet_name, headers):
    """Connects to a specific worksheet and ensures the header row exists."""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE_PATH, scope)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)

        try:
            sheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(worksheet_name, rows="100", cols="20")

        if not sheet.row_values(1):
            sheet.append_row(headers)

        logger.info(f"Successfully connected to worksheet '{worksheet_name}'.")
        return sheet

    except Exception as e:
        logger.error(f"Error initializing Google Sheet worksheet '{worksheet_name}': {e}")
        return None


# Initialize the Google Sheet clients globally
try:
    SHEET_CLIENT_FEEDBACK = initialize_worksheet("Feedback", ['timestamp', 'rating', 'feedback_text', 'contact_email'])
    SHEET_CLIENT_USERS = initialize_worksheet("Users", ['user_id', 'username', 'hashed_password'])
except Exception:
    SHEET_CLIENT_FEEDBACK = None
    SHEET_CLIENT_USERS = None

# Path for the Recommender Engine
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

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
    "https://nutrition-app-ivory.vercel.app",  # Your deployed frontend URL
    "http://localhost:63342",
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
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


# API Endpoint - POST Request for feedback
@app.post("/submit_feedback")
async def submit_feedback(feedback: Feedback):
    if SHEET_CLIENT_FEEDBACK is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        timestamp = datetime.now().isoformat()
        data_to_write = [
            timestamp,
            feedback.rating,
            feedback.feedback_text,
            feedback.contact_email
        ]

        SHEET_CLIENT_FEEDBACK.append_row(data_to_write)

        logger.info(f"Feedback submitted to Google Sheet: {feedback.rating}")
        return {"message": "Feedback submitted successfully!"}

    except Exception as e:
        logger.error(f"Error saving feedback to Google Sheet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


# API Endpoint - POST Request for signup
@app.post("/signup")
async def signup_user(user: UserSignup):
    if SHEET_CLIENT_USERS is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        # Check if username already exists
        all_users = SHEET_CLIENT_USERS.get_all_records()
        if any(u['username'] == user.username for u in all_users):
            raise HTTPException(status_code=400, detail="Username already registered.")

        # Hash the password
        hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Generate a unique user ID
        user_id = str(uuid.uuid4())

        # Prepare data for Google Sheet (matching the new order)
        data_to_write = [
            user_id,
            user.username,
            hashed_password
        ]

        # Append the new user row to the Google Sheet
        SHEET_CLIENT_USERS.append_row(data_to_write)

        logger.info(f"New user signed up: {user.username}")
        return {"message": "User registered successfully!", "user_id": user_id}

    except Exception as e:
        logger.error(f"Error during user signup: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during signup: {e}")

# API Endpoint - POST Request for login
@app.post("/login")
async def login_user(user: UserLogin):
    if SHEET_CLIENT_USERS is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        # Find the user by username
        all_users = SHEET_CLIENT_USERS.get_all_records()
        user_in_db = next((u for u in all_users if u['username'] == user.username), None)

        if not user_in_db:
            raise HTTPException(status_code=400, detail="Invalid username or password.")

        # Check the password
        if bcrypt.checkpw(user.password.encode('utf-8'), user_in_db['hashed_password'].encode('utf-8')):
            logger.info(f"User logged in: {user.username}")
            return {"message": "Login successful!", "user_id": user_in_db['user_id']}
        else:
            raise HTTPException(status_code=400, detail="Invalid username or password.")

    except Exception as e:
        logger.error(f"Error during user login: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during login: {e}")


@app.get("/")
async def read_root():
    return {"message": "Diabetes Diet Recommender API is running!"}
