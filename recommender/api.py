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
import json
from typing import Dict, Any
from datetime import datetime, timedelta

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


class SugarReading(BaseModel):
    value: float
    timestamp: str
    meal_context: str
    user_id: str


class WaterIntake(BaseModel):
    amount: float
    unit: str
    timestamp: str
    source: Optional[str] = None
    user_id: str


class Exercise(BaseModel):
    exercise_type: str
    duration_minutes: int
    intensity: str
    timestamp: str
    notes: Optional[str] = None
    user_id: str


# --- GOOGLE SHEETS CONFIGURATION (using environment variables) ---
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH")
GOOGLE_CREDENTIALS_JSON_STRING = os.environ.get("GOOGLE_CREDENTIALS_JSON")

if GOOGLE_SHEET_ID is None:
    logging.error("Environment variable GOOGLE_SHEET_ID is not set.")
    sys.exit(1)


# --- GOOGLE SHEETS INITIALIZATION (updated to handle multiple sheets and secure credentials) ---
def initialize_worksheet(worksheet_name, headers):
    """Connects to a specific worksheet and ensures the header row exists."""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

        if GOOGLE_CREDENTIALS_JSON_STRING:
            # Case 1: Raw JSON credentials from environment variable (like on Render)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                json.loads(GOOGLE_CREDENTIALS_JSON_STRING), scope
            )
            logger.info("Using JSON string credentials.")
        elif GOOGLE_CREDENTIALS_PATH:
            # Case 2: Credentials from a file path (like on local machine)
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                GOOGLE_CREDENTIALS_PATH, scope
            )
            logger.info("Using JSON file path credentials.")
        else:
            raise ValueError(
                "No Google Sheets credentials found. Set either GOOGLE_CREDENTIALS_JSON or GOOGLE_CREDENTIALS_PATH.")

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

try:
    SHEET_CLIENT_SUGAR = initialize_worksheet("SugarReadings",
        ['timestamp', 'user_id', 'value', 'meal_context', 'created_at'])
    SHEET_CLIENT_WATER = initialize_worksheet("WaterIntake",
        ['timestamp', 'user_id', 'amount', 'unit', 'source', 'created_at'])
    SHEET_CLIENT_EXERCISE = initialize_worksheet("Exercise",
        ['timestamp', 'user_id', 'exercise_type', 'duration_minutes', 'intensity', 'notes', 'created_at'])
except Exception:
    SHEET_CLIENT_SUGAR = None
    SHEET_CLIENT_WATER = None
    SHEET_CLIENT_EXERCISE = None

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
    "http://127.0.0.1:8000",
    "https://nutrition-app-pndw.vercel.app",
    "https://nutrition-app-pi.vercel.app",   # Add comma here
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


# API Endpoint for signup
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


# API Endpoint for login
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


@app.post("/api/sugar-readings")
async def create_sugar_reading(reading: SugarReading):
    if SHEET_CLIENT_SUGAR is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        created_at = datetime.now().isoformat()
        data_to_write = [
            reading.timestamp,
            reading.user_id,
            reading.value,
            reading.meal_context,
            created_at
        ]

        SHEET_CLIENT_SUGAR.append_row(data_to_write)

        logger.info(f"Sugar reading saved: {reading.value} mg/dL for user {reading.user_id}")
        return {"message": "Sugar reading saved successfully!", "value": reading.value}

    except Exception as e:
        logger.error(f"Error saving sugar reading to Google Sheet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save sugar reading: {e}")


# API Endpoint - POST Request for water intake
@app.post("/api/water-intake")
async def create_water_intake(water: WaterIntake):
    if SHEET_CLIENT_WATER is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        created_at = datetime.now().isoformat()
        data_to_write = [
            water.timestamp,
            water.user_id,
            water.amount,
            water.unit,
            water.source or "",
            created_at
        ]

        SHEET_CLIENT_WATER.append_row(data_to_write)

        logger.info(f"Water intake saved: {water.amount} {water.unit} for user {water.user_id}")
        return {"message": "Water intake saved successfully!", "amount": water.amount, "unit": water.unit}

    except Exception as e:
        logger.error(f"Error saving water intake to Google Sheet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save water intake: {e}")


# API Endpoint - POST Request for exercise
@app.post("/api/exercise")
async def create_exercise(exercise: Exercise):
    if SHEET_CLIENT_EXERCISE is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        created_at = datetime.now().isoformat()
        data_to_write = [
            exercise.timestamp,
            exercise.user_id,
            exercise.exercise_type,
            exercise.duration_minutes,
            exercise.intensity,
            exercise.notes or "",
            created_at
        ]

        SHEET_CLIENT_EXERCISE.append_row(data_to_write)

        logger.info(
            f"Exercise saved: {exercise.exercise_type} ({exercise.duration_minutes} min) for user {exercise.user_id}")
        return {"message": "Exercise saved successfully!", "type": exercise.exercise_type,
                "duration": exercise.duration_minutes}

    except Exception as e:
        logger.error(f"Error saving exercise to Google Sheet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save exercise: {e}")


# API Endpoint - GET Request for recent readings
@app.get("/api/readings")
async def get_recent_readings(
        user_id: str = Query("user_001", description="User ID"),
        limit: int = Query(5, description="Number of recent readings to fetch")
):
    if SHEET_CLIENT_SUGAR is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        all_readings = SHEET_CLIENT_SUGAR.get_all_records()

        # Filter by user_id and sort by timestamp (most recent first)
        user_readings = [r for r in all_readings if r.get('user_id') == user_id and r.get('value')]

        # Sort by timestamp descending
        user_readings.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Limit results
        recent_readings = user_readings[:limit]

        # Format for frontend
        formatted_readings = []
        for reading in recent_readings:
            try:
                formatted_readings.append({
                    "value": float(reading['value']),
                    "timestamp": reading['timestamp'],
                    "meal_context": reading.get('meal_context', '')
                })
            except (ValueError, TypeError):
                continue  # Skip invalid readings

        return formatted_readings

    except Exception as e:
        logger.error(f"Error fetching readings from Google Sheet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch readings: {e}")


# API Endpoint - GET Request for dashboard data
@app.get("/api/dashboard-data")
async def get_dashboard_data(user_id: str = Query("user_001", description="User ID")):
    try:
        dashboard_data = {
            "currentReading": None,
            "todaysGoals": {
                "readings": {"current": 0, "target": 4, "unit": ""},
                "water": {"current": 0, "target": 8, "unit": "cups"},
                "exercise": {"current": 0, "target": 30, "unit": "min"}
            },
            "chartData": {
                "sugarTrend": {"labels": [], "data": []},
                "weeklyOverview": {"data": [85, 12, 3]}
            }
        }

        today = datetime.now().date()
        today_str = today.isoformat()

        # Get current/latest reading
        if SHEET_CLIENT_SUGAR:
            try:
                sugar_readings = SHEET_CLIENT_SUGAR.get_all_records()
                user_readings = [r for r in sugar_readings if r.get('user_id') == user_id and r.get('value')]

                if user_readings:
                    # Sort by timestamp and get the most recent
                    user_readings.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    latest_reading = user_readings[0]
                    dashboard_data["currentReading"] = {
                        "value": float(latest_reading['value']),
                        "timestamp": latest_reading['timestamp']
                    }

                    # Count today's readings
                    todays_readings = [r for r in user_readings if r.get('timestamp', '').startswith(today_str)]
                    dashboard_data["todaysGoals"]["readings"]["current"] = len(todays_readings)

                    # Prepare chart data (last 7 days)
                    chart_data = get_chart_data(user_readings)
                    dashboard_data["chartData"]["sugarTrend"] = chart_data

            except Exception as e:
                logger.error(f"Error processing sugar readings: {e}")

        # Get today's water intake
        if SHEET_CLIENT_WATER:
            try:
                water_records = SHEET_CLIENT_WATER.get_all_records()
                todays_water = [w for w in water_records
                                if w.get('user_id') == user_id and w.get('timestamp', '').startswith(today_str)]

                total_cups = 0
                for water in todays_water:
                    try:
                        amount = float(water.get('amount', 0))
                        unit = water.get('unit', 'cups')

                        # Convert to cups for consistency
                        if unit == 'cups':
                            total_cups += amount
                        elif unit == 'ml':
                            total_cups += amount / 240  # Rough conversion
                        elif unit == 'liters':
                            total_cups += amount * 4.17  # Rough conversion
                        elif unit == 'fl-oz':
                            total_cups += amount / 8
                        elif unit == 'bottles':
                            total_cups += amount * 2.11  # 16.9 fl oz bottles

                    except (ValueError, TypeError):
                        continue

                dashboard_data["todaysGoals"]["water"]["current"] = round(total_cups, 1)

            except Exception as e:
                logger.error(f"Error processing water intake: {e}")

        # Get today's exercise
        if SHEET_CLIENT_EXERCISE:
            try:
                exercise_records = SHEET_CLIENT_EXERCISE.get_all_records()
                todays_exercise = [e for e in exercise_records
                                   if e.get('user_id') == user_id and e.get('timestamp', '').startswith(today_str)]

                total_minutes = sum(int(e.get('duration_minutes', 0)) for e in todays_exercise
                                    if e.get('duration_minutes', '').isdigit())

                dashboard_data["todaysGoals"]["exercise"]["current"] = total_minutes

            except Exception as e:
                logger.error(f"Error processing exercise data: {e}")

        return dashboard_data

    except Exception as e:
        logger.error(f"Error generating dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {e}")


# Helper function for chart data
def get_chart_data(readings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate chart data from readings for the last 7 days"""
    try:
        # Get last 7 days
        today = datetime.now().date()
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]

        labels = []
        data = []

        for day in days:
            day_str = day.isoformat()
            day_readings = [r for r in readings if r.get('timestamp', '').startswith(day_str)]

            if day == today:
                labels.append("Today")
            else:
                labels.append(day.strftime("%b %d"))

            if day_readings:
                # Average readings for the day
                avg_reading = sum(float(r['value']) for r in day_readings if r.get('value')) / len(day_readings)
                data.append(round(avg_reading, 1))
            else:
                data.append(None)

        return {"labels": labels, "data": data}

    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        return {"labels": [], "data": []}


@app.get("/")
async def read_root():
    return {"message": "Diabetes Diet Recommender API is running!"}
