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
import asyncio
import smtplib
import ssl
from email.message import EmailMessage
from urllib.parse import quote
import pandas as pd

# --- Load environment variables from .env file ---
load_dotenv()

# --- Email + Reset Config ---
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587")) 
SMTP_USERNAME = os.environ.get("SMTP_USERNAME") 
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD") 
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USERNAME)
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://127.0.0.1:5500")
RESET_TOKEN_TTL_MIN = int(os.environ.get("RESET_TOKEN_TTL_MIN", "15"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for data
class Feedback(BaseModel):
    user_id: Optional[str] = None
    rating: int  # 1 to 5
    liked_food: Optional[bool] = True # True if they liked the meal, False if they disliked specific items
    disliked_items: Optional[str] = None # "Omena, Kales"
    feedback_text: Optional[str] = None # Reason
    contact_email: Optional[str] = None


class UserSignup(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    identifier: str 
    password: str


class ResetRequest(BaseModel):
    identifier: str 


class ResetPassword(BaseModel):
    identifier: str
    token: str
    new_password: str


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


def build_reset_link(identifier: str, token: str) -> str:
    base = FRONTEND_BASE_URL.rstrip("/")
    return f"{base}/Frontend/reset_password.html?identifier={quote(identifier)}&token={quote(token)}"


def _send_email_sync(to_email: str, subject: str, body: str):
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)

            msg = EmailMessage()
            msg["From"] = SMTP_FROM
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.set_content(body)

            server.send_message(msg)

        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")


async def send_reset_email(to_email: str, identifier: str, token: str):
    link = build_reset_link(identifier, token)
    subject = "NutriApp – Reset Your Password"
    body = (
        "Hello,\n\n"
        "We received a request to reset your NutriApp password.\n"
        f"Click the link below (valid for {RESET_TOKEN_TTL_MIN} minutes):\n\n"
        f"{link}\n\n"
        "If you didn’t request this, just ignore this email.\n\n"
        "— NutriApp Team"
    )
    await asyncio.to_thread(_send_email_sync, to_email, subject, body)


# --- GOOGLE SHEETS CONFIGURATION ---
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH")
GOOGLE_CREDENTIALS_JSON_STRING = os.environ.get("GOOGLE_CREDENTIALS_JSON")

if GOOGLE_SHEET_ID is None:
    logging.error("Environment variable GOOGLE_SHEET_ID is not set.")
    sys.exit(1)


# --- GOOGLE SHEETS INITIALIZATION ---
def initialize_worksheet(worksheet_name, headers):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

        if GOOGLE_CREDENTIALS_JSON_STRING:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                json.loads(GOOGLE_CREDENTIALS_JSON_STRING), scope
            )
        elif GOOGLE_CREDENTIALS_PATH:
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                GOOGLE_CREDENTIALS_PATH, scope
            )
        else:
            raise ValueError("No Google Sheets credentials found.")

        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)

        try:
            sheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(worksheet_name, rows="100", cols="20")

        if not sheet.row_values(1):
            sheet.append_row(headers)

        return sheet

    except Exception as e:
        logger.error(f"Error initializing Google Sheet worksheet '{worksheet_name}': {e}")
        return None


# Initialize Sheets
try:
    SHEET_CLIENT_FEEDBACK = initialize_worksheet("Feedback", ['timestamp', 'user_id', 'rating', 'feedback_text', 'contact_email'])
    SHEET_CLIENT_DISLIKES = initialize_worksheet("FoodDislikes", ['user_id', 'food_name', 'reason', 'timestamp']) # NEW SHEET
    SHEET_CLIENT_USERS = initialize_worksheet("Users", ['user_id', 'username', 'email', 'hashed_password'])
    SHEET_CLIENT_SUGAR = initialize_worksheet("SugarReadings", ['timestamp', 'user_id', 'value', 'meal_context', 'created_at'])
    SHEET_CLIENT_WATER = initialize_worksheet("WaterIntake", ['timestamp', 'user_id', 'amount', 'unit', 'source', 'created_at'])
    SHEET_CLIENT_EXERCISE = initialize_worksheet("Exercise", ['timestamp', 'user_id', 'exercise_type', 'duration_minutes', 'intensity', 'notes', 'created_at'])
    SHEET_CLIENT_RESETS = initialize_worksheet("PasswordResets", ['identifier', 'token', 'expiry'])
except Exception:
    SHEET_CLIENT_FEEDBACK = None
    SHEET_CLIENT_DISLIKES = None
    SHEET_CLIENT_USERS = None
    SHEET_CLIENT_SUGAR = None
    SHEET_CLIENT_WATER = None
    SHEET_CLIENT_EXERCISE = None
    SHEET_CLIENT_RESETS = None

# --- Nutrient Lookup Class ---
class NutrientLookup:
    def __init__(self, csv_path):
        self.data = None
        try:
            self.data = pd.read_csv(csv_path)
            self.data['Name_Key'] = self.data['food_name'].astype(str).str.strip().str.lower()
        except Exception as e:
            logger.error(f"Failed to load nutrient data: {e}")

    def get_details(self, food_name):
        if self.data is None: return {}
        clean_name = str(food_name).strip().lower()
        row = self.data[self.data['Name_Key'] == clean_name]
        if row.empty:
             row = self.data[self.data['Name_Key'].str.contains(clean_name, regex=False, na=False)]

        if not row.empty:
            return {
                "calories": int(row.iloc[0].get('calories', 0)),
                "carbs": round(float(row.iloc[0].get('carbohydrates_g', 0)), 3),
                "fiber": round(float(row.iloc[0].get('fiber_g', 0)), 3),
                "protein": round(float(row.iloc[0].get('protein_g', 0)), 3),
                "fat": round(float(row.iloc[0].get('fat_g', 0)), 3)
            }
        return {"calories": "-", "carbs": "-", "fiber": "-"}

# Path for Recommender
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

app = FastAPI(title="Diet Recommender API")

# CORS
origins = ["*"] # simplified for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Recommender
try:
    food_data_full_path = os.path.join(current_script_dir, "data", "KenyaFoodCompositionsClean.csv")
    models_full_dir = os.path.join(current_script_dir, "models")

    recommender_instance = DiabetesDietRecommender(
        food_data_path=food_data_full_path,
        models_dir=models_full_dir
    )
    nutrient_lookup = NutrientLookup(food_data_full_path)
except Exception as e:
    recommender_instance = None
    nutrient_lookup = None

# --- Helper: Get User Dislikes ---
def get_user_dislikes(user_id: str) -> List[str]:
    """Fetches disliked foods from Google Sheets for a specific user."""
    if not user_id or SHEET_CLIENT_DISLIKES is None:
        return []
    try:
        # Get all records
        records = SHEET_CLIENT_DISLIKES.get_all_records()
        # Filter for this user and return just the food names
        dislikes = [r['food_name'] for r in records if str(r.get('user_id')) == user_id]
        logger.info(f"Loaded dislikes for {user_id}: {dislikes}")
        return dislikes
    except Exception as e:
        logger.error(f"Error fetching dislikes: {e}")
        return []

# --- Helper: Get Stats ---
def get_today_stats(user_id: str):
    stats = {"water_cups": 0, "exercise_mins": 0}
    try:
        today = datetime.now().date()
        # Water
        if SHEET_CLIENT_WATER:
            records = SHEET_CLIENT_WATER.get_all_records()
            daily_water = 0
            for r in records:
                if str(r.get('user_id')) == user_id:
                    ts = r.get('timestamp') or r.get('created_at')
                    if ts:
                        try:
                            if 'T' in ts: dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            else: dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                            if dt.date() == today:
                                amt = float(r.get('amount', 0))
                                daily_water += amt # Simplified unit logic for brevity
                        except: continue
            stats['water_cups'] = round(daily_water, 1)

        # Exercise
        if SHEET_CLIENT_EXERCISE:
            records = SHEET_CLIENT_EXERCISE.get_all_records()
            daily_exercise = 0
            for r in records:
                if str(r.get('user_id')) == user_id:
                    ts = r.get('timestamp') or r.get('created_at')
                    if ts:
                        try:
                            if 'T' in ts: dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            else: dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                            if dt.date() == today:
                                daily_exercise += int(r.get('duration_minutes', 0))
                        except: continue
            stats['exercise_mins'] = daily_exercise
            
    except Exception as e:
        logger.error(f"Error stats: {e}")
    return stats


# --- UPDATED ENDPOINT: /recommend_meal ---
@app.get("/recommend_meal", response_model=Dict[str, Any])
async def recommend_meal(
        user_id: Optional[str] = Query(None),
        fbs_level: Optional[float] = Query(None),
        rbs_level: Optional[float] = Query(None),
        meal_type: str = Query("lunch"),
        num_alternatives_per_slot: int = Query(1, ge=1)
):
    if recommender_instance is None:
        raise HTTPException(status_code=500, detail="Recommender not initialized.")

    try:
        # 1. Fetch Stats
        daily_stats = get_today_stats(user_id) if user_id else {"water_cups": 0, "exercise_mins": 0}

        # 2. Fetch User Dislikes (NEW)
        user_dislikes = get_user_dislikes(user_id) if user_id else []

        # 3. Build Profile
        user_profile = {
            'fbs': fbs_level,
            'rbs': rbs_level,
            'water_cups': daily_stats['water_cups'],
            'exercise_mins': daily_stats['exercise_mins']
        }

        # 4. Generate Insight & Meal (PASSING DISLIKES)
        tip_message = recommender_instance.generate_insight(user_profile)
        
        raw_recs = recommender_instance.recommend_diet(
            user_profile,
            meal_type=meal_type,
            num_alternatives=num_alternatives_per_slot,
            disliked_foods=user_dislikes # <--- Passing the dislikes here
        )
        
        # 5. Format Output
        enhanced_results = []
        for rec_string in raw_recs:
            if ":" in rec_string:
                category, food_list_str = rec_string.split(":", 1)
                
                if num_alternatives_per_slot == 1:
                    foods = [food_list_str.strip()] 
                else:
                    foods = [f.strip() for f in food_list_str.split(",")]

                for food in foods:
                    details = nutrient_lookup.get_details(food)
                    enhanced_results.append({
                        "name": food,
                        "category": category.strip(),
                        "nutrients": details
                    })
            else:
                enhanced_results.append({
                    "name": rec_string,
                    "category": "Info",
                    "nutrients": {"calories": "-", "carbs": "-", "fiber": "-"}
                })
        
        return {
            "tip": tip_message,
            "recommendations": enhanced_results
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_feedback")
async def submit_feedback(feedback: Feedback):
    if SHEET_CLIENT_FEEDBACK is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed.")
    try:
        timestamp = datetime.now().isoformat()
        
        # 1. Save General Feedback
        SHEET_CLIENT_FEEDBACK.append_row([
            timestamp, 
            feedback.user_id or "Anonymous",
            feedback.rating, 
            feedback.feedback_text, 
            feedback.contact_email
        ])

        # 2. Handle DISLIKED Items (The Feedback Loop)
        if not feedback.liked_food and feedback.disliked_items and feedback.user_id:
            # Assume items are comma separated: "Omena, Kales"
            items = [x.strip() for x in feedback.disliked_items.split(',') if x.strip()]
            
            if SHEET_CLIENT_DISLIKES:
                for item in items:
                    SHEET_CLIENT_DISLIKES.append_row([
                        feedback.user_id,
                        item,
                        feedback.feedback_text or "User Dislike",
                        timestamp
                    ])
                logger.info(f"Registered dislikes for user {feedback.user_id}: {items}")

        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")

# ... (Rest of your existing endpoints: signup, login, reset, etc. remain unchanged) ...
# Just ensure you include the dashboard endpoint I gave earlier if you haven't already.
# I will cut the rest to save space, but DO NOT DELETE your existing login/signup code!


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

        if any(u['email'] == user.email for u in all_users):
            raise HTTPException(status_code=400, detail="Email already registered.")

        # Hash the password
        hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Generate a unique user ID
        user_id = str(uuid.uuid4())

        # Prepare data for Google Sheet (matching the new order)
        data_to_write = [
            user_id,
            user.username,
            user.email,
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
        # Fetch all users from Google Sheets
        all_users = SHEET_CLIENT_USERS.get_all_records()

        # Find the user by username or email
        user_in_db = next(
            (u for u in all_users if u['username'] == user.identifier or u['email'] == user.identifier),
            None
        )

        if not user_in_db:
            raise HTTPException(status_code=400, detail="Invalid username or password.")

        # Verify password
        if bcrypt.checkpw(user.password.encode('utf-8'), user_in_db['hashed_password'].encode('utf-8')):
            logger.info(f"User logged in: {user.identifier}")
            return {
                "message": "Login successful!",
                "user_id": user_in_db.get("user_id"),
                "username": user_in_db.get("username"),
                "email": user_in_db.get("email")
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid username or password.")

    except Exception as e:
        logger.error(f"Error during user login: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during login.")


@app.post("/request-password-reset")
async def request_password_reset(req: ResetRequest):
    if SHEET_CLIENT_USERS is None or SHEET_CLIENT_RESETS is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        identifier = req.identifier.strip().lower()

        # Look up user
        all_users = SHEET_CLIENT_USERS.get_all_records()
        user = next(
            (u for u in all_users if str(u['username']).lower() == identifier or str(u['email']).lower() == identifier),
            None
        )

        # Always respond the same (don’t leak whether the user exists)
        generic_response = {"message": "Password reset link sent to your email."}

        if not user:
            logger.info(f"Password reset requested for non-existing identifier: {identifier}")
            return generic_response

        # Generate token + expiry
        token = str(uuid.uuid4())
        expiry = (datetime.utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MIN)).isoformat()

        # Save reset entry
        SHEET_CLIENT_RESETS.append_row([identifier, token, expiry])

        # Send email to the actual user’s email
        await send_reset_email(user["email"], identifier, token)

        logger.info(f"Password reset token generated and email sent for {identifier}")
        return generic_response

    except Exception as e:
        logger.error(f"Error during password reset request: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate password reset link.")


@app.post("/reset-password")
async def reset_password(data: ResetPassword):
    if SHEET_CLIENT_USERS is None or SHEET_CLIENT_RESETS is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        identifier = data.identifier.strip().lower()
        token = data.token
        new_password = data.new_password

        # Load resets
        all_resets = SHEET_CLIENT_RESETS.get_all_records()
        reset_entry = next(
            (r for r in all_resets if r['identifier'].lower() == identifier and r['token'] == token),
            None
        )
        if not reset_entry:
            raise HTTPException(status_code=400, detail="Invalid or expired token.")

        # Check expiry
        if datetime.utcnow() > datetime.fromisoformat(reset_entry['expiry']):
            raise HTTPException(status_code=400, detail="Token expired.")

        # Hash new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update user in Users sheet
        all_users = SHEET_CLIENT_USERS.get_all_records()
        for i, u in enumerate(all_users, start=2):  # row index starts at 2 (1 is header)
            if u['username'].lower() == identifier or u['email'].lower() == identifier:
                SHEET_CLIENT_USERS.update_cell(i, list(u.keys()).index("hashed_password") + 1, hashed_password)
                break

        # Remove reset entry
        for i, r in enumerate(all_resets, start=2):
            if r['identifier'].lower() == identifier and r['token'] == token:
                SHEET_CLIENT_RESETS.delete_rows(i)
                break

        logger.info(f"Password reset successful for {identifier}")
        return {"message": "Password reset successful."}

    except Exception as e:
        logger.error(f"Error during password reset: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password.")


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
        user_id: str = Query(..., description="User ID (required)"),  # <-- require user_id
        limit: int = Query(5, ge=1, le=50, description="How many to return")  # <-- sane bounds
):
    if SHEET_CLIENT_SUGAR is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed on startup.")

    try:
        all_readings = SHEET_CLIENT_SUGAR.get_all_records()

        # Filter by user_id and ensure value exists
        user_readings = [
            r for r in all_readings
            if str(r.get('user_id', '')).strip() == user_id
               and str(r.get('value', '')).strip() != ''
        ]

        # Robust timestamp parsing (prefer 'timestamp', then 'created_at')
        def parse_ts(row):
            ts = row.get('timestamp') or row.get('created_at') or ""
            try:
                # Handle potential trailing 'Z'
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return datetime.min  # push invalid timestamps to the end

        # Sort most recent first
        user_readings.sort(key=parse_ts, reverse=True)

        # Limit results
        recent_readings = user_readings[:limit]

        # Shape for frontend
        formatted_readings = []
        for r in recent_readings:
            try:
                formatted_readings.append({
                    "value": float(r.get('value')),
                    "timestamp": r.get('timestamp') or r.get('created_at') or "",
                    "meal_context": r.get('meal_context') or ""
                })
            except (ValueError, TypeError):
                # Skip any malformed row
                continue

        return formatted_readings

    except Exception as e:
        logger.exception("Error fetching readings from Google Sheet")
        raise HTTPException(status_code=500, detail=f"Failed to fetch readings: {e}")


# API Endpoint - GET Request for dashboard data
# REPLACE the existing get_dashboard_data function in api.py with this:

@app.get("/api/dashboard-data")
async def get_dashboard_data(
        user_id: str = Query(..., description="User ID (required)")
):
    try:
        # 1. Default Clean Slate
        dashboard_data = {
            "currentReading": None,
            "todaysGoals": {
                "readings": {"current": 0, "target": 4, "unit": ""},
                "water": {"current": 0, "target": 8, "unit": "cups"},
                "exercise": {"current": 0, "target": 30, "unit": "min"}
            },
            "chartData": {
                "sugarTrend": {"labels": [], "values": [], "contexts": []},
                "weeklyOverview": {"data": [0, 0, 0]} 
            }
        }

        # Get today's date (Local time)
        today = datetime.now().date()
        today_str = today.isoformat()
        
        # Debug Print
        print(f"--- DEBUG: Fetching Dashboard for User: {user_id} ---")

        # ------------------ Sugar readings & Charts ------------------
        if SHEET_CLIENT_SUGAR:
            try:
                sugar_readings = SHEET_CLIENT_SUGAR.get_all_records()
                
                # Filter for this user
                user_readings = [
                    r for r in sugar_readings
                    if str(r.get('user_id', '')).strip() == user_id
                    and str(r.get('value', '')).strip() != ''
                ]

                print(f"DEBUG: Found {len(user_readings)} total readings for this user.")

                if user_readings:
                    # Helper to safely parse dates
                    def safe_parse_date(ts_str):
                        if not ts_str: return None
                        try:
                            # Try standard ISO
                            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                        except ValueError:
                            try:
                                # Try format with space instead of T
                                return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)
                            except:
                                return None

                    # Sort by timestamp
                    user_readings.sort(key=lambda x: safe_parse_date(x.get('timestamp') or x.get('created_at')) or datetime.min, reverse=True)

                    # --- A. CALCULATE WEEKLY OVERVIEW ---
                    in_range = 0
                    high = 0
                    low = 0
                    
                    # 7 Days ago
                    cutoff = datetime.now() - timedelta(days=7)
                    
                    valid_weekly_count = 0

                    for r in user_readings:
                        ts_str = r.get('timestamp') or r.get('created_at') or ""
                        r_date = safe_parse_date(ts_str)
                        
                        if r_date and r_date >= cutoff:
                            valid_weekly_count += 1
                            try:
                                val = float(r['value'])
                                if val < 70: low += 1
                                elif val > 180: high += 1
                                else: in_range += 1
                            except:
                                continue
                    
                    print(f"DEBUG: Valid readings in last 7 days: {valid_weekly_count}")
                    print(f"DEBUG: Breakdown - High: {high}, Low: {low}, Range: {in_range}")

                    total = in_range + high + low
                    if total > 0:
                        # Calculate percentages
                        p_range = round((in_range / total) * 100)
                        p_high = round((high / total) * 100)
                        p_low = round((low / total) * 100)
                        
                        # Adjust rounding errors to ensure sum is 100
                        diff = 100 - (p_range + p_high + p_low)
                        if diff != 0:
                            p_range += diff
                            
                        dashboard_data["chartData"]["weeklyOverview"]["data"] = [p_range, p_high, p_low]
                    else:
                        # Explicitly set 0s so frontend knows it's empty
                        dashboard_data["chartData"]["weeklyOverview"]["data"] = [0, 0, 0]

                    # --- B. Latest Reading ---
                    latest_reading = user_readings[0]
                    dashboard_data["currentReading"] = {
                        "value": float(latest_reading['value']),
                        "timestamp": latest_reading.get('timestamp')
                    }

                    # --- C. Today's Count ---
                    todays_readings_count = 0
                    for r in user_readings:
                        r_date = safe_parse_date(r.get('timestamp') or r.get('created_at'))
                        if r_date and r_date.date() == today:
                            todays_readings_count += 1
                    
                    dashboard_data["todaysGoals"]["readings"]["current"] = todays_readings_count

                    # --- D. Chart Data ---
                    # Reuse the get_chart_data logic but inline for safety
                    labels = []
                    values = []
                    contexts = []
                    
                    # Reverse for the line chart (Oldest -> Newest)
                    chart_source = [r for r in user_readings if safe_parse_date(r.get('timestamp')) and safe_parse_date(r.get('timestamp')) >= cutoff][::-1]
                    
                    for r in chart_source:
                        d = safe_parse_date(r.get('timestamp'))
                        labels.append(d.strftime("%b %d %H:%M"))
                        values.append(float(r['value']))
                        contexts.append(r.get('meal_context', ''))

                    dashboard_data["chartData"]["sugarTrend"] = {
                        "labels": labels, "values": values, "contexts": contexts
                    }

            except Exception as e:
                print(f"ERROR processing sugar: {e}")

        # ------------------ Water ------------------
        if SHEET_CLIENT_WATER:
            try:
                water_records = SHEET_CLIENT_WATER.get_all_records()
                total_cups = 0
                for w in water_records:
                    if str(w.get('user_id', '')).strip() == user_id:
                         # Simple string check for today
                        ts = w.get('timestamp', '')
                        if ts.startswith(today_str):
                            try:
                                amount = float(w.get('amount', 0))
                                unit = w.get('unit', 'cups')
                                if unit == 'cups': total_cups += amount
                                elif unit == 'ml': total_cups += amount / 240
                                elif unit == 'liters': total_cups += amount * 4.17
                            except: continue
                dashboard_data["todaysGoals"]["water"]["current"] = round(total_cups, 1)
            except Exception as e:
                print(f"ERROR processing water: {e}")

        # ------------------ Exercise ------------------
        if SHEET_CLIENT_EXERCISE:
            try:
                exercise_records = SHEET_CLIENT_EXERCISE.get_all_records()
                total_minutes = 0
                for e in exercise_records:
                    if str(e.get('user_id', '')).strip() == user_id:
                        ts = e.get('timestamp', '')
                        if ts.startswith(today_str):
                            try:
                                total_minutes += int(e.get('duration_minutes', 0))
                            except: continue
                dashboard_data["todaysGoals"]["exercise"]["current"] = total_minutes
            except Exception as e:
                print(f"ERROR processing exercise: {e}")

        return dashboard_data

    except Exception as e:
        print(f"CRITICAL ERROR in dashboard-data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {e}")

@app.get("/api/Loggedsugar-readings")
async def get_sugar_readings(user_id: str, limit: int = 10):
    if SHEET_CLIENT_SUGAR is None:
        raise HTTPException(status_code=500, detail="Google Sheets connection failed.")
    
    try:
        # Get all records from the sheet
        all_records = SHEET_CLIENT_SUGAR.get_all_records()
        
        # Filter by user_id
        user_readings = [r for r in all_records if r.get('user_id') == user_id]
        
        # Sort by timestamp (most recent first)
        user_readings.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Return limited results
        return user_readings[:limit] if limit else user_readings
        
    except Exception as e:
        logger.error(f"Error fetching sugar readings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch readings: {e}")

@app.get("/")
async def read_root():
    return {"message": "Diabetes Diet Recommender API is running!"}
