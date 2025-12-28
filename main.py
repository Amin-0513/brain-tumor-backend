from fastapi import FastAPI, HTTPException
import requests
from database import users_collection, analysis_collection, fl_data_collection
from models import UserLogin, UserCreate, ImageUpload, UserResponse, UserUpdate ,TumorPredictionRequest, fLData
from auth import verify_password, create_access_token, hash_password
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import base64
import uuid
import os
from tumor_prediction import TumorPrediction
from XAI import explain_with_lime , display_lime_explanation, predict_and_display

from bson import ObjectId
from PIL import Image
import io

app = FastAPI()

class_names = ["glioma", "meningioma", "notumor", "pituitary"]



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # MUST include OPTIONS
    allow_headers=["*"],   # MUST include Content-Type
)


tumor_model = TumorPrediction()



@app.get("/")
def root():
    return {
        "message": "Welcome to Brain Tumor Detection API Server",
        "status": "Server is running"
    }

@app.post("/login")
def login(user: UserLogin):
    print("Login attempt for:", user.email)
    
    # Query by email
    db_user = users_collection.find_one({"email": user.email})
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify hashed password
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create JWT token
    token = create_access_token({
        "user_id": str(db_user["_id"]),
        "email": db_user["email"]
    })
    
    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "SZABMU",
        "user_name":db_user["username"],
        "user_id": str(db_user["_id"]),
    }

@app.post("/insertuser")
def insertuser(user: UserCreate):
    # ✅ Convert date → datetime
    dob_datetime = datetime.combine(user.dob, datetime.min.time())

    users_collection.insert_one({
        "firstName": user.firstName,
        "lastName": user.lastName,
        "gender": user.gender,
        "phoneNo": user.phoneNo,
        "dob": dob_datetime,   # ✅ FIX HERE
        "address": user.address,
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
    })

    return {
        "message": "User Created Successfully"
    }

@app.get("/users", response_model=list[UserResponse])
def get_all_users():
    users = []

    for user in users_collection.find():
        users.append({
            "id": str(user["_id"]),   # ✅ CRITICAL
            "firstName": user["firstName"],
            "lastName": user["lastName"],
            "gender": user["gender"],
            "phoneNo": user["phoneNo"],
            "dob": user["dob"],
            "address": user["address"],
            "username": user["username"],
            "email": user["email"],
        })

    return users

@app.get("/allanalysis", response_model=list[TumorPredictionRequest])
def get_all_users():
    analysis = []

    for user in analysis_collection.find():
        analysis.append({
            "id": str(user["_id"]),   # ✅ CRITICAL
            "image_base64": user["image_base64"],
            "prediction": user["prediction"],
            "xai_image_base64": user["xai_imagebase64"],
            "report": user["report"],
        })

    return analysis

@app.get("/allfl", response_model=list[fLData])
def get_all_fl_data():
    fl_data = []

    for user in fl_data_collection.find():
        fl_data.append({
            "id": str(user["_id"]),   # ✅ CRITICAL
            "username": user["username"],
            "filepath": user["filepath"],
            "accuracy": user["accuracy"],
            "status": user["status"],
        })
    return fl_data


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")

    result = users_collection.delete_one({"_id": ObjectId(user_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "message": "User deleted successfully"
    }



@app.put("/users/{user_id}")
def update_user(user_id: str, user: UserUpdate):
    update_data = {
        "firstName": user.firstName,
        "lastName": user.lastName,
        "gender": user.gender,
        "phoneNo": user.phoneNo,
        "dob": user.dob,
        "address": user.address,
        "username": user.username,
        "email": user.email,
    }

    # 🔐 Only update password if provided
    if user.password:
        update_data["password"] = hash_password(user.password)

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User updated successfully"}


@app.post("/upload-image")
def upload_image(data: ImageUpload):
    # 1️⃣ Validate base64
    try:
        if "base64," in data.image_base64:
            data.image_base64 = data.image_base64.split("base64,")[1]

        image_bytes = base64.b64decode(data.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # 2️⃣ Save image temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, temp_filename)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(temp_path)
    payload = {
        "image_base64": data.image_base64,
        "prediction": "glioma"
    }

    response = requests.post(
        "http://127.0.0.1:5000/generate-report",
        json=payload
    )



    # 3️⃣ Run tumor prediction
    try:
        prediction = tumor_model.predict(temp_path)
        print("Step 1: Making prediction...")
        predicted_class, confidence, original_image = predict_and_display(temp_path)
        
        # Step 2: Generate LIME explanation
        print("\\nStep 2: Generating LIME explanation...")
        explanation, image_np = explain_with_lime(temp_path)
        
        # Step 3: Display LIME explanation
        print("\\nStep 3: Displaying LIME explanation...")
        base_image=display_lime_explanation(explanation, image_np, predicted_class, class_names)
        print(base_image)
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # 4️⃣ Delete temp image
    os.remove(temp_path)

    # 5️⃣ Save to MongoDB
    doc = {
        "image_base64": data.image_base64,
        "prediction": prediction,
        "xai_imagebase64":base_image,
        "report":response.json().get("report"),
        "created_at": datetime.utcnow()
    }
    result = analysis_collection.insert_one(doc)

    return {
        "message": "Image uploaded & analyzed successfully",
        "prediction": prediction,
        "xai_imagebase64":base_image,
        "report":response.json().get("report"),
        "image_id": str(result.inserted_id)
    }

@app.post("/upload-image2")
def upload_image2(data: ImageUpload):
    # 1️⃣ Validate base64
    try:
        if "base64," in data.image_base64:
            data.image_base64 = data.image_base64.split("base64,")[1]

        image_bytes = base64.b64decode(data.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # 2️⃣ Save image temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, temp_filename)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(temp_path)
    payload = {
        "image_base64": data.image_base64,
        "prediction": "glioma"
    }

    # response = requests.post(
    #     "http://127.0.0.1:5000/generate-report",
    #     json=payload
    # )



    # 3️⃣ Run tumor prediction
    try:
        prediction = tumor_model.predict(temp_path)
        print("Step 1: Making prediction...")
        predicted_class, confidence, original_image = predict_and_display(temp_path)
        
        # Step 2: Generate LIME explanation
        print("\\nStep 2: Generating LIME explanation...")
        explanation, image_np = explain_with_lime(temp_path)
        
        # Step 3: Display LIME explanation
        print("\\nStep 3: Displaying LIME explanation...")
        base_image=display_lime_explanation(explanation, image_np, predicted_class, class_names)
        print(base_image)
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # 4️⃣ Delete temp image
    os.remove(temp_path)

    # 5️⃣ Save to MongoDB
    doc = {
        "image_base64": data.image_base64,
        "prediction": prediction,
        "xai_imagebase64":base_image,
        "report":"hello",
        "created_at": datetime.utcnow()
    }
    result = analysis_collection.insert_one(doc)

    return {
        "message": "Image uploaded & analyzed successfully",
        "prediction": prediction,
        "xai_imagebase64":base_image,
        "report":"hi",
        "image_id": str(result.inserted_id)
    }
