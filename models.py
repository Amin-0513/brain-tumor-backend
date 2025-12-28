from pydantic import BaseModel, EmailStr
from typing import Optional

# Model for login (only email & password)
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Model for full user registration or DB (optional)
from datetime import date
class UserCreate(BaseModel):
    firstName: str
    lastName: str
    gender: str
    phoneNo: int
    dob: date
    address: str
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id:str
    firstName: str
    lastName: str
    gender: str
    phoneNo: int
    dob: date
    address: str
    username: str
    email: EmailStr


class TumorPredictionRequest(BaseModel):
    id:str
    image_base64:str
    prediction: str
    xai_image_base64:str
    report:str

class fLData(BaseModel):
        id:str
        username: str
        filepath: str
        accuracy: float
        status: str
       
    
    
class ImageUpload(BaseModel):
    image_base64: str
   
    # xai_image_base64:str         


class UserUpdate(BaseModel):
    firstName: str
    lastName: str
    gender: str
    phoneNo: int
    dob: str
    address: str
    username: str
    email: EmailStr
    password: Optional[str] = None   # ✅ OPTIONAL




class roles(BaseModel):
    rolename:str
