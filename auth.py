from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

# ======================
# Config
# ======================
SECRET_KEY = "SECRET_KEY_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Use argon2 instead of bcrypt to avoid 72-byte password limit
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ======================
# Password Utilities
# ======================
def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.
    Handles passwords of any length safely.
    """
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hashed password.
    """
    return pwd_context.verify(password, hashed_password)

# ======================
# JWT Utilities
# ======================
def create_access_token(data: dict) -> str:
    """
    Create a JWT token with an expiration time.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ======================
# Example usage
# ======================
if __name__ == "__main__":
    # Test password hashing
    password = "my_very_long_password_that_is_definitely_over_72_characters_" * 2
    hashed = hash_password(password)
    print("Hashed password:", hashed)

    # Test verification
    assert verify_password(password, hashed) == True
    print("Password verified successfully!")

    # Test JWT creation
    token = create_access_token({"sub": "user@example.com"})
    print("JWT token:", token)
