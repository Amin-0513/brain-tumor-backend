from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"

client = MongoClient(MONGO_URL)
db = client["braintumor"]
db2 = client["federated_db"]
users_collection = db["user"]
rolecolection=db["roles"]
analysis_collection = db["analysis"]
fl_data_collection = db2["uploaded_files"]
analysis_collection= db["analysis"]