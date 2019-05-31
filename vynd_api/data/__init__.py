import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

PASSWORD: str = os.environ.get('DATABASE_PASSWORD')
CLIENT: MongoClient = MongoClient(f"mongodb+srv://vynduser:{PASSWORD}@vyndcluster-8swzc.mongodb.net/test?retryWrites=true")
