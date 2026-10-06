from typing import Optional;
from pymongo import AsyncMongoClient;
from pymongo.asynchronous.database import AsyncDatabase;
from src.config import get_settings;


settings = get_settings()


class DatabaseManager:
    
    client: Optional[AsyncMongoClient] = None
    
    @classmethod
    async def connect_to_mongo(cls) -> None:
        """
        Establish a connection to the MongoDB database.
        """
        cls.client = AsyncMongoClient(
            settings.MONGO_URI,
            maxPoolSize=10,
            minPoolSize=2,
            serverSelectionTimeoutMS=5000
        )
        await cls.client.admin.command('ping')  # Test the connection
        print("Connected to MongoDB")
    
    @classmethod
    async def close_mongo_connection(cls) -> None:
        """
        Close the connection to the MongoDB database.
        """
        if cls.client:
            cls.client.close()
            print("MongoDB connection closed")
            
    def get_database(cls) -> AsyncDatabase:
        """
        Get the MongoDB database instance.
        """
        if cls.client is None:
            raise Exception("MongoDB client is not connected. Call connect_to_mongo first.")
        return cls.client[settings.MONGODB_DATABASE_NAME]
    
async def get_db_collection() -> AsyncDatabase:
    """
    Dependency function to get the MongoDB collection.
    """
    db = DatabaseManager().get_database()
    return db[settings.MONGODB_COLLECTION_NAME]