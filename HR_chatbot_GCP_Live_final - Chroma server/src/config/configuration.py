from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity import (Base_Config)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)


    

    def get_base_config(self) -> Base_Config:
        config = self.config.config

        base_config = Base_Config(
            MONGODB_URI=config.MONGODB_URI,
            DB_NAME=config.DB_NAME,
            HISTORY_COLLECTION_NAME=config.HISTORY_COLLECTION_NAME,
            collection_user=config.collection_user,
            HISTORY_COLLECTION_Logs=config.HISTORY_COLLECTION_Logs,
            SECRET_KEY=config.SECRET_KEY,
            ALGORITHM=config.ALGORITHM,
            ACCESS_TOKEN_EXPIRE_MINUTES=config.ACCESS_TOKEN_EXPIRE_MINUTES,
            CIPHER_KEY = config.CIPHER_KEY,
           # DB_PATH = config.DB_PATH,
            CHROMA_PATH=config.CHROMA_PATH,
            RAG_MODEL=config.RAG_MODEL,
            EMBEDD_MODEL=config.EMBEDD_MODEL,
            API_KEY=config.API_KEY,
            CHROMA_COLLECTION=config.CHROMA_COLLECTION,
            CHROMA_HOST=config.CHROMA_HOST,
            CHROMA_PORT=config.CHROMA_PORT,
           # SQLLITE_CONNECTION_STRING=config.SQLLITE_CONNECTION_STRING,
            MAX_POOL_SIZE=config.maxPoolSize
            
        )

        return base_config
    





    
