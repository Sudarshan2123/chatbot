from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity import (AgentState, Base_Config)

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
            CSRF_SECRET = config.CSRF_SECRET,
           # DB_PATH = config.DB_PATH,
            CHROMA_PATH=config.CHROMA_PATH,
            RAG_MODEL=config.RAG_MODEL,
            EMBEDD_MODEL=config.EMBEDD_MODEL,
            API_KEY=config.API_KEY,
            CHROMA_COLLECTION=config.CHROMA_COLLECTION,
           # SQLLITE_CONNECTION_STRING=config.SQLLITE_CONNECTION_STRING,
            MAX_POOL_SIZE=config.maxPoolSize,
            REDIS_USERNAME=config.REDIS_USERNAME,
            REDIS_PASSWORD=config.REDIS_PASSWORD,
            REDIS_HOST= config.REDIS_HOST,
            REDIS_PORT= config.REDIS_PORT,
            REDIS_DB= config.REDIS_DB,
            CACHE_TTL= config.CACHE_TTL,
            ORACLE_HOST=config.ORACLE_HOST,
            ORACLE_PORT=config.ORACLE_PORT,
            ORACLE_DBNAME=config.ORACLE_DBNAME,
            ORACLE_USER=config.ORACLE_USER,
            ORACLE_SCHEMA=config.ORACLE_SCHEMA,
            ORACLE_PASSWORD=config.ORACLE_PASSWORD 
        )

        return base_config
    
    def AgentState(self,default_connection_data,user_input):
         conn_data=default_connection_data
         agent_state: AgentState = {
            "input": user_input,
            "available_tables": conn_data['table_metadata'],
            "loaded_data": {},
            "selected_tables": None,
            "decision": None,
            "response": None,
            "analysis_context": None
        }
         return agent_state
    
    # def get_token():
    #     CSRF_SECRET = config.CSRF_SECRET,
        
    





    
