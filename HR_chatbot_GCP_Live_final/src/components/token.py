
import uuid,bcrypt,os,pytz,jwt
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Optional
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from src.entity import Base_Config
from src.logging import logger
from src.exception import CustomException
from src.utils.mongo_ops import dynamic_operation
from src.entity import User_Master
class Token:
    def __init__(self, config: Base_Config):
        self.config = config
        self.client = MongoClient(config.MONGODB_URI,maxPoolSize=config.MAX_POOL_SIZE)
        self.db=self.client[config.DB_NAME]
        self.user_collection=self.db[config.collection_user]
    
    def create_access_token(self,data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(pytz.utc) + expires_delta
        else:
            expire = datetime.now(pytz.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM)
        return encoded_jwt
    
    def create_update_token(self,data:dict):
            logger.info("Entering access token validation method ")
            User_name = data["userName"]
            session_id = data["session_id"]
            access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = self.create_access_token(data={"userName": User_name,"session_id":session_id}, expires_delta=access_token_expires)
            #users_collection.update_one({"employee_code": employee_code}, {"$set": {"token": access_token}})
            #updated_users = dynamic_operation('UPDATE',self.config.SQLLITE_CONNECTION_STRING, User_Master, filter_conditions={'name': User_name}, update_values={'token':access_token , 'session_id': session_id})
            # self.data_updateSQL(User_name,access_token,session_id)
            try:
                 result=self.user_collection.update_one(
                      {"employee_code":User_name},
                      {"$set":{"token":access_token,"session_id":session_id}}
                    )
                 if result.modified_count==0:
                      logger.warning(f"User {User_name} not found during token update.")
            except Exception as e :
                 logger.error(f"Error updating token in MongoDB: {e}")
                 raise CustomException(f"Failed to update token in database: {e}")
            return access_token


    def get_user_name_from_access_token(self,access_token):
            decoded_jwt = jwt.decode(access_token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
            userName = decoded_jwt.get('userName')  # Adjust according to your token payload
            return userName  
    
    
    def validate_access_token(self,token: str) -> Optional[dict]:
        try:
            logger.info("Entering access token validation method ")
         
            decoded_jwt = jwt.decode(token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
            userName = decoded_jwt.get('userName')  # Adjust according to your token payload        
            if not userName:
                logger.error("Token is invalid does not contain required fields")
                return None

            token_record = self.user_collection.find_one({'employee_code': userName,'token': token})  
            # user = cur.fetchone()  
            if token_record:
                return decoded_jwt
            else:
                logger.error("Token is invalid does not match requeird token")
                return None
        except ExpiredSignatureError:
            # Handle expired token error
            logger.error("Token has expired.")
            return None
        except InvalidTokenError:
            # Handle invalid token error
            logger.error("Invalid token.")
            return None
        except Exception as e:
             logger.error(f"An unexpected error occurred during token validation:{e}")
             return None
            
    
    def data_update(self, users_collection, employee_code, access_token, session_id):
        users_collection.update_one(
            {"employee_code": employee_code},  # Filter criteria
            {"$set": {"token": access_token, "session_id": session_id}}  # Fields to update
        )
    
    def close_mongo_connection(self):
         if self.client:
              self.client.close()
    # def data_updateSQL(self, userName, access_token, session_id):
    #     params=(access_token, session_id, userName)
    #     query=("UPDATE User_Master SET token = ?, session_id = ? WHERE name = ?")
    #     execute_query(query,params,self.config.DB_PATH)

