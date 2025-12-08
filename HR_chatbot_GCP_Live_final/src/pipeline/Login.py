import os
from src.logging import logger
from src.components.token import Token
from src.utils.common import decode_data
from src.config.configuration import ConfigurationManager
from pymongo import MongoClient
import uuid
from datetime import timedelta
import requests
from pymongo.errors import ConnectionFailure, PyMongoError
from requests.exceptions import RequestException

class Login:
    def __init__(self):
        try:
            config_obj = ConfigurationManager()
            self.config = config_obj.get_base_config()
            self.token = Token(self.config)
            self.client = MongoClient(self.config.MONGODB_URI, maxPoolSize=self.config.MAX_POOL_SIZE)
            self.db = self.client[self.config.DB_NAME]
            self.user_collection = self.db[self.config.collection_user]  # Use the user collection from config
        except ConnectionFailure as e:
            logger.error(f"Error initializing Login class: Could not connect to MongoDB: {e}")
            raise  # Re-raise to prevent the app from running with a broken DB connection
        except PyMongoError as e:
            logger.error(f"Error initializing Login class: PyMongoError: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing Login class: Unexpected error: {e}")
            raise

    def login_user(self, userName: int, password: str) -> dict:
        try:
            logger.info("Entering the login process function")
            # Use MongoDB to find the user by userName
            user = self.user_collection.find_one({'employee_code': userName})  # Assuming 'name' field stores the username

            if user:
                url = 'https://docker.mactech.net.in:5013/ldap-service/loginthroughEmail'
                data = {"userName": userName, "password": password}
                result,flag = call_api_post(url, data)
                

                if result.status_code == 200:
                    if flag =="active":
                        session_id = str(uuid.uuid4())
                        access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
                        access_token = self.token.create_access_token(
                            data={"userName": userName, "session_id": session_id},
                            expires_delta=access_token_expires
                        )
                        # Update the user's record in MongoDB with the token and session ID
                        updated_user = self.user_collection.update_one(
                            {'employee_code': userName},  # Use 'name' to find the user
                            {"$set": {'token': access_token, 'session_id': session_id}}
                        )
                        if updated_user.modified_count:
                            logger.info(f"Token and session updated for user: {userName}")
                        else:
                            logger.warning(f"User {userName} not found during token update in login.")
                        return {"access_token": access_token, "token_type": "bearer", "status": "success","user_Status":"active"}
                    elif flag=="Inactive":
                          logger.warning(f"User {userName} is Inactive,OTP verification Required")
                          session_id = str(uuid.uuid4())
                          access_token_expires = timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
                          access_token = self.token.create_access_token(
                            data={"userName": userName, "session_id": session_id},
                            expires_delta=access_token_expires
                            )
                        # Update the user's record in MongoDB with the token and session ID
                          updated_user = self.user_collection.update_one(
                            {'employee_code': userName},  # Use 'name' to find the user
                            {"$set": {'token': access_token, 'session_id': session_id}}
                           )
                          if updated_user.modified_count:
                             logger.info(f"Token and session updated for user: {userName}")
                          else:
                             logger.warning(f"User {userName} not found during token update in login.")
                          return{"access_token": access_token, "token_type": "bearer", "status": "success","user_Status":"Inactive"}
                    else:
                        logger.warning(f"User {userName} not Found during Login ,User_status is invalid")     
                else:
                    logger.error(f"LDAP service returned an error: {result.status_code}, {result.text}")
                    return {'status': 'error', 'message': 'Authentication failed with LDAP service'}
            else:
                return {"status": "OTP verification failed"}  # Or "User not found", depending on your logic
        except PyMongoError as e:
            logger.error(f"Login failed for user {userName}: MongoDB error: {e}")
            return {'status': 'error', 'message': 'Database error occurred during login'}
        except RequestException as e:
            logger.error(f"Login failed for user {userName}: LDAP service request failed: {e}")
            return {'status': 'error', 'message': 'Failed to communicate with authentication service'}
        except Exception as e:
            logger.error(f"Login failed for user {userName}: Unexpected error: {e}")
            return {'status': 'error', 'message': 'An unexpected error occurred during login'}
        finally:
            if self.client:
                self.client.close()


import requests
import json

def call_api_post(url: str, data: dict) -> requests.Response:
    """
    Sends a POST request to the specified URL with JSON data.
    Args:
        url: The URL to send the request to.
        data: The data to send in the request body.
    Returns:
        The response from the server.
    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        response_data=response.json()
        flag=response_data.get("status")
        return response,flag
    except requests.exceptions.RequestException as e:
        logger.error(f"API call to {url} failed: {e}")
        raise  # Re-raise the exception to be handled by the caller
