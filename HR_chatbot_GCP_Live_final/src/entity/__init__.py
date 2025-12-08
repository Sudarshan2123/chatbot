from pydantic import BaseModel,Field
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


class OTP_generator_request(BaseModel):
    Phone_number: str
    class Config:
        extra = 'forbid'


class ChatRegister(BaseModel):
    employee_code: str
    firm_id: str
    class Config:
        extra = 'forbid'


class ChatRequest(BaseModel):
    input: str
    class Config:
        extra = 'forbid'
class ChatRequest2(BaseModel):
    input: str
    lang:str
    class Config:
        extra = 'forbid'


class ChatHistoryClear(BaseModel):
    access_token: str
    class Config:
        extra = 'forbid'

class TranslationRequest(BaseModel):
    text: str
    target_language: str
    # access_token:str
    class Config:
        extra = 'forbid'

class TextToSpeechRequest(BaseModel):
    text: str
    language: str
    # access_token:str
    class Config:
        extra = 'forbid'

class EncryptedLoginData(BaseModel):
    userName: str
    password: str
    class Config:
        extra = 'forbid'



class Base_Config(BaseModel):
    MONGODB_URI: str
    DB_NAME: str
    HISTORY_COLLECTION_NAME: str
    HISTORY_COLLECTION_Logs:str
    collection_user:str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    CIPHER_KEY: bytes
    #DB_PATH : str
    CHROMA_PATH: str
    RAG_MODEL: str
    EMBEDD_MODEL: str
    API_KEY: str
    CHROMA_COLLECTION: str
    #SQLLITE_CONNECTION_STRING: str
    MAX_POOL_SIZE: int


class API_Config(BaseModel):
    API_Key: str


class User_Master(Base):
    __tablename__ = 'User_Master'

    employee_code = Column(Integer, primary_key=True)
    name = Column(String)
    token = Column(String)
    session_id = Column(String)

# from langchain_core.pydantic_v1 import BaseModel, Field

class AnalysisQueryModel(BaseModel):
    analysis: str = Field(description="analysis of the query")
    query: str = Field(description="query")

