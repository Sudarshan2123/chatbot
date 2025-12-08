import os
from src.logging import logger
# from src.components.Chatprocess import Chatbot_Manager
from src.components.token import Token
from src.config.configuration import ConfigurationManager
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import html
from src.components.Chatprocess import Chatbot_Manager
from src.pipeline.get_utils import get_gcp_credentials
from src.utils.common import Translate_process_chat


class Chatbot_Pipeline:
    def __init__(self):
        self.config_obj=ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        self.token = Token(self.config)
        credentials=get_gcp_credentials()
        self.Chatbot_manager = Chatbot_Manager(config=self.config,credentials=credentials)


    def main_process(self,session_id,user_input) -> dict:
        try:
           
          
            # self.Chatbot_manager.clean_orphaned_messages()
            user_id = session_id
            if not user_id:
                return {'status': 'error', 'message': 'Invalid employee code or password'}

            self.Chatbot_manager.clean_session_history(user_id)
            with_message_history=self.Chatbot_manager.build_rag_chain()
            #input_validated= self.Chatbot_manager.validate_question(user_input)
            response = with_message_history.invoke({"question":user_input }, {"configurable": {"session_id": user_id}})
 
            return {'status': 'success', 'answer': response}
        except Exception as e:
            print(f"Error in saving and fetching user query : {e}")
            response="Apologies, something went wrong while processing your request. Could you please try again? If the issue persists, feel free to reach out for assistance." 
            return {'status': 'success', 'answer': response}
        
  

    def main_chatbot(self,access_token,input_text,lang):
        tok_data=self.token.validate_access_token(access_token)
        if lang!="en-US":
            Translated_input=Translate_process_chat(input_text,"en",self.config.API_KEY)
        else:
            Translated_input=input_text
        if tok_data:
            session_id = tok_data["session_id"]
            if access_token and Translated_input:
                try:
                    response =  self.main_process(session_id, Translated_input)
                    if lang!="en-US":
                        Translated_response=Translate_process_chat(response.get('answer', ''),lang,self.config.API_KEY)
                    else:
                        Translated_response=response.get('answer', '')
                    new_access_token=self.token.create_update_token(tok_data)
                    encoded_answer = html.escape(Translated_response)
                    user_name=self.token.get_user_name_from_access_token(access_token)
                    self.Chatbot_manager.mongo_log_chat(user_name,Translated_input,Translated_response)
                    # return JSONResponse({'status': 'success','access_token':new_access_token, 'answer': encoded_answer}, status_code=200)
                    response = JSONResponse(content={'status': 'success','answer': encoded_answer}, status_code=200)
                    response.headers['Authorization'] = f"Bearer {new_access_token}"
                    return response
                finally:
                    # Ensure Chroma connection is properly closed
                    self.Chatbot_manager.close_chroma_connection()

            else:
                raise HTTPException(status_code=400, detail="Session ID and input required for chat")
        else:
            raise HTTPException(status_code=400, detail="Invalid Token")
