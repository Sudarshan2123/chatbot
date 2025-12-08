from src.config.configuration import ConfigurationManager
from src.components.token import Token
from src.logging import logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import urllib,requests
class TextToSpeach:
    def __init__(self):
        try:
            config_obj = ConfigurationManager()
            self.config = config_obj.get_base_config()
            self.token = Token(self.config)
        except Exception as e:
            logger.error(f"Error initializing Login class: {e}")
            raise

    def Text_to_speech_process(self,data,access_token):
        logger.info("Entering the Text_to_speech_process function")
        try:
            url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.config.API_KEY}"

            requestBody = {
                "input": {"text": urllib.parse.unquote(data.text)},
                "voice": {"languageCode": urllib.parse.unquote(data.language), "ssmlGender": "NEUTRAL"},
                "audioConfig": {"audioEncoding": "MP3"}
            }
            tok_data=self.token.validate_access_token(access_token)
            if tok_data:
                response = requests.post(url, json=requestBody, headers={"Content-Type": "application/json"})
                if response.status_code == 200:
                    data = response.json()
                    audio_content = data.get("audioContent")
                    if audio_content:
                        new_access_token=self.token.create_update_token(tok_data)
                        response = JSONResponse(content={"audioContent": audio_content})
                        response.headers['Authorization'] = f"Bearer {new_access_token}"
                        return response
                    else:
                        raise HTTPException(status_code=400, detail="No audio content received.")
                else:
                    raise HTTPException(status_code=response.status_code, detail=response.json())
            else:
                raise HTTPException(status_code=400, detail="Invalid Token")
        
        except Exception as e:
            print(f"Error in converting Text to speech {e}")
            return {'status': 'error', 'message': str(e)}

   
        
