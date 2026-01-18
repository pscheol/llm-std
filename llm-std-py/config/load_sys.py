from dotenv import load_dotenv
import os
load_dotenv()


google_model = os.getenv("GOOGLE_AI_MODEL", "gemini-3-flash-preview")
open_ai_model = os.getenv("OPEN_AI_MODEL", 'gpt-3.5-turbo-instruct')
