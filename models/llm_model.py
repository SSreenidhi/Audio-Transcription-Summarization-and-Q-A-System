from langchain_google_genai import GoogleGenerativeAI
import os

def load_llm_model(api_key=None, model_name="gemini-1.5-flash-latest", temperature=0.2):
    # Set the API key - use provided key or fallback to hardcoded on
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        # Use the provided API key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBsHFD8YFYUurU1le_7QAfJB92Lldkr6Ms"
    
    # Create and return the model
    return GoogleGenerativeAI(model=model_name, temperature=temperature)
