import os
import google.generativeai as genai

API_KEY = "AIzaSyBDVd1au8lRt6Z8uZuCJ2UCGWeusCr8yGY"

def chat(prompt, model = "gemini-1.5-flash-002"):
    # Configure Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel()
    response = model.generate_content(prompt)

    return response.text
