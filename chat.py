import os
import google.generativeai as genai

API_KEY = "GEMINI_API_KEY"

def chat(prompt, model = "gemini-1.5-flash-002"):
    # Configure Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel()
    response = model.generate_content(prompt)

    return response.text
