import os
import google.generativeai as genai
from dotenv import load_dotenv

print("--- Simple Google API Connection Test ---")

print("\n1. Loading .env file...")
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("!!! TEST FAILED: Could not find GOOGLE_API_KEY in your .env file.")
    exit()

print("   API Key loaded successfully.")

print("\n2. Configuring Google AI with your key...")
try:
    genai.configure(api_key=api_key)
    print("   Configuration successful.")
except Exception as e:
    print(f"!!! TEST FAILED: Could not configure Google AI. Error: {e}")
    exit()

print("\n3. Asking Google what models are available...")
try:
    model_found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - Found model: {m.name}")
            if "gemini-1.0-pro" in m.name:
                model_found = True
    
    print("\n--- Test Result ---")
    if model_found:
        print("✅ SUCCESS! Your environment is working correctly and can see 'gemini-1.0-pro'.")
    else:
        print("❌ FAILED: Could not find 'gemini-1.0-pro' in the list of available models.")

except Exception as e:
    print(f"\n!!! TEST FAILED: An error occurred while listing models. Error: {e}")