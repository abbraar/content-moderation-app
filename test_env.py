import os
from dotenv import load_dotenv

print("Loading .env file...")
load_dotenv()

print("Environment variables:")
for key, value in os.environ.items():
    if 'GEMINI' in key.upper():
        print(f"{key}: {'*' * 8} (value hidden for security)")
    else:
        print(f"{key}: {value}")

print("\nTrying to access GEMINI_API_KEY:")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print("GEMINI_API_KEY is set!")
    print(f"Key length: {len(gemini_key)} characters")
    print(f"Starts with: {gemini_key[:3]}...")
    print(f"Ends with: ...{gemini_key[-3:]}")
else:
    print("GEMINI_API_KEY is NOT set or is empty")
