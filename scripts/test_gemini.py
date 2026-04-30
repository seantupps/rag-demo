# python scripts/test_gemini.py

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

TEST_MODELS = [
    "gemini-2.5-flash"
]

def test_model(model_id):
    print(f"\n--- Testing Model: {model_id} ---")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Hey there"
        )
        print("SUCCESS!")
        print(f"Response: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    for m in TEST_MODELS:
        test_model(m)
