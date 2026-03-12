# ==============================
# Load Environment Variables
# ==============================
from dotenv import load_dotenv
import os

load_dotenv()

# Get GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# ==============================
# Convert Image → Base64
# ==============================
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ==============================
# Setup GROQ Multimodal Model
# ==============================
from groq import Groq

# Recommended model
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ==============================
# Image + Query Analysis
# ==============================
def analyze_image_with_query(query, image_path):

    encoded_image = encode_image(image_path)

    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL
    )

    return response.choices[0].message.content


# ==============================
# Test Run
# ==============================
if __name__ == "__main__":
    query = "Is there any medical issue visible?"
    image_path = "Sample Images/acne.jpg"  # change image if needed

    result = analyze_image_with_query(query, image_path)

    print("\n🩺 Doctor AI Response:\n")
    print(result)