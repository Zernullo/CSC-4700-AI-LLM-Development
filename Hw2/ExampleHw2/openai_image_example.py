"""
OpenAI Image Generation Example

What this does:
    Shows how to generate images from text descriptions using OpenAI's DALL-E.
    You provide a text prompt describing what you want, and the API creates
    an image that you can save to your computer.

What you'll need:
    - openai: Library to connect to OpenAI's API
    - base64: Converts the image data from text format to actual image bytes
    - os: Reads environment variables (like your API key)
    - dotenv: Loads your API key from a .env file

What you'll learn:
    - How to generate images from text descriptions
    - How to set image options (size, quality, how many to generate)
    - How to decode the base64 image data
    - How to save the generated images to your computer
"""

from openai import OpenAI
import base64
import os
from dotenv import load_dotenv


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Oh, when Dr. Mills did his capstone project ECE 492...
prompt = "Jeep-like toy rover sliding across an ice patch on the Moon"

# Different function to generate images
response = client.images.generate(
    model="gpt-image-1",        # Model
    prompt=prompt,              # Text prompt
    n=1,                        # Number of images to generate
    size="1024x1024",           # Image size (affects cost)
    quality="medium"            # {low, medium, high} quality setting
)                               # (Affects cost & speed)

# Extract & save image
image_base64 = response.data[0].b64_json

# Decode and save the image
with open("moon_rover.png", "wb") as f:
    f.write(base64.b64decode(image_base64))

print("moon_rover.png")
