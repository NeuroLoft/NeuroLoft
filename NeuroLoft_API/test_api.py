import requests
from PIL import Image, ImageDraw
import io
import numpy as np

# 1. Create a dummy image (Black background, White digit-like shape)
img = Image.new('L', (28, 28), color=0)
d = ImageDraw.Draw(img)
d.text((10, 10), "7", fill=255) # Draw a '7'

# Save to bytes
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# 2. Send POST request
url = 'http://127.0.0.1:8000/predict'
files = {'file': ('digit.png', img_byte_arr, 'image/png')}

print("📡 Sending request to NeuroLoft API...")
try:
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print("✅ Success!")
        print("Response:", response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("Make sure the server is running (python -m uvicorn app.main:app --reload)")
