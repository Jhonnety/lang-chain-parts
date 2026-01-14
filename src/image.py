from dotenv import load_dotenv

from base64 import b64encode

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from pathlib import Path

load_dotenv()

model = init_chat_model(
    model = 'gpt-4.1-mini'
)

# This case is for url image
message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image'},
        {'type': 'image', 'url': 'https://lokl.life/_next/image?url=https%3A%2F%2Flokl-assets.s3.us-east-1.amazonaws.com%2Fhome%2Fbenefits%2FBenefit-04.png&w=3840&q=75'}
    ]
}

# This case is for decode a image in base64

BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / "assets" / "resource-1.jpg"


#message64_v0 = {
#    "role": "user",
#    "content": [
#        {"type": "text", "text": "Describe the contents of this image"},
#        {
#            "type": "image",
#            "base64": b64encode(
#                open("assets/resource-1.jpg", "rb").read()
#            ).decode(),
#            "mime-type": "image/jpeg"
#        }
#    ]
#}

message64_v1 = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the contents of this image"},
        {
            "type": "image",
            "base64": b64encode(image_path.read_bytes()).decode(),
            "mime_type": "image/jpeg"
        }
    ]
}

message64_v2 = HumanMessage(
    content = [
        {"type": "text", "text": "Describe the contents of this image"},
        {
            "type": "image",
            "base64": b64encode(image_path.read_bytes()).decode(),
            "mime_type": "image/jpeg"
        }
    ]
)

response = model.invoke([message64_v2])
print(response.content)