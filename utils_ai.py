import os
import base64
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, PineconeApiException
import replicate

load_dotenv()

# initialize openai client
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)
model_gpt = "gpt-4-0125-preview"
model_embeddings = "text-embedding-3-small"
model_vision = "gpt-4-vision-preview"

# initialize pinecone client and index
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index_name = os.getenv('PINECONE_INDEX')
index = pc.Index(index_name)

# replicate api for llava
replicate = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# extract order information as json using gpt
def json_gpt(clean_order):
    request_message_content = f"""
    You are given a food order and a JSON file template.
    Your task is to extract information from the food order and return the information in the given JSON format.
    Return only the JSON file, nothing else.
    Please make sure to return complete information and to not invent any information.
    Use ISO 8601 format for datetime. We are currently in 2024.
    Use the official abbreviation of the unit of measurement like “kg”, “g”, or “lbs”,  and use "each" if the unit of measurement is not specified or that it is "by the unit".
    If you cannot fill in a field, leave it blank.
                    
    Food order:
    "{clean_order}"

    JSON format:
    {{
        "order": {{
            "items": [
                {{
                    "name": "ItemName1",
                    "code": "ItemCode1",
                    "unit": "UnitTypeItem1",
                    "quantity": quantityNumberItem1,
                }},
            ...
            ],
            "delivery": {{
                "datetime": "YYYY-MM-DDTHH:MM:SS",
                "method": "deliveryMethod"
            }},
            "comment": "Any comment in the order"
        }}
    }}
    """
    try:
        completion = client.chat.completions.create(model=model_gpt, messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": request_message_content},
        ], stream=False, max_tokens=2000, temperature=0)
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Failed to generate JSON with GPT: {e}")

# embed items and search in vector index
def query_embeddings_and_search(query):
    try:
        xq = client.embeddings.create(input=query, model=model_embeddings).data[0].embedding
        res = index.query(vector=[xq], top_k=5, include_metadata=True)
        return res
    except PineconeApiException as e:
        raise Exception(f"An error occurred during vector search: {e}")

# transcribe audio with whisper
def transcribe_audio(audio):
    try:
        filename = 'input.mp3'
        with open(filename, "wb") as wav_file:
            wav_file.write(audio.read())
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                language="en",
                file=audio_file,
                response_format="text"
            )
        os.remove(filename)
        return transcript
    except Exception as e:
        raise Exception(f"Failed to transcribe audio: {e}")

# encode image to base64 for gpt-vision and llava api calls
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# extract text from an image using gpt-vision
def extract_text_from_image(uploaded_image):
    try:
        base64_image = encode_image(uploaded_image)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        payload = {
            "model": f"{model_vision}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please return all the text that is on this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Failed to extract text from image with status code {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to read image: {e}")

# extract text from an image using llava 1.6 34b
def extract_text_from_image_llava(uploaded_image):
    try:
        base64_image = encode_image(uploaded_image)
        output = replicate.run(
            "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
            input={
                "prompt": "return all the text in the image",
                "image": f"data:image/jpeg;base64,{base64_image}"
            }
        )
        # the model streams the answer so we store it gradually before returning it
        text_output = ""
        for item in output:
            text_output += item + "\n"
        return text_output
    except Exception as e:
        raise Exception(f"Failed to read image: {e}")