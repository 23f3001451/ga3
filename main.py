from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load the API key from your .env file
load_dotenv()

# Create the FastAPI app (this IS your web server)
app = FastAPI()

# Create the OpenAI client using your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Define what the REQUEST should look like ---
# FastAPI uses this to validate incoming data
class CommentRequest(BaseModel):
    comment: str

# --- Define what the RESPONSE should look like ---
# This also tells OpenAI exactly what JSON shape we want back
class SentimentResponse(BaseModel):
    sentiment: str   # "positive", "negative", or "neutral"
    rating: int      # 1 to 5

# --- The actual endpoint ---
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    # Basic validation — don't process empty comments
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        # Call OpenAI with Structured Outputs
        # This is the key part — we use `parse()` instead of `create()`
        # and pass our Pydantic model as `response_format`
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the user's comment and return: "
                        "sentiment as 'positive', 'negative', or 'neutral', "
                        "and rating as an integer from 1 (very negative) to 5 (very positive)."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format=SentimentResponse,  # This enforces the JSON structure!
        )

        # Extract the structured result directly — no parsing needed!
        result = completion.choices[0].message.parsed
        return result

    except Exception as e:
        # If anything goes wrong (bad API key, network issue, etc.), return a clean error
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
