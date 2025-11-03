from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the same model you used in your notebook
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Build the FastAPI app
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    result = sentiment_analyzer(input.text)[0]
    return {
        "input_text": input.text,
        "label": result["label"],
        "score": round(result["score"], 3)
    }

@app.get("/")
def home():
    return {"message": "Sentiment API is running successfully!"}