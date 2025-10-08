# September 10, 2025
# from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"message": "Hello, FastAPI with UV!"}
    

from typing import Union
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
from PIL import Image
from torchvision import transforms
import torch
from app.helper_lib.model import get_model
import numpy as np

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)
nlp = spacy.load("en_core_web_lg")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embeddings")
def embeddings(request: EmbeddingRequest):
    emb_word = nlp(request.word)
    return emb_word.vector.tolist()

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
model = get_model("Homework2")
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
state_dict = torch.load(
            "C:/Users/nikki/sps_genai/app/helper_lib/cnn_epoch_002.pth",  
            map_location=device,
        )
model.load_state_dict(state_dict['model_state_dict'], strict=True)

CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)
@app.post("/CNN")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        label = CLASSES[predicted.item()]
    return {"prediction": label}

#Push to Git
#git status - check which files were updated
#git add .
#git commit -m "Insert Comment" 
#git push