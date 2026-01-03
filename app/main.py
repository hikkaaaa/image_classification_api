from fastapi import FastAPI, UploadFile, File
import torch
from app.model import SimpleCNN
from app.utils import transform_image
import os
import torch.nn.functional as F
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load the model once
#determine the path to the model file safely
script_dir = os.path.dirname(os.path.abspath(__file__)) 
#__file__ - speacial variable in python that contains the path to the current python file that is being executed
#absolute path - a full path from the root of the filesystem
model_path = os.path.join(script_dir, '..', 'models', 'classifier.pth')

model = SimpleCNN()
model.load_state_dict(torch.load(model_path))
model.eval() #set the model to evaluation mode

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

@app.post('/predict/')
async def predict(file: UploadFile = File(...)): 
    #read bytes
    image_bytes = await file.read()
    
    #transform
    tensor = transform_image(image_bytes)

    #predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim = 1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()

    return {
        "class": predicted_class,
        "confidence": float(confidence_score)
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")