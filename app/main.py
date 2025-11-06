
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
from torchvision.utils import save_image

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]
'''
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
'''


#CNN Model
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
model = get_model("Homework2")
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
state_dict = torch.load(
            "C:/Users/nikki/sps_genai/app/helper_lib/cnn_epoch_002.pth",  
            map_location=device,
        )
model.load_state_dict(state_dict['model_state_dict'], strict=True)

'''
#GAN Model
generator,_ = get_model("GAN")
gan_state_dict = torch.load(
            "C:/Users/nikki/sps_genai_v2/app/checkpoints_gan/epoch_009.pth",  
            map_location=device,
        )
generator.load_state_dict(gan_state_dict['model_state_dict'], strict=True)


'''

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


'''
@app.post("/GAN")
async def generate_image():
    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        fake = generator(noise).detach() #Tensor
        save_image(fake, 'C:/Users/nikki/sps_genai_v2/app/data/GeneratedImages/image3.png')
    return {"status": "Ok"}
'''




def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    nn_energy_model.eval()

    for w in nn_energy_model.parameters():
         w.requires_grad = False
    inp_imgs = inp_imgs.detach().requires_grad_(True)

    for step in range(steps):
        # We add noise to the input images, but we will
        # need to calculate the gradients with the transformed
        # noisy images, so tell pytorch not to track the gradient 
        # yet, this way we can avoid unnecessary computations that
        # pytorch does in order to calculate the gradients later:
        
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        # Compute energy and gradients
        energy = nn_energy_model(inp_imgs)

        # The gradient with respect to parameters is usually done automatically 
        # when we train a neural network as part of .backward() call.
        # Here we do it manually and specify that the gradient should be with 
        # respect to the input images, not the parameters.
        # In addition because energy contains energy values for each input image
        # in a batch, we need to specify an extra grad_outputs argument for the 
        # right gradients to be calculated for each input image.
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # Finally, apply gradient clipping for stabilizing the sampling
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size*grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


#Energy Model
nn_energy_model = get_model("Energy")
energy_state_dict = torch.load(
            "app/helper_lib/checkpoints/energy_epoch_001.pth",  
            map_location=device,
        )
nn_energy_model.load_state_dict(energy_state_dict['model_state_dict'], strict=True)

@app.post("/Energy")
async def generate_image():
    x = torch.rand((8, 3, 32, 32), device=device) * 2 - 1  # Uniform in [-1, 1]
    new_imgs = generate_samples(nn_energy_model, x, steps=256, step_size=10.0, noise_std=0.01)
    save_image(new_imgs, 'app/data/GeneratedImages/Energy1.png') 
    return {"status": "Ok"}



#Diffusion Get Model
diffusion_model = get_model("Diffusion")
diffusion_state_dict = torch.load(
            "app/helper_lib/checkpoints/diffusion_epoch_001.pth",  
            map_location=device,
        )
diffusion_model.network.load_state_dict(diffusion_state_dict['model_state_dict'], strict=True)


#Diffusion
@app.post("/Diffusion")
async def generate_image():
    with torch.no_grad():
        samples = diffusion_model.generate(num_images=1, image_size=32, diffusion_steps=1000)  # returns tensor in [0, 1]
        # Convert to numpy for plotting
        image = samples[0].cpu()
        save_image(image, 'C:/Users/nikki/sps_genai_v2/app/data/GeneratedImages/Diffusion1.png')
    return {"status": "Ok"}


#Push to Git
#git status - check which files were updated
#git add .
#git commit -m "Insert Comment" 
#git push