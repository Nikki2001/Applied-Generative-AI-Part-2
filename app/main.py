
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
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

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


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
gpt2_state_dict = torch.load(
            "app/gpt2_checkpoints/gpt2.pth",  
            map_location=device,
        )
model.load_state_dict(gpt2_state_dict['model_state_dict'], strict=True)
tokenizer = AutoTokenizer.from_pretrained('./app/gpt2data')
SEQ_LEN = 30
class TextGenerator:
    def __init__(self, model, top_k=10):
        self.model = model
        self.model.to(device)

    def sample_from(self, probs, temperature):
        probs[1] = 0  # Mask out UNK token (index 1) to prevent generating <UNK>
        probs = torch.nn.functional.softmax(probs/temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        return next_id, probs

    def generate(self, question, context, max_tokens, temperature):
        self.model.eval()
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        generated_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=SEQ_LEN,
            padding = False,
            return_tensors=None
        ).input_ids

        info = []

        with torch.no_grad():
            while len(generated_tokens) < max_tokens:
                x = torch.tensor([generated_tokens], dtype=torch.long)
                x = x.to(device)
                logits = self.model(x).logits
                last_logits = logits[0, -1] # .cpu().numpy()
                sample_token, probs = self.sample_from(last_logits, temperature)
                generated_tokens.append(sample_token)
                info.append({
                    "prompt": question,
                    "word_probs": probs,
                })
                if sample_token == 0:
                    break
        print("GEN", generated_tokens)
        generated_words = tokenizer.decode(generated_tokens)
        print("generated text:" + " ".join(generated_words))
        return generated_words


text_generator = TextGenerator(model)

class TextGenerationRequest(BaseModel):
    question: str
    context: str
    length: int

@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    generated_text = text_generator.generate(request.question, request.context, max_tokens=30, temperature=3.0)
    return {"generated_text": generated_text}

#Push to Git
#git status - check which files were updated
#git add .
#git commit -m "Insert Comment" 
#git push