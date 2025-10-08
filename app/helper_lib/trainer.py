import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """
    Enhanced training loop with checkpoint functionality

    TODO: Implement training loop that:
    1. Trains the model for specified epochs
    2. Tracks training and validation metrics
    3. Automatically saves checkpoints each epoch
    4. Saves the best performing model
    5. Returns the trained model

    Hint: Look at the FCNN notebook for checkpoint implementation examples
    """
    # TODO: Implement training loop with checkpoint saving
    return model