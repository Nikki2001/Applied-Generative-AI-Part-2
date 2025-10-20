import torch
import os

#Update checkpoints_cnn name based on the project running
def save_checkpoint(model, optimizer, epoch, loss, accuracy='NA', checkpoint_dir='checkpoints_cnn'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    # Save latest checkpoint #Update file name to be FCNN/CNN/CAN/etc. based on the run
   #f'epoch_
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy