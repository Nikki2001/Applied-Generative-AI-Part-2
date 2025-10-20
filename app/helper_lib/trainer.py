import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint
import torch.optim as optim


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
    
    datalogs = []
    best_accuracy = 0.0

# Train the model
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.0005)

# To load from a checkpoint, uncomment the line below:
# start_epoch, _, _ = load_checkpoint(model, optimizer, 'checkpoints_fcnn/fcnn_epoch_005.pth', device)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.argmax(outputs.data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()  

            if (batch_number % 100 == 99):
                train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

                datalogs.append({
                    "epoch": epoch + batch_number / len(train_loader), 
                    "train_loss": running_loss / (batch_number + 1),
                    "train_accuracy": running_correct/running_total,
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        datalogs.append({
            "epoch": epoch + 1, 
            "train_loss": epoch_loss,
            "train_accuracy": running_correct/running_total,
        })

        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy
        )

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, 
                checkpoint_dir='checkpoints_fcnn/best'
            )
            print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")
        print(f"Checkpoint saved: {checkpoint_path}")

        widget.src = losschart(datalogs)

        print("Finished Training")
        return model
    


def train_gan(generator, critic, data_loader, criterion, device='cpu', epochs=10):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model
    from tqdm import tqdm

    datalogs = []

    z_dim = generator.z_dim
    lr = 5e-5
    n_critic = 5
    clip_value = 0.01

    gen = generator.to(device)
    critic = critic.to(device)

    opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)

    for epoch in range(epochs):
        # for batch_idx, (real, _) in enumerate(dataloader):
        # model.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            ## === Train Critic === ##
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise).detach()
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            ## === Train Generator === ##
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()


            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix(
                    {
                        "Batch": f"{batch_number}/{len(data_loader)}",
                        "D loss": f"{loss_critic.item():.4f}",
                        "G loss": f"{loss_gen.item():.4f}",
                    }
                )
                datalogs.append(
                    {
                        "epoch": epoch + batch_number / len(data_loader),
                        "Batch": batch_number/len(data_loader),
                        "D loss": loss_critic.item(),
                        "G loss": loss_gen.item(),
                    }
                )

            # if batch_idx % 100 == 0:
            #     print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(dataloader)}] "
            #           f"[D loss: {loss_critic.item():.4f}] [G loss: {loss_gen.item():.4f}]")

            save_checkpoint(generator, opt_gen, epoch, loss_gen, checkpoint_dir='checkpoints_gan')
    return generator, critic