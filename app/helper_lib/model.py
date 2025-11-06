import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 64x64x3 -> 64x64x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # halves H,W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32x32x16 -> 32x32x32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # After two pools: 64->32->16, so 32*16*16=8192, then 100 units
        self.fc2 = nn.Linear(100, 10)            # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))     # -> [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)             # Flatten
        x = F.relu(self.fc1(x))                  # 100 units
        x = self.fc2(x)                          # 10 classes
        return x

class Critic(nn.Module):
        def __init__(self):        
            super(Critic, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
            self.batchnorm = nn.BatchNorm2d(128)
            self.act2 = nn.LeakyReLU(0.2, inplace=True)
            #self.drop2 = nn.Dropout(0.3)

            self.flatten = nn.Flatten()
            self.fc = nn.Linear(7 * 7 * 128, 1)


        def forward(self, x):
            x = self.conv1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.batchnorm(x)
            x = self.act2(x)
            #x = self.drop2(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

class Generator(nn.Module):
        def __init__(self, z_dim):
            super(Generator, self).__init__()
            self.z_dim = z_dim

            self.fc = nn.Linear(z_dim, 7 * 7 * 128)
            self.reshape = lambda x: x.view(x.size(0), 128, 7, 7)

            self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
            self.act1 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

            self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.fc(x)
            x = self.reshape(x)

            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            x = self.deconv2(x)
            x = self.tanh(x)
            return x

def swish(x):
            return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
        def __init__(self):
            super(EnergyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 2 * 2, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = swish(self.conv1(x))
            x = swish(self.conv2(x))
            x = swish(self.conv3(x))
            x = swish(self.conv4(x))
            x = self.flatten(x)
            x = swish(self.fc1(x))
            return self.fc2(x)
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)         
    
class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        # x = self.norm(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

import torch
class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2) 

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2) 
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size like TF reference
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips) 
        x = self.down3(x, skips)    

        x = self.mid1(x)     
        x = self.mid2(x)   

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)
    
import copy

class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        # Override to() to ensure both networks move to the same device
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Use EMA network for inference, main network for training
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            # Debug generation process
            if step % max(1, diffusion_steps // 4) == 0:  # Print 4 times during generation
                print(f"Generation Step {step}/{diffusion_steps}: t={1-step*step_size:.3f}")
                print(f"  Current images std: {current_images.std().item():.4f}")
                print(f"  Pred images std: {pred_images.std().item():.4f}")
                print(f"  Signal rate: {signal_rates.mean().item():.4f}, Noise rate: {noise_rates.mean().item():.4f}")

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))



def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # Flatten diffusion_times to handle any shape
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Compute start and end angles from signal rate bounds
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Linearly interpolate angles
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    # Compute signal and noise rates
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


def get_model(model_name):
    if model_name == "Homework2":
        return SimpleCNN()
    
    if model_name == "GAN":
         z_dim = 100
         gen = Generator(z_dim)
         critic = Critic()
         return gen,critic
    
    if model_name == "Energy":
         nn_energy_model = EnergyModel()
         return nn_energy_model
    
    if model_name == "Diffusion":
         NOISE_EMBEDDING_SIZE = 64
         NUM_CHANNELS = 3
         unet = UNet(32, NUM_CHANNELS, NOISE_EMBEDDING_SIZE)
         diffusion_model = DiffusionModel(unet, offset_cosine_diffusion_schedule)
         # Generate images
         #diffusion_model.eval()
         return diffusion_model


