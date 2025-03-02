from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
import torchvision.transforms as transforms
from datasets import Dataset, Image
from TrainingConfig import config
import matplotlib.pyplot as plt
import torch
from PIL import Image as PILImage
from torchvision import transforms
import numpy as np

def display_sample(sample, i):
    image_pil = sample
    plt.imshow(image_pil)
    plt.title(f"Image at step {i}")
    plt.axis('off')  # Hide axis
    plt.draw()
    plt.pause(0.001)
    return image_pil

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = Dataset.from_csv(path_or_paths="abei-images/dataset.csv", split="train").cast_column("image", Image())
print(dataset.column_names)


train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

sample_image = transform(dataset[0]["image"]).unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

sample = PILImage.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
display_sample(sample, 0)

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size2)),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5]),
    ]
)
dataset.set_transform(lambda examples: {"images": [preprocess(image.convert("RGB")) for image in examples["image"]]})

train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=(config.image_size, config.image_size2),  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

import torch.nn.functional as F

print("Sample image shape:", sample_image.shape)  

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

plt.show()
