import torch
from diffusers import DDPMScheduler
import PIL.Image
import numpy as np
import uuid
import tqdm
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

repo_id = "google/ddpm-cat-256"
id_inference = uuid.uuid1()


model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
model.to("mps")

torch.manual_seed(0)

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size).to("mps")

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample


scheduler = DDPMScheduler.from_pretrained(repo_id)

less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
less_noisy_sample.shape

plt.ion()
plt.figure()
plt.show()

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    plt.imshow(image_pil)
    plt.title(f"Image at step {i}")
    plt.axis('off')  # Hide axis
    plt.draw()
    plt.pause(0.001)
    return image_pil


sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        pil = display_sample(sample, i + 1)

final_image = display_sample(sample, len(scheduler.timesteps))
final_image.save(f"outputs/final_noisy_inference_{id_inference}_final.png")

plt.ioff()
plt.show()