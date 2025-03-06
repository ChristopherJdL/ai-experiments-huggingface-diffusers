from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from datasets import Dataset, Image
from TrainingConfig import config
from PIL import Image as PILImage
from torchvision import transforms
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import os

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

#Here, we initialize the dataset object using the huggingface library.
dataset = Dataset.from_csv(path_or_paths="abei-images/dataset.csv", split="train")

#This is a very important step. We cast the column image with the type datasets.Image,
# Which will help processing images directly during training.
dataset = dataset.cast_column("image", Image())

train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

def display_sample(sample, i):
    image_pil = sample
    plt.imshow(image_pil)
    plt.title(f"Image at step {i}")
    plt.axis('off')  # Hide axis
    plt.draw()
    plt.pause(0.001)
    return image_pil

"""
A tensor, in image deep learning, is metadata about an image and also its numerical representation.
There are three-dimension tensors: [3,256,256] [RGB channels, width, height]
And four-dimension tensors: [1, 3, 256, 256] [batch size (number of images), RGB channels, width, height]

Unsqueeze will produce a four-dimension tensor.
Some models require four-dimension tensors, like PyTorch's UNet2DModel.
Example of a 4D tensor:
image_tensor = [
    # Batch dimension (1 image)
    [
        # Red channel (256x256 grid)
        [[0.784, 0.212, ..., 0.031], ...],  # 256 rows
        
        # Green channel 
        [[0.102, 0.945, ..., 0.627], ...],
        
        # Blue channel
        [[0.439, 0.298, ..., 0.882], ...]
    ]
]
Important: Values represent normalized intensity, not actual bytes (0.0=black, 1.0=full color).
+--------- APPLE TENSOR ---------+
| Area   |   R   |   G   |   B   |
+========+=======+=======+=======+
| Bright | 0.784 | 0.102 | 0.439 |
|--------|-------|-------|-------|
| Medium | 0.212 | 0.945 | 0.298 |
|--------|-------|-------|-------|
| Shadow | 0.031 | 0.627 | 0.882 |
+==================================+
 R: Red   G: Green   B: Blue
"""
def initialize_tensor(image):
    threeDimensionTensor = transform(image)
    fourDimensionTensor = threeDimensionTensor.unsqueeze(0).to("cpu")
    return fourDimensionTensor

def make_noisy_image(sample_image, timesteps):
    #a scheduler is a class that decides how much noise to apply to an image.
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise = torch.randn(sample_image.shape).to("cpu")
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    return noisy_image, noise, noise_scheduler

def make_PIL_sample(noisy_image):
    # permutation is necessary because the elements in the array need to be reordered from BCHW → BHWC.
    # PyTorch Default: Batch, Channels, Height, Width (BCHW)
	# PIL Requirement: Height, Width, Channels (HWC)
    permuted_tensor = noisy_image.permute(0, 2, 3, 1)
    # scaling operation is necessary to put the values into an 8-bit unsigned int, or an unsigned char,
    # so that it can be displayed on as an image. Otherwise, garbage would be displayed.
    scaled_tensor = (permuted_tensor + 1.0) * 127.5
    # we cast the tensor into an unsigned char array, and then we convert it to a numpy array as necessary by PIL
    # Let's note the [0] at the end, which removes the batch dimension of the array, so that we take the only image from the batch.
    unsigned_char_numpy_array = (scaled_tensor).type(torch.uint8).cpu().numpy()[0]
    sample = PILImage.fromarray(unsigned_char_numpy_array)
    return sample

# This method initializes a UNet2DModel, which is a Convolutional Neural Network.
# CNNs are primarily used for computer vision.
def initialize_cnn_model():
    return UNet2DModel(
    sample_size=(config.image_size, config.image_size2),  # the target image resolution
    in_channels=3, # the number of input channels, 3 for RGB images
    out_channels=3, # the number of output channels
    layers_per_block=2, # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512), # the number of output channels for each UNet block
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
).to("cpu")

# The loss, in computer vision learning, defines the difference between the prediction and the reality.
def calculate_loss(actual_noise):
    noise_pred = model(noisy_image, timesteps).sample
    loss = F.mse_loss(noise_pred, actual_noise)
    return loss

sample_image = initialize_tensor(dataset[0]["image"])

#The timesteps is a tensor as well, so we have to convert it to Metal-compatible variable. 
#That's because every tensor has to be with mps in order to be exploitable withtin the GPU.
timesteps = torch.LongTensor([50]).to("cpu")

noisy_image, noise, noise_scheduler = make_noisy_image(sample_image, timesteps)
sample = make_PIL_sample(noisy_image)
#display_sample(sample, 0)

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

#Here we pre-process the image to be in RGB format.
dataset.set_transform(lambda examples: {
    "images": [preprocess(image.convert("RGB")) for image in examples["image"]]
    })

train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = initialize_cnn_model()

loss = calculate_loss(noise)

#for the training phase, an optimizer and a scheduler are required
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline):#ranger après
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler): #put in its own file and separate the modules in the 
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)