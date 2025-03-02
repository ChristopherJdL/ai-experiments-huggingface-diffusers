from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import uuid

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

image = pipeline("""
                 A plush toy of a polar bear wearing a red scarf in a house with a view of London, super realistic 4K image.
                 The plush toy is round and soft and wears the red scarf around its neck. The plush toy is short, and has a long nose.
                 The eyes of the polar bear are like watermelon seeds.
                 """).images[0]
image.save("new{}.png".format(uuid.uuid1()))