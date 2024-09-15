import torch
import diffusers
from diffusers import FluxPipeline
#from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer

# Modify the rope function to handle MPS device
_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)

diffusers.models.transformers.transformer_flux.rope = new_flux_rope

def create_flux_pipeline():
    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("mps")
    return pipeline

def generate_image(pipeline, prompt, height=1024, width=1024,num_inference_steps=50):
    image = pipeline(prompt, height=height, width=width, num_inference_steps=num_inference_steps).images[0]
    return image

if __name__ == "__main__":
    pipeline = create_flux_pipeline()
    prompt = "A beautiful landscape with mountains and a lake"
    image = generate_image(pipeline, prompt)
    image.save("flux_generated_image.png")