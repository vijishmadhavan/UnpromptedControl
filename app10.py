import gradio as gr
import numpy as np
import torch
from src.pipeline_stable_diffusion_controlnet_inpaint import *

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2


controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

# speed up diffusion process with faster scheduler and memory optimization
# remove following line if xformers is not installed
#pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')



def predict2(input_dict, mask_dict, prompt):
    # Get the drawn input image and mask
    input_image = input_dict["image"].convert("RGB")
    mask1 = input_dict["mask"].convert("RGB")

    mask = mask_dict["mask"].convert("RGB")

    # Process the input image and mask
    input_image = np.array(input_image)
    low_threshold = 500
    high_threshold = 600
    canny = cv2.Canny(input_image, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    generator = torch.manual_seed(0)
    output = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=input_image,
        control_image=canny_image,
        controlnet_conditioning_scale=0.5,
        mask_image=mask
    ).images[0]
    return output
def predict3(input_dict, prompt):
    # Get the drawn input image and mask
    image = input_dict["image"].convert("RGB")
    input_image = input_dict["mask"].convert("RGB")
    #mask = mask_dict["mask"].convert("RGB")


    # Convert images to numpy arrays
    image_np = np.array(image)
    input_image_np = np.array(input_image)

    # Convert input_image_np to grayscale and normalize to [0, 1] range
    mask_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY) / 255.0

    # Blend the original image and the input_image using the mask
    blended_image_np = image_np * (1 - mask_np)[:, :, None] + input_image_np * mask_np[:, :, None]

    # Convert the blended image back to a PIL Image
    blended_image = Image.fromarray(np.uint8(blended_image_np))

    # Process the blended image
    blended_image_np = np.array(blended_image)
    low_threshold = 2000
    high_threshold = 2100
    canny = cv2.Canny(blended_image_np, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    #save canny image
    canny_image.save("canny.png")

    generator = torch.manual_seed(0)
    output = pipe(
        prompt,
        negative_prompt="person,animals,birds,pets, text, objects, vehicles, any object",
        num_inference_steps=20,
        generator=generator,
        image=blended_image_np,
        control_image=canny_image,
        controlnet_conditioning_scale=0.6,
        mask_image=input_image
    ).images[0]
    
    return output


def predict(input_dict):
    # Get the drawn input image and mask
    image = input_dict["image"].convert("RGB")
    input_image = input_dict["mask"].convert("RGB")

    # Convert images to numpy arrays
    image_np = np.array(image)
    input_image_np = np.array(input_image)

    # Convert input_image_np to grayscale and normalize to [0, 1] range
    mask_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY) / 255.0

    # Apply OpenCV inpainting
    inpainted_image_np = cv2.inpaint(image_np, (mask_np * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # Blend the original image and the inpainted image using the mask
    blended_image_np = image_np * (1 - mask_np)[:, :, None] + inpainted_image_np * mask_np[:, :, None]

    # Convert the blended image back to a PIL Image
    blended_image = Image.fromarray(np.uint8(blended_image_np))

    # Process the blended image
    blended_image_np = np.array(blended_image)
    low_threshold = 800
    high_threshold = 900
    canny = cv2.Canny(blended_image_np, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    canny_image.save("canny.png")


    generator = torch.manual_seed(0)
    output = pipe(
        prompt="",
        num_inference_steps=20,
        generator=generator,
        image=blended_image_np,
        control_image=canny_image,
        controlnet_conditioning_scale=0.9,
        mask_image=input_image
    ).images[0]
    
    return output
image_blocks = gr.Blocks()

with image_blocks as demo:
    with gr.Row():
        with gr.Column():
            # Allow user to draw on the input image
            input_image = gr.Image(source='upload', tool='sketch', elem_id="input_image_upload", type="pil", label="Upload & Draw on Image")
            # Allow user to draw the mask
            #mask = gr.Image(source='upload', tool='sketch', elem_id="mask_upload", type="pil", label="Draw Mask")
            #prompt = gr.Textbox(label='Your prompt (what you want to add in place of what you are removing)')
            btn = gr.Button("Run")
        with gr.Column():
            result = gr.Image(label="Result")
        btn.click(fn=predict, inputs=[input_image], outputs=result)
demo.launch()
