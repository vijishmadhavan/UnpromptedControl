#best restoration model

import gradio as gr
import numpy as np
import torch
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from scratch_detection import ScratchDetection

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import time
import os

device = "cuda"

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

# speed up diffusion process with faster scheduler and memory optimization
# remove following line if xformers is not installed
#pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')

def combine_masks(mask1, mask2):
    mask1_np = np.array(mask1)
    mask2_np = np.array(mask2)
    combined_mask_np = np.maximum(mask1_np, mask2_np)
    combined_mask = Image.fromarray(combined_mask_np)
    return combined_mask

if not os.path.exists("input_images"):
    os.makedirs("input_images")

def generate_scratch_mask(input_dict):
    # Save the input image to a directory
    input_image = input_dict["image"].convert("RGB")
    input_image_path = "input_images/input_image.png"
    input_image_resized = resize_image(input_image, 768)
    input_image_resized.save(input_image_path)

    test_path = "input_images"
    output_dir = "output_masks"
    scratch_detector = ScratchDetection(test_path, output_dir, input_size="scale_256", gpu=0)
    scratch_detector.run()
    mask_image = scratch_detector.get_mask_image("input_image.png")
    
    # Resize the mask to match the input image size
    mask_image = mask_image.resize(input_image.size, Image.BICUBIC)

    # Apply dilation to make the lines bigger
    kernel = np.ones((5, 5), np.uint8)
    mask_image_np = np.array(mask_image)
    mask_image_np_dilated = cv2.dilate(mask_image_np, kernel, iterations=2)
    mask_image_dilated = Image.fromarray(mask_image_np_dilated)

    return mask_image_dilated

def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(source='upload', tool='sketch', elem_id="input_image_upload", type="pil", label="Upload & Draw on Image")
        mask_image = gr.Image(label="mask")
        output_image = gr.Image(label="output")
    with gr.Row():
        generate_mask_button = gr.Button("Generate Scratch Mask")
        submit = gr.Button("Inpaint")
    
    def inpaint(input_dict, mask):
        image = input_dict["image"].convert("RGB")
        draw_mask = input_dict["mask"].convert("RGB")

        image = resize_image(image, 768)
        
        mask = Image.fromarray(mask)
        mask = resize_image(mask, 768)
        draw_mask = resize_image(draw_mask, 768)

        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        canny = cv2.Canny(image, low_threshold, high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny_image = Image.fromarray(canny)
        generator = torch.manual_seed(0)

        # Combine drawn mask and generated mask
        combined_mask = combine_masks(draw_mask, mask)

        output = pipe(
            prompt="",
            num_inference_steps=20,
            generator=generator,
            image=image,
            control_image=canny_image,
            controlnet_conditioning_scale=0,
            mask_image=combined_mask
        ).images[0]
        return output

    generate_mask_button.click(generate_scratch_mask, inputs=[input_image], outputs=[mask_image])
    submit.click(inpaint, inputs=[input_image, mask_image], outputs=[output_image])
    demo.launch(debug=True)


       
