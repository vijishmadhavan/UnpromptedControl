# UnpromptedControl

ControlNet is a highly regarded tool for guiding StableDiffusion models, and it has been widely acknowledged for its effectiveness. In this repository, I have discovered a technique that allows for the restoration or removal of objects without requiring user prompts. By leveraging this approach, the workflow can be significantly streamlined, leading to enhanced process efficiency.
![restore Result](examples/eg2gif.gif)
![restore Result](examples/objgif.gif)
## Image Restoration 

In this image restoration is accomplished using the controlnet-canny and stable-diffusion-2-inpainting techniques, with only "" blank input prompts. Additionally, for automatic scratch segmentation, the FT_Epoch_latest.pt model is being used. However, if the segmentation output is not satisfactory, it is possible to manually sketch and refine the mask to achieve better results. As ControlNet model is trained on pairs of images, one of which has missing parts, and it learns to predict the missing parts based on the content of the complete image.

![restore Result](examples/eg1.jpg)

![restore Result](examples/eg2.jpg)

## Object Removal

Automatically removing objects from images is a challenging task that requires a combination of computer vision and deep learning techniques. This code leverages the power of OpenCV inpainting, deep learning-based image restoration, and blending techniques to achieve this task automatically, without the need for user prompts. The ControlNetModel and StableDiffusionInpaintPipeline models play a crucial role in guiding the inpainting process and restoring the image to a more natural-looking state. Overall, this code provides an efficient and effective way to remove unwanted objects from images and produce natural-looking results that are consistent with the surrounding image content. 

**For sure it has limitations and fails with some images(mostly faces), we need to mask the object as well as the shadow from the object to get good results.**


![restore Result](examples/obj2.jpg)
![restore Result](examples/obj1.jpg)





