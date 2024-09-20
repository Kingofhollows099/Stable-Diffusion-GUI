import warnings
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore", message="Accessing config attribute `vae_latent_channels` directly via 'VaeImageProcessor' object attribute is deprecated.")

# Load base model
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)
base.to("cpu")

# Function to get user input
def get_user_input():
    use_image = input("Do you want to use an image as a reference? (yes/no): ").lower().strip() == 'yes'
    prompt = input("Enter your prompt: ")
    num_images = int(input("Number of images to generate: "))
    return use_image, prompt, num_images

# Function to load image
def load_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        return Image.open(file_path)
    return None

# Get user input
use_image, prompt, num_images_per_prompt = get_user_input()

# Load image if user chose to use one
init_image = None
if use_image:
    init_image = load_image()
    if init_image is None:
        print("No image selected. Proceeding with text-to-image generation.")
        use_image = False

# Set up the appropriate pipeline
if use_image:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae
    )
else:
    pipe = base

pipe.to("cpu")

print("Generating images... This may take a while.")

# Generate images
if use_image:
    images = pipe(
        prompt=prompt,
        image=init_image,
        num_images_per_prompt=num_images_per_prompt
    ).images
else:
    images = pipe(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="pil",
        num_images_per_prompt=num_images_per_prompt
    ).images

print("Images generated successfully!")

# Display the images
fig, axes = plt.subplots(1, num_images_per_prompt, figsize=(20, 5))
for i, image in enumerate(images):
    if num_images_per_prompt > 1:
        axes[i].imshow(image)
        axes[i].axis('off')
    else:
        axes.imshow(image)
        axes.axis('off')
plt.tight_layout()
plt.show()

# Create a simple tkinter window for the dialogs
root = tk.Tk()
root.withdraw()  # Hide the main window

# Show a confirmation dialog
if messagebox.askyesno("Save Images", "Do you want to save these images?"):
    # If user clicks "Yes", open the folder selection dialog
    folder_path = filedialog.askdirectory(title="Select folder to save images")
    if folder_path:
        for i, image in enumerate(images):
            file_path = f"{folder_path}/image_{i+1}.png"
            image.save(file_path)
        print(f"Images saved to: {folder_path}")
        messagebox.showinfo("Save Complete", f"Images saved to: {folder_path}")
    else:
        print("Save cancelled")
else:
    print("Images not saved")

root.destroy()  # Close the tkinter window