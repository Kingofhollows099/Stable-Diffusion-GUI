from diffusers import DiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)
base.to("cpu")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae
)
refiner.to("cpu")

# Define parameters
n_steps = 40
high_noise_frac = 0.8
prompt = input("Prompt: ")

# Define how many steps and what % of steps to be run on each experts (80/20) here
latents = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images[0]

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latents,
).images[0]

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()

# Create a simple tkinter window for the prompt
root = tk.Tk()
root.withdraw()  # Hide the main window

# Show a confirmation dialog
if messagebox.askyesno("Save Image", "Do you want to save this image?"):
    # If user clicks "Yes", open the file dialog
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        # Save the image
        image.save(file_path)
        print(f"Image saved to: {file_path}")
    else:
        print("Save cancelled")
else:
    print("Image not saved")