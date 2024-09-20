from diffusers import DiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings

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
num_images_per_prompt = 1

print("Generating images... This may take a while.")

# Generate multiple images
latents = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    num_images_per_prompt=num_images_per_prompt
).images

images = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latents,
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