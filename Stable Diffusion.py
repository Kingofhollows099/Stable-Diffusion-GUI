#Adds a progress bar
import tkinter as tk
from tkinter import ttk, filedialog
from diffusers import DiffusionPipeline
import threading
from PIL import Image, ImageTk

class StableDiffusionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stable Diffusion Image Generator")
        master.geometry("600x700")

        self.prompt_label = ttk.Label(master, text="Prompt:")
        self.prompt_label.pack(pady=5)

        self.prompt_entry = ttk.Entry(master, width=50)
        self.prompt_entry.pack(pady=5)

        self.generate_button = ttk.Button(master, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.progress_label = ttk.Label(master, text="0%")
        self.progress_label.pack(pady=5)

        self.status_label = ttk.Label(master, text="")
        self.status_label.pack(pady=5)

        self.image_label = ttk.Label(master)
        self.image_label.pack(pady=10)

        self.save_button = ttk.Button(master, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        self.base = None
        self.refiner = None
        self.generated_image = None

    def load_models(self):
        self.status_label.config(text="Loading models...")
        self.master.update()

        self.base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        self.base.to("cpu")

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae
        )
        self.refiner.to("cpu")

        self.status_label.config(text="Models loaded successfully!")

    def generate_image(self):
        if not self.base or not self.refiner:
            self.load_models()

        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.config(text="Please enter a prompt.")
            return

        self.status_label.config(text="Generating image...")
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="0%")
        self.master.update()

        def generate():
            n_steps = 40
            high_noise_frac = 0.8

            def base_callback(step, timestep, latents):
                progress = (step / n_steps) * 50  # Base model progress is 50% of total
                self.master.after(0, self.update_progress, progress)

            def refiner_callback(step, timestep, latents):
                progress = 50 + (step / n_steps) * 50  # Refiner progress is the other 50%
                self.master.after(0, self.update_progress, progress)

            latents = self.base(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
                callback=base_callback,
                callback_steps=1
            ).images

            images = self.refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=latents,
                callback=refiner_callback,
                callback_steps=1
            ).images

            self.generated_image = images[0]
            self.master.after(0, self.update_gui)

        threading.Thread(target=generate, daemon=True).start()

    def update_progress(self, progress):
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"{progress:.1f}%")
        self.master.update_idletasks()

    def update_gui(self):
        self.display_image(self.generated_image)
        self.status_label.config(text="Image generated successfully!")
        self.generate_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 100
        self.progress_label.config(text="100%")

    def display_image(self, image):
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def save_image(self):
        if self.generated_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.generated_image.save(file_path)
                self.status_label.config(text=f"Image saved to: {file_path}")
        else:
            self.status_label.config(text="No image to save.")

root = tk.Tk()
app = StableDiffusionGUI(root)
root.mainloop()