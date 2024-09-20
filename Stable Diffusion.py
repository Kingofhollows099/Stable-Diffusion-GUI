import tkinter as tk
from tkinter import ttk, filedialog
from diffusers import DiffusionPipeline
import threading
from PIL import Image, ImageTk
import torch

class CropWindow(tk.Toplevel):
    def __init__(self, master, image, target_size=(512, 512)):
        super().__init__(master)
        self.title("Crop Image")
        self.image = image
        self.target_size = target_size
        self.cropped_image = None

        self.canvas = tk.Canvas(self, width=600, height=600)
        self.canvas.pack()

        self.display_image = ImageTk.PhotoImage(self.image.resize((600, 600), Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)

        self.crop_rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="red")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.crop_button = ttk.Button(self, text="Crop", command=self.crop_image)
        self.crop_button.pack(pady=10)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        w, h = self.target_size
        aspect_ratio = w / h

        width = cur_x - self.start_x
        height = width / aspect_ratio

        self.canvas.coords(self.crop_rect, self.start_x, self.start_y, cur_x, self.start_y + height)

    def on_release(self, event):
        pass

    def crop_image(self):
        bbox = self.canvas.coords(self.crop_rect)
        x1, y1, x2, y2 = [int(coord * self.image.width / 600) for coord in bbox]
        self.cropped_image = self.image.crop((x1, y1, x2, y2)).resize(self.target_size, Image.LANCZOS)
        self.destroy()

class StableDiffusionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stable Diffusion Image Generator")
        master.geometry("800x600")

        # Left frame for controls and reference images
        left_frame = ttk.Frame(master)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.prompt_label = ttk.Label(left_frame, text="Prompt:")
        self.prompt_label.pack(pady=5)

        self.prompt_entry = ttk.Entry(left_frame, width=50)
        self.prompt_entry.pack(pady=5)

        self.reference_button = ttk.Button(left_frame, text="Add Reference Image", command=self.add_reference_image)
        self.reference_button.pack(pady=10)

        self.reference_frame = ttk.Frame(left_frame)
        self.reference_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.generate_button = ttk.Button(left_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(left_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.progress_label = ttk.Label(left_frame, text="0%")
        self.progress_label.pack(pady=5)

        self.status_label = ttk.Label(left_frame, text="")
        self.status_label.pack(pady=5)

        self.save_button = ttk.Button(left_frame, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Right frame for generated image
        right_frame = ttk.Frame(master)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(right_frame)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)

        self.base = None
        self.refiner = None
        self.generated_image = None
        self.reference_images = []

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

    def add_reference_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            original_image = Image.open(file_path).convert("RGB")
            crop_window = CropWindow(self.master, original_image)
            self.master.wait_window(crop_window)
            if crop_window.cropped_image:
                self.reference_images.append(crop_window.cropped_image)
                self.display_reference_images()
                self.status_label.config(text=f"{len(self.reference_images)} reference image(s) selected.")

    def display_reference_images(self):
        for widget in self.reference_frame.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.reference_frame, width=300, height=300)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.reference_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        for i, img in enumerate(self.reference_images):
            img_resized = img.copy()
            img_resized.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img_resized)
            label = ttk.Label(inner_frame, image=photo)
            label.image = photo
            label.grid(row=i//3, column=i%3, padx=5, pady=5)

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
                progress = (step / n_steps) * 50
                self.master.after(0, self.update_progress, progress)

            def refiner_callback(step, timestep, latents):
                progress = 50 + (step / n_steps) * 50
                self.master.after(0, self.update_progress, progress)

            if self.reference_images:
                references = [self.base.image_processor.preprocess(img) for img in self.reference_images]
                reference = torch.cat(references, dim=0)
            else:
                reference = None

            latents = self.base(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
                callback=base_callback,
                callback_steps=1,
                image=reference
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
        self.display_image(self.generated_image, self.image_label)
        self.status_label.config(text="Image generated successfully!")
        self.generate_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 100
        self.progress_label.config(text="100%")

    def display_image(self, image, label):
        image = image.resize((400, 400))  # Increased size for the generated image
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

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