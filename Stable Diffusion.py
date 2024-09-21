import tkinter as tk
from tkinter import ttk, filedialog
from diffusers import DiffusionPipeline
import threading
from PIL import Image, ImageTk
import torch
import imageSettings

class CropWindow(tk.Toplevel):
    """
    A window for cropping an image to a specific size.

    Attributes:
        master (tkinter.Tk): The parent window.
        image (PIL.Image): The image to be cropped.
        target_size (tuple): The size to which the image should be cropped.
    """

    def __init__(self, master, image, target_size=(imageSettings.image_width, imageSettings.image_height)):
        """
        Initializes the CropWindow.

        Args:
            master (tkinter.Tk): The parent window.
            image (PIL.Image): The image to be cropped.
            target_size (tuple, optional): The target size for cropping. Defaults to the imageSettings.
        """
        super().__init__(master)
        self.title("Crop Image")
        self.image = image
        self.target_size = target_size
        self.cropped_image = None

        # Calculate the scaling factor to fit the image within 600x600
        scale = min(600 / self.image.width, 600 / self.image.height)
        self.display_width = int(self.image.width * scale)
        self.display_height = int(self.image.height * scale)

        self.canvas = tk.Canvas(self, width=self.display_width, height=self.display_height)
        self.canvas.pack()

        self.display_image = ImageTk.PhotoImage(self.image.resize((self.display_width, self.display_height), Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)

        self.crop_rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="red")

        # Bind mouse events for cropping
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.crop_button = ttk.Button(self, text="Crop", command=self.crop_image, state=tk.DISABLED)
        self.crop_button.pack(pady=10)

    def on_press(self, event):
        """Handles mouse press event to start cropping."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.crop_button.config(state=tk.DISABLED)

    def on_drag(self, event):
        """Handles mouse drag event to update the crop rectangle."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        w, h = self.target_size
        aspect_ratio = w / h

        width = cur_x - self.start_x
        height = width / aspect_ratio

        self.canvas.coords(self.crop_rect, self.start_x, self.start_y, cur_x, self.start_y + height)

    def on_release(self, event):
        """Handles mouse release event to enable the crop button if the crop rectangle is valid."""
        if self.canvas.coords(self.crop_rect) != (0, 0, 0, 0):
            self.crop_button.config(state=tk.NORMAL)

    def crop_image(self):
        """Crops the image based on the defined rectangle and closes the window."""
        bbox = self.canvas.coords(self.crop_rect)
        x1, y1, x2, y2 = [int(coord * self.image.width / 600) for coord in bbox]
        self.cropped_image = self.image.crop((x1, y1, x2, y2)).resize(self.target_size, Image.LANCZOS)
        self.destroy()


class StableDiffusionGUI:
    """
    The main GUI class for the Stable Diffusion image generator application.

    Attributes:
        master (tkinter.Tk): The main application window.
        base (DiffusionPipeline): The base model for image generation.
        refiner (DiffusionPipeline): The refiner model for image refinement.
        generated_image (PIL.Image): The generated image.
        reference_images (list): List of reference images.
    """

    def open_settings(self):
        """Opens a new settings window when the settings button is clicked."""
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Settings")
        settings_window.geometry("400x200")
        settings_window.resizable(False, False)

        # Placeholder for settings content
        ttk.Label(settings_window, text="Settings will be added here.").pack(pady=20)

    def __init__(self, master):
        """
        Initializes the main GUI.

        Args:
            master (tkinter.Tk): The main application window.
        """
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

        # Right frame for generated image and save button
        right_frame = ttk.Frame(master)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(right_frame)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)

        # Save button in the right frame
        self.save_button = ttk.Button(right_frame, text="Save Generated Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=10, side=tk.BOTTOM)

        # Settings button
        self.settings_button = ttk.Button(left_frame, text="Settings", command=self.open_settings)
        self.settings_button.pack(pady=10)

        self.base = None
        self.refiner = None
        self.generated_image = None
        self.reference_images = []

    def load_models(self):
        """Loads the Stable Diffusion models into memory."""
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
        """Adds a reference image to the GUI."""
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
        """Displays the reference images in the reference image frame."""
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
            label.grid(row=i // 3, column=i % 3, padx=5, pady=5)

    def generate_image(self):
        """Generates an image based on the prompt and reference images."""
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
            """Thread function to generate the image."""
            def base_callback(step, timestep, latents):
                """Callback for the base model."""
                progress = (step / imageSettings.num_inference_steps) * 100 * imageSettings.high_noise_frac
                self.master.after(0, self.update_progress, progress)

            def refiner_callback(step, timestep, latents):
                """Callback for the refiner model."""
                progress = 100 * imageSettings.high_noise_frac + (step / imageSettings.num_inference_steps) * 100 * (1 - imageSettings.high_noise_frac)
                self.master.after(0, self.update_progress, progress)

            if self.reference_images:
                reference = self.reference_images[0].resize((1024, 1024))
                reference = self.base.image_processor.preprocess(reference)
            else:
                reference = None

            latents = self.base(
                prompt=prompt,
                num_inference_steps=imageSettings.num_inference_steps,
                denoising_end=imageSettings.high_noise_frac,
                output_type="latent",
                callback=base_callback,
                callback_steps=1,
                image=reference,
                strength=imageSettings.referance_strength
            ).images

            images = self.refiner(
                prompt=prompt,
                num_inference_steps=imageSettings.num_inference_steps,
                denoising_start=imageSettings.high_noise_frac,
                image=latents,
                callback=refiner_callback,
                callback_steps=1
            ).images

            self.generated_image = images[0]
            self.master.after(0, self.update_gui)

        threading.Thread(target=generate, daemon=True).start()

    def update_progress(self, progress):
        """Updates the progress bar and label."""
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"{progress:.1f}%")
        self.master.update_idletasks()

    def update_gui(self):
        """Updates the GUI with the generated image."""
        self.display_image(self.generated_image, self.image_label)
        self.status_label.config(text="Image generated successfully!")
        self.generate_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 100
        self.progress_label.config(text="100%")

    def display_image(self, image, label):
        """Displays an image in a label."""
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def save_image(self):
        """Saves the generated image to a file."""
        if self.generated_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.generated_image.save(file_path)
                self.status_label.config(text=f"Image saved to: {file_path}")
        else:
            self.status_label.config(text="No image to save.")

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = StableDiffusionGUI(root)
    root.mainloop()
