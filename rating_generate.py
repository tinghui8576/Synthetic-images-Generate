
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk
import os
import pandas as pd
from pathlib import Path
import random


# Create the rating system
class RatingApp:
    def __init__(self, master, image_folder, output_file, total_observations=100):
        self.master = master
        self.master.title("Image Rating System")
        self.image_folder = image_folder
        self.output_file = output_file
        self.total_observations = total_observations  # Total number of observations you want
        self.positive_images = [f for f in os.listdir(image_folder) if f.split('.')[0].isdigit() and int(f.split('.')[0]) > 4]
        self.negative_images = [f for f in os.listdir(image_folder) if f.split('.')[0].isdigit() and int(f.split('.')[0]) <= 4]
        
        # We want exactly 50 positive and 50 negative observations
        self.positive_observations = 50
        self.negative_observations = 50
        
        # Repeat the positive and negative images to achieve 50 observations each
        self.positive_selection = random.choices(self.positive_images, k=self.positive_observations)
        self.negative_selection = random.choices(self.negative_images, k=self.negative_observations)

        # Combine positive and negative selections and shuffle them
        self.all_images = self.positive_selection + self.negative_selection
        random.shuffle(self.all_images)

        self.index = 0
        self.ratings = []
        self.total_images = len(self.all_images)

        # Setup the UI components
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.rating_label = tk.Label(master, text="Rate this image (1-9):", font=("Arial", 14))
        self.rating_label.pack(pady=10)

        # Use a Scale (slider) for discrete ratings
        self.scale = tk.Scale(master, from_=1, to=9, orient=tk.HORIZONTAL, length=300, resolution=1, font=("Arial", 12))
        self.scale.pack(pady=10)

        self.submit_button = tk.Button(master, text="Submit Rating", command=self.save_rating)
        self.submit_button.pack(pady=20)

        self.status_label = tk.Label(master, text="")
        self.status_label.pack()

        # Bind keyboard events (1-9) to control the slider
        for i in range(1, 10):
            self.master.bind(str(i), self.set_slider_value)

        # Bind the Enter key to submit the rating
        self.master.bind('<Return>', self.submit_rating)

        self.load_image()

    def load_image(self):
        """Load the current image and display it."""

        if self.index < self.total_images:
            # Select the next image
            selected_image = self.all_images[self.index]

            image_path = os.path.join(self.image_folder, selected_image)
            img = Image.open(image_path).resize((300, 400))  # Resize the image for display
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            self.status_label.config(text=f"Image {self.index + 1} of {self.total_images}")
            self.index += 1
        else:
            self.save_to_file()
            self.master.quit()

    def save_rating(self):
        """Save the rating for the current image and move to the next one."""
        rating = int(self.scale.get())  # Get discrete rating from Scale
        image_name = self.all_images[self.index - 1]  # Get the current image being rated
        self.ratings.append((image_name, rating)) 
        self.load_image()

    def save_to_file(self):
        """Save all ratings to a CSV file."""
        df = pd.DataFrame(self.ratings, columns=["ImageName", "Rating"])
        df.to_csv(self.output_file, index=False)
        print(f"Ratings saved to {self.output_file}")

    def set_slider_value(self, event):
        """Set the slider value based on key press (1-9)."""
        self.scale.set(int(event.char))

    def submit_rating(self, event):
        """Submit the rating when Enter key is pressed."""
        self.save_rating()

# Create the application window
if __name__ == "__main__":
    image_folder = "figures/generated_faces"  # Replace with the folder containing face images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.csv"
    output_file = filename

    root = tk.Tk()
    app = RatingApp(root, image_folder, output_file)
    root.mainloop()
