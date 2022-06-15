from keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab
import numpy as np
import os
import cv2


# Select a model from the model folders
model = load_model(os.getcwd() + "/model/model_final.h5")

char_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 
             11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 
             21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}


def process_img(image):
    # # Resize image to 28x28 pixels
    # img = image.resize((28, 28))

    # # Convert RGB to grayscale
    # img_processed = img.convert("L")
    # img_final = np.array(img_processed)

    # # Reshape to support model input
    # img_final = img_final.reshape(1, 28, 28, 1)

    # Read the screenshot as a grayscale image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    img_processed = cv2.GaussianBlur(img, (7,7), 0)

    # Apply threshold to remove noises from the image
    _, img_thresh = cv2.threshold(img_processed, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Reshape to support model input
    img_final = cv2.resize(img_thresh, (28,28))
    img_final = np.reshape(img_final, (1,28,28,1))

    return img_final

def predict_character(img):
    # Predict the class
    # print(img.argmax())
    # Return no prediction in case the image is blank
    if img.argmax() == 0:
        return None, None

    results = model.predict(img)[0]

    # Get the highest probability value and its index
    char, prob = np.argmax(results), max(results)

    return char, prob


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Create elements
        self.title("Handwritten Character Detector")
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a Character", font=("Helvetica", 24))
        self.btn_classify = tk.Button(self, text="Predict", command=self.classify_character)
        self.btn_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=5, sticky=tk.W, padx=5)
        self.label.grid(row=0, column=1, pady=5, padx=5)
        self.btn_classify.grid(row=1, column=1, pady=5, padx=5)
        self.btn_clear.grid(row=1, column=0, pady=5)

        # Bind LMB to draw_lines function
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_character(self):
        # Get the handle of the canvas
        handle = self.canvas.winfo_id()

        # Get the coordinate of the canvas
        canvas_coords = win32gui.GetWindowRect(handle)

        # Take a screenshot of the canvas
        img = ImageGrab.grab(canvas_coords)

        # Predict the character then display the prediction and the probability of the image being that character
        character, probability = predict_character(process_img(img))
        
        if character is not None and probability is not None:
            self.label.configure(text=f"Character: {char_dict[character]}\nProbability: {round(probability * 100, 2)}%")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 10 # Brush thickness

        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


if __name__ == "__main__":
    app = App()
    tk.mainloop()
