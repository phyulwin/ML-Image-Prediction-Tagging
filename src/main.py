import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import requests
from io import BytesIO
import tempfile
import os

from TestModel import _COCO2017_model, _Food101_model, _ImageNetR_model
from TestModel2 import _HF_Funiture_model
# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------
current_image_path = None  # store local path of last-uploaded image

def run_prediction(img_path):
    mode = model_var.get()
    try:
        if mode == "COCO":
            preds = _COCO2017_model(img_path)
            return ", ".join(preds) if preds else "Image Not Supported"

        elif mode == "Food101":
            name = _Food101_model(img_path)
            return name if name else "Image Not Supported"

        elif mode == "ImageNet-R":
            results = _ImageNetR_model(img_path, topk=5)
            if not results:
                return "Image Not Supported"
            labels = [label for label, _ in results]
            return ", ".join(labels)

        elif mode == "HF-Furniture":
            preds = _HF_Funiture_model(img_path)
            return ", ".join(preds) if preds else "Image Not Supported"
        
        elif mode == "General":
            # Combine predictions from all models
            coco_preds = _COCO2017_model(img_path)
            food101_name = _Food101_model(img_path)
            furniture_preds = _HF_Funiture_model(img_path)
            imagenet_results = _ImageNetR_model(img_path, topk=5)

            # Process ImageNet-R results
            imagenet_labels = [label for label, _ in imagenet_results] if imagenet_results else []

            # Combine all predictions into a single list
            combined_preds = []
            if coco_preds:
                combined_preds.extend(coco_preds)
            if food101_name:
                combined_preds.append(food101_name)
            if furniture_preds:
                combined_preds.extend(furniture_preds)
            if imagenet_labels:
                combined_preds.extend(imagenet_labels)

            # Return combined predictions as a comma-separated string
            return ", ".join(combined_preds) if combined_preds else "Image Not Supported"
        else:
            return "(unknown model)"

    except Exception as e:
        print(f"Model predict error: {e}")
        return "(prediction error)"

def open_image():
    global current_image_path
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if not path:
        return

    current_image_path = path
    url_entry.delete(0, tk.END)

    img = Image.open(path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    caption_label.config(text="Prediction: ")

def predict_image():
    global current_image_path
    url = url_entry.get().strip()

    # 1) Load the image (and display it)
    if url:
        source = url
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        except Exception as e:
            caption_label.config(text=f"Error loading URL: {e}")
            return
    elif current_image_path:
        source = current_image_path
        img = Image.open(source)
    else:
        caption_label.config(text="No image to predict")
        return

    # display it
    img.thumbnail((500, 500))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # 2) Run the model
    caption_label.config(text="Predicting…")
    caption_label.update_idletasks()

    source_for_pred = source
    if source.startswith("http") and model_var.get() != "COCO":
        ext = os.path.splitext(source)[1]
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            ext = ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(resp.content)
        tmp.flush()
        source_for_pred = tmp.name

    prediction = run_prediction(source_for_pred)
    caption_label.config(text=f"Prediction: {prediction}")

# ==========================================
# GUI setup
# ==========================================
try:
    root = tk.Tk()
    root.title("Kelly Lwin - Multi-Model Image Prediction & Tagging")
    root.geometry("1200x700")
    root.resizable(False, False)

    # canvas
    canvas = tk.Canvas(root, width=1200, height=700, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # load background image
    bg_image = Image.open(r"C:\Users\646ca\VSprojects\CPP-Spring2025-CS4200-FinalProject\assets\app_bg.png")
    bg_image = bg_image.resize((1200, 700), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    canvas.create_image(0, 0, anchor="nw", image=bg_photo)
    canvas.bg_photo = bg_photo

    # left and right frame
    left_frame = tk.Frame(canvas, bd=0, highlightthickness=0, bg="white", width=400, height=500)
    canvas.create_window(300, 375, window=left_frame)
    left_frame.pack_propagate(False)

    right_frame = tk.Frame(canvas, bd=0, highlightthickness=0)
    canvas.create_window(800, 350, window=right_frame)

    # set dropdown
    model_var = tk.StringVar(root)
    model_var.set("COCO")
    dropdown = tk.OptionMenu(left_frame, model_var, "COCO", "Food101", "ImageNet-R","HF-Furniture", "General")
    dropdown.config(font=("Consolas", 14), width=15, bg="black", fg="white")
    dropdown.pack(pady=10)

    url_label = tk.Label(left_frame, text="Image URL:", font=("Consolas", 14))
    url_label.pack(pady=(20, 5))
    url_entry = tk.Entry(left_frame, font=("Consolas", 14), width=30)
    url_entry.pack()

    predict_button = tk.Button(
        left_frame,
        text="Predict",
        font=("Consolas", 16),
        bg="black",
        fg="white",
        activebackground="black",
        activeforeground="white",
        command=predict_image,
        width=15,
        height=2,
        bd=0,
        highlightthickness=0,
        relief="flat"
    )
    predict_button.pack(pady=20)

    upload_button = tk.Button(
        left_frame,
        text="Upload Image",
        font=("Consolas", 16),
        bg="black",
        fg="white",
        activebackground="black",
        activeforeground="white",
        command=open_image,
        width=15,
        height=2,
        bd=0,
        highlightthickness=0,
        relief="flat"
    )
    upload_button.pack(pady=20)

    caption_label = tk.Label(
        left_frame,
        text="Predictions: ",
        font=("Consolas", 14),
        wraplength=300,
        justify="left"
    )
    caption_label.pack(pady=20)

    panel = tk.Label(right_frame)
    panel.pack()

    root.mainloop()
except Exception as e:
    print(e)