import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
import os, io, requests
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
from io import BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
#=========================================================
# Model - COCO-2017 80-Category
#=========================================================
MODEL_PATH = r"C:\Users\646ca\VSprojects\CPP-Spring2025-CS4200-FinalProject\src\model\modelPhase1.h5"

# COCO-2017’s 80 class names
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

def coco80_predict(img_source: str,
                   model_path: str = MODEL_PATH,
                   threshold: float = 0.5,
                   _cache: dict = {}) -> list:
    # 1) load & cache model
    if model_path not in _cache:
        _cache[model_path] = load_model(model_path, compile=False)
    model = _cache[model_path]

    # 2) open image (URL or local)
    if img_source.lower().startswith(("http://", "https://")):
        r = requests.get(img_source)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
    else:
        img = Image.open(img_source)
    img = img.convert("RGB").resize((256, 256))

    # 3) preprocess & predict
    arr = np.expand_dims(np.asarray(img, np.float32) / 255.0, 0)
    probs = model.predict(arr, verbose=0)[0]

    # 4) decode and return
    return [COCO_CLASSES[i] for i, p in enumerate(probs) if p > threshold]

def coco80_predict_multi(
    img_source: str,
    model_path: str = MODEL_PATH,
    threshold: float = 0.5,
    _cache: dict = {}
) -> list:
    """
    Load (and cache) the COCO model, run a forward pass on the image,
    and return the top-5 class names as a list.
    """
    # 1) load & cache model
    if model_path not in _cache:
        _cache[model_path] = load_model(model_path)
    model = _cache[model_path]

    # 2) open image (URL or local)
    if img_source.lower().startswith(("http://", "https://")):
        r = requests.get(img_source)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
    else:
        img = Image.open(img_source)
    img = img.convert("RGB").resize((256, 256))

    # 3) preprocess & predict
    arr = np.expand_dims(np.asarray(img, np.float32) / 255.0, 0)
    probs = model.predict(arr, verbose=0)[0]

    # 4) select top-5 indices and return their class names
    top_idxs = probs.argsort()[::-1][:5]
    return [COCO_CLASSES[i] for i in top_idxs]

def _COCO2017_model(source: str, threshold: float = 0.5) -> list:
    """
    Single-image wrapper around coco80_predict_multi.
    Returns the top-5 labels (possibly empty list).
    """
    try:
        # always use the multi-class top-5
        preds = coco80_predict_multi(source, threshold=threshold)
        print(f"[COCO] {source} -> {preds}")
        return preds or []
    except Exception as e:
        print(f"Error in _COCO2017_model: {e}")
        return []

# try:
#     IMAGE = r"https://i.pinimg.com/originals/0d/06/a6/0d06a65937745ab9b7b780f172b00e64.jpg"
#     _COCO2017_model(IMAGE)
# except Exception as e:
#     print(e)
#=========================================================
# Model - Food-101
#=========================================================
IMG_SIZE = (128, 128)
NUM_CLASSES = 101

# Food-101 class names
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
    'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
    'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
    'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
    'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta',
    'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

# Load your model from the correct path
model = tf.keras.models.load_model(
    r"C:\Users\646ca\VSprojects\CPP-Spring2025-CS4200-FinalProject\src\model\modelPhase3.h5"
)

def predict_image(image_source):
    try:
        # Load image from URL or local path
        if image_source.startswith(('http://', 'https://')):
            resp = requests.get(image_source)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            print("Loaded image from URL.")
        else:
            img = Image.open(image_source).convert('RGB')
            print("Loaded image from local path.")

        # Preprocess
        img_resized = img.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_batch)
        idx = np.argmax(preds, axis=1)[0]
        return idx, class_names[idx]

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def _Food101_model(image_source):
    try:
        predicted_class, predicted_class_name = predict_image(image_source)

        if predicted_class is not None:
            print(f"Predicted class ID:   {predicted_class}")
            print(f"Predicted class name: {predicted_class_name}")
        else:
            print("Prediction failed.")
        return predicted_class_name # returns one class name
    except Exception as e:
        print(e)

# try:
#     image = 'https://th.bing.com/th/id/OIP.5EJUcpJstOrfXzgRe92G6wHaL_?rs=1&pid=ImgDetMain'
#     prediction = _Food101_model(image)
# except Exception as e:
#     print(f"Error processing image: {e}")
#=========================================================
# Model - ImageNet-R 200-Category
#=========================================================
def load_imagenet_r_model(
    readme_path: str,
    model_path: str,
    img_size: tuple = (128, 128),
    device: torch.device = None,
):
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transforms
    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load synset → human label mapping
    def load_synset_mapping(path):
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if parts and parts[0].startswith("n0"):
                    mapping[parts[0]] = parts[1]
        return mapping

    syn2human = load_synset_mapping(readme_path)

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # Figure out how many classes we have
    num_classes = checkpoint['classifier.1.weight'].shape[0]
    sorted_synsets = sorted(syn2human.keys())
    if len(sorted_synsets) != num_classes:
        print(f"Warning: README lists {len(sorted_synsets)} synsets "
              f"but model has {num_classes} classes.")

    # Build idx → label
    idx_to_label = {}
    for i in range(num_classes):
        if i < len(sorted_synsets):
            idx_to_label[i] = syn2human.get(sorted_synsets[i],
                                           sorted_synsets[i])
        else:
            idx_to_label[i] = f"class_{i}"

    # Instantiate the model architecture
    model = mobilenet_v2(pretrained=False)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, num_classes)
    )
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # The function you’ll call to predict on one image
    def _model_(path_or_url: str, topk: int = 5):
        # Load image from URL or local path
        if path_or_url.startswith(("http://", "https://")):
            resp = requests.get(path_or_url)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")

        # Preprocess and forward
        tensor = tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.topk(topk, dim=1)

        # Format results
        top_probs = top_probs.cpu().squeeze().tolist()
        top_idxs  = top_idxs.cpu().squeeze().tolist()
        if topk == 1:
            top_probs = [top_probs]
            top_idxs  = [top_idxs]

        # Print & return
        print(f"Top {topk} predictions:")
        for rank, (idx, p) in enumerate(zip(top_idxs, top_probs), start=1):
            print(f"  {rank}. {idx_to_label[idx]:20s} {p*100:6.2f}%")

        return [(idx_to_label[idx], p) for idx, p in zip(top_idxs, top_probs)]

    return _model_

_ImageNetR_model = load_imagenet_r_model(
    readme_path = r"C:\Users\646ca\VSprojects\CPP-Spring2025-CS4200-FinalProject\assets\README.txt",
    model_path  = r"C:\Users\646ca\VSprojects\CPP-Spring2025-CS4200-FinalProject\src\model\modelPhase5__v2.h5",
    img_size    = (128, 128)
)
# try:
#     IMAGE_SOURCE = "https://img.freepik.com/premium-photo/two-happy-dogs-illustration_762761-164.jpg"
#     prediction = _ImageNetR_model(IMAGE_SOURCE, topk=5)
# except Exception as e:
#     print(e)