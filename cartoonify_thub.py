import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB").resize((256, 256))
    img = np.asarray(img) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def save_image(tensor, out_path):
    img = np.squeeze(tensor) * 255.0
    img = Image.fromarray(np.uint8(img))
    img.save(out_path)

def cartoonify(image_path, output_path):
    model = hub.load("https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1")
    image_tensor = load_image(image_path)
    output_tensor = model(image_tensor)
    save_image(output_tensor, output_path)

if __name__ == "__main__":
    cartoonify("input_images/sample.jpg", "output_images/cartoonified.jpg")
