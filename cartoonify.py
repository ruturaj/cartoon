# cartoonify.py
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 127.5 - 1  # Normalize
    return np.expand_dims(img, axis=0)

def save_image(img_array, out_path):
    img = ((img_array[0] + 1) * 127.5).astype(np.uint8)
    Image.fromarray(img).save(out_path)

def cartoonify(image_path, model_path, output_path):
    model = tf.saved_model.load(model_path)
    img = load_image(image_path)
    out = model(img)
    save_image(out.numpy(), output_path)

if __name__ == '__main__':
    cartoonify('input_images/sample.jpg', 'saved_model/cartoon', 'output_images/cartoonified.jpg')
