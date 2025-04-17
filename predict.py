import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps, ImageEnhance
import cv2
import glob
from tqdm import tqdm
import time

TEST_DIR = r"D:\Project\Olympic AI HCMC\test"  # CHANGLE PATH TO TEST FOLDER

MODEL_PATH = r"D:\Project\Olympic AI HCMC\mushroom_model_best.h5"  # CHANGE PATH TO MODEL


OUTPUT_FILE = "mushroom_predictions.csv"  # Tên file output
IMG_SIZE = 32                             # Kích thước ảnh đầu vào
USE_TTA = False                           # Sử dụng test-time augmentation

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, dropout_rate=0.1, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, patch, training=False):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        encoded = self.dropout(encoded, training=training)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

def gelu(x):
    dtype = x.dtype
    x = tf.cast(x, tf.float32)
    result = 0.5 * x * (1.0 + tf.math.tanh(
        tf.cast(tf.math.sqrt(2.0 / np.pi), tf.float32) * 
        (x + 0.044715 * tf.math.pow(x, 3))
    ))
    return tf.cast(result, dtype)

custom_objects = {
    'Patches': Patches,
    'PatchEncoder': PatchEncoder,
    'gelu': gelu
}
tf.keras.utils.get_custom_objects().update(custom_objects)

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    if minutes > 0:
        return f"{minutes} min {seconds_remainder:.3f} sec"
    else:
        return f"{seconds_remainder:.3f} sec"

def preprocess_image(img_path, target_size=None):
    if target_size is None:
        target_size = (IMG_SIZE, IMG_SIZE)
        
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def apply_test_time_augmentation(model, img, num_augmentations=5):
    if img.max() <= 1.0:
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
    else:
        img_pil = Image.fromarray(img.astype(np.uint8))
    
    aug_images = []

    aug_images.append(img)

    aug_images.append(np.array(ImageOps.mirror(img_pil)) / 255.0)

    aug_images.append(np.array(img_pil.rotate(90)) / 255.0)

    brightness = ImageEnhance.Brightness(img_pil)
    aug_images.append(np.array(brightness.enhance(1.2)) / 255.0)

    contrast = ImageEnhance.Contrast(img_pil)
    aug_images.append(np.array(contrast.enhance(1.2)) / 255.0)
    
    aug_batch = np.stack(aug_images).astype(np.float32)
    
    predictions = model.predict(aug_batch, verbose=0)
    
    avg_prediction = np.mean(predictions, axis=0)
    
    return avg_prediction

def predict_mushrooms():
    start_time = time.time()
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
    
    print(f"Processing images from {TEST_DIR}...")
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    
    if not test_images:
        print(f"No images found in {TEST_DIR}")
        return
    
    test_ids = [os.path.basename(img).split('.')[0] for img in test_images]
    
    X_test = []
    for img_path in tqdm(test_images, desc="Processing test images"):
        img = preprocess_image(img_path)
        if img is not None:
            X_test.append(img)
    
    X_test = np.array(X_test, dtype=np.float32)
    
    print(f"Making predictions on {len(X_test)} images...")
    
    if USE_TTA:
        print("Using test-time augmentation...")
        predictions = np.array([apply_test_time_augmentation(model, img) for img in tqdm(X_test, desc="TTA")])
    else:
        predictions = model.predict(X_test, verbose=1)
    
    pred_classes = np.argmax(predictions, axis=1)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'type': pred_classes
    })
    
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to {OUTPUT_FILE}")
    
    prediction_time = time.time() - start_time
    num_images = len(test_images)
    
    print(f"Prediction time for {num_images} images: {format_time(prediction_time)}")
    print(f"Average time per image: {format_time(prediction_time/num_images)}")
    print(f"Throughput: {num_images/prediction_time:.2f} images/second")
    
    class_counts = pd.Series(pred_classes).value_counts().sort_index()
    class_names = {
        0: "Nấm mỡ",
        1: "Nấm bào ngư",
        2: "Nấm đùi gà",
        3: "Nấm linh chi trắng"
    }
    
    print("\nPrediction results:")
    for class_idx, count in class_counts.items():
        print(f"{class_names.get(class_idx, f'Class {class_idx}')}: {count} images")
    
    return submission_df

if __name__ == "__main__":
    predict_mushrooms()