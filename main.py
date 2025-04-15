import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import cv2
from tensorflow.keras.utils import to_categorical
import random
from tqdm import tqdm
import glob
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates
import time
import gc
import json
from datetime import datetime
import argparse
from pathlib import Path

os.system("chcp 65001")  # Set console to UTF-8 for Vietnamese characters

def parse_args():
    parser = argparse.ArgumentParser(description='Train a mushroom classification model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, 
                        help='Use mixed precision (default: False)')
    parser.add_argument('--k_fold', type=int, default=0, help='Number of folds for cross-validation (0 to disable)')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--tta', action='store_true', default=False, 
                        help='Use test-time augmentation (default: False)')
    parser.add_argument('--ensemble', type=int, default=1, help='Number of models for ensemble')
    return parser.parse_args()

args = argparse.Namespace(
    seed=42,
    img_size=32,
    batch_size=32,
    epochs=50,
    learning_rate=1e-4,
    val_split=0.2,
    model_size='base',
    use_mixed_precision=False,
    k_fold=0,
    augment=True,
    tta=False,
    ensemble=1
)

SEED = args.seed
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

BASE_DIR = '.'  #REPLACE PATH IN HERE (Your current directory)
TRAIN_DIR = os.path.join(BASE_DIR, 'train')  #(Name folder)
TEST_DIR = os.path.join(BASE_DIR, 'test')  #(Name folder)
CSV_FILE = os.path.join(BASE_DIR, 'csv', 'mushroom_labels.csv')  #(Name folder + Name file .csv)
OUTPUT_DIR = BASE_DIR

config = {
    'seed': SEED,
    'img_size': args.img_size,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'val_split': args.val_split,
    'model_size': args.model_size,
    'use_mixed_precision': args.use_mixed_precision,
    'k_fold': args.k_fold,
    'augment': args.augment,
    'tta': args.tta,
    'ensemble': args.ensemble
}

MODEL_CONFIGS = {
    'tiny': {
        'patch_size': 8,
        'num_heads': 4,
        'transformer_layers': 4,
        'projection_dim': 64,
        'mlp_units': [128, 64],
        'dropout_rate': 0.1
    },
    'small': {
        'patch_size': 4,
        'num_heads': 6,
        'transformer_layers': 5,
        'projection_dim': 128,
        'mlp_units': [256, 128],
        'dropout_rate': 0.1
    },
    'base': {
        'patch_size': 4,
        'num_heads': 8,
        'transformer_layers': 6,
        'projection_dim': 48,
        'mlp_units': [96, 48],
        'dropout_rate': 0.1
    },
    'large': {
        'patch_size': 4,
        'num_heads': 12,
        'transformer_layers': 8,
        'projection_dim': 256,
        'mlp_units': [512, 256],
        'dropout_rate': 0.2
    }
}

model_config = MODEL_CONFIGS[args.model_size]

IMG_SIZE = args.img_size
NUM_CLASSES = 4
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
VALIDATION_SPLIT = args.val_split
PATCH_SIZE = model_config['patch_size']
NUM_HEADS = model_config['num_heads']
TRANSFORMER_LAYERS = model_config['transformer_layers']
PROJECTION_DIM = model_config['projection_dim']
MLP_UNITS = model_config['mlp_units']
DROPOUT_RATE = model_config['dropout_rate']

if args.use_mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled with policy:", policy)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    if minutes > 0:
        return f"{minutes} min {seconds_remainder:.3f} sec"
    else:
        return f"{seconds_remainder:.3f} sec"

def load_labels():
    try:
        labels_df = pd.read_csv(CSV_FILE)
        print(f"Loaded {len(labels_df)} labeled images from CSV file")
        return labels_df
    except Exception as e:
        print(f"Error loading labels: {e}")
        raise

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

def elastic_transform(image, alpha=36, sigma=4):
    random_state = np.random.RandomState(None)
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.shape[-1] != 3 and len(image.shape) == 3:
        image = np.stack([image] * 3, axis=2)
    
    shape = image.shape
    
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    distorted_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1).reshape(shape[:2])
    
    distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
    
    return distorted_image

def mixup_images(img1, img2, alpha=0.2):
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    
    return mixed_img

def cutmix_images(img1, img2, cut_size=None):
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    h, w = img1.shape[0], img1.shape[1]
    if cut_size is None:
        cut_size = min(h, w) // 2
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = max(cx - cut_size // 2, 0)
    y1 = max(cy - cut_size // 2, 0)
    x2 = min(cx + cut_size // 2, w)
    y2 = min(cy + cut_size // 2, h)
    
    mixed_img = img1.copy()
    mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    return mixed_img

def augment_image_enhanced(img, file_prefix=None, save_dir=None, all_images=None):
    augmented_images = []
    augmented_paths = []
    
    if isinstance(img, np.ndarray):
        if img.max() <= 1.0:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(img.astype(np.uint8))
    else:
        img_pil = img
    
    for angle in [90, 180, 270]:
        rotated = img_pil.rotate(angle)
        if save_dir and file_prefix:
            save_path = os.path.join(save_dir, f"{file_prefix}_rot{angle}.jpg")
            rotated.save(save_path)
            augmented_paths.append(save_path)
        augmented_images.append(np.array(rotated) / 255.0)
    
    flipped_h = ImageOps.mirror(img_pil)
    flipped_v = ImageOps.flip(img_pil)
    if save_dir and file_prefix:
        flipped_h.save(os.path.join(save_dir, f"{file_prefix}_fliph.jpg"))
        flipped_v.save(os.path.join(save_dir, f"{file_prefix}_flipv.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_fliph.jpg"),
            os.path.join(save_dir, f"{file_prefix}_flipv.jpg")
        ])
    augmented_images.extend([np.array(flipped_h) / 255.0, np.array(flipped_v) / 255.0])
    
    brightness_enhancer = ImageEnhance.Brightness(img_pil)
    brightened = brightness_enhancer.enhance(1.5)
    darkened = brightness_enhancer.enhance(0.7)
    if save_dir and file_prefix:
        brightened.save(os.path.join(save_dir, f"{file_prefix}_bright.jpg"))
        darkened.save(os.path.join(save_dir, f"{file_prefix}_dark.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_bright.jpg"),
            os.path.join(save_dir, f"{file_prefix}_dark.jpg")
        ])
    augmented_images.extend([np.array(brightened) / 255.0, np.array(darkened) / 255.0])
    
    contrast_enhancer = ImageEnhance.Contrast(img_pil)
    increased_contrast = contrast_enhancer.enhance(1.5)
    decreased_contrast = contrast_enhancer.enhance(0.7)
    if save_dir and file_prefix:
        increased_contrast.save(os.path.join(save_dir, f"{file_prefix}_contrast_high.jpg"))
        decreased_contrast.save(os.path.join(save_dir, f"{file_prefix}_contrast_low.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_contrast_high.jpg"),
            os.path.join(save_dir, f"{file_prefix}_contrast_low.jpg")
        ])
    augmented_images.extend([np.array(increased_contrast) / 255.0, np.array(decreased_contrast) / 255.0])
    
    combined = ImageOps.mirror(img_pil.rotate(90))
    if save_dir and file_prefix:
        combined.save(os.path.join(save_dir, f"{file_prefix}_rot90_flip.jpg"))
        augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_rot90_flip.jpg"))
    augmented_images.append(np.array(combined) / 255.0)
    
    noisy_img = img_pil.copy()
    noisy_array = np.array(noisy_img)
    noise = np.random.normal(0, 15, noisy_array.shape).astype(np.uint8)
    noisy_array = np.clip(noisy_array + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_array)
    if save_dir and file_prefix:
        noisy_img.save(os.path.join(save_dir, f"{file_prefix}_noise.jpg"))
        augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_noise.jpg"))
    augmented_images.append(np.array(noisy_img) / 255.0)
    
    if all_images is not None and len(all_images) > 1:
        img_array = np.array(img_pil)
        
        random_idx = random.randint(0, len(all_images) - 1)
        random_img = all_images[random_idx]
        
        mixup_img = mixup_images(img, random_img, alpha=0.2)
        if save_dir and file_prefix:
            mixup_img_pil = Image.fromarray((mixup_img * 255).astype(np.uint8))
            mixup_img_pil.save(os.path.join(save_dir, f"{file_prefix}_mixup.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_mixup.jpg"))
        augmented_images.append(mixup_img)
        
        cutmix_img = cutmix_images(img, random_img)
        if save_dir and file_prefix:
            cutmix_img_pil = Image.fromarray((cutmix_img * 255).astype(np.uint8))
            cutmix_img_pil.save(os.path.join(save_dir, f"{file_prefix}_cutmix.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_cutmix.jpg"))
        augmented_images.append(cutmix_img)
        
        elastic_img = elastic_transform(img_array)
        if save_dir and file_prefix:
            elastic_img_pil = Image.fromarray(elastic_img)
            elastic_img_pil.save(os.path.join(save_dir, f"{file_prefix}_elastic.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_elastic.jpg"))
        augmented_images.append(np.array(elastic_img) / 255.0)
    
    return augmented_images, augmented_paths

def create_data_generator(images, labels, batch_size=BATCH_SIZE, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(images), seed=SEED)
    
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_and_augment_data_enhanced(labels_df, augment=True, cache_folder=None):
    X = []
    y = []
    all_images = []
    
    id_to_label = dict(zip(labels_df['id'], labels_df['type']))
    
    folders = ['bao ngu xam trang', 'dui ga baby', 'linh chi trang', 'nam mo']
    folder_to_class = {
        'bao ngu xam trang': 1,
        'dui ga baby': 2,
        'linh chi trang': 3,
        'nam mo': 0
    }
    
    for folder in folders:
        folder_path = os.path.join(TRAIN_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
            
        print(f"Processing folder: {folder}")
        images = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        for img_path in tqdm(images, desc=f"Loading {folder}"):
            img_id = os.path.basename(img_path).split('.')[0]
            
            if img_id not in id_to_label:
                print(f"Warning: Image ID {img_id} not found in labels CSV")
                continue
                
            img = preprocess_image(img_path)
            if img is None:
                continue
                
            X.append(img)
            y.append(id_to_label[img_id])
            all_images.append(img)
    
    if augment:
        print("Applying enhanced augmentations...")
        
        aug_X = []
        aug_y = []
        
        for i, (img, label) in enumerate(tqdm(zip(all_images, list(y)), total=len(all_images), desc="Augmenting")):
            matching_ids = [id for id, lbl in id_to_label.items() if lbl == label]
            if matching_ids:
                img_id = matching_ids[0]
                aug_imgs, _ = augment_image_enhanced(img, img_id, all_images=all_images)
                aug_X.extend(aug_imgs)
                aug_y.extend([label] * len(aug_imgs))
            
            if i % 100 == 0 and i > 0:
                X.extend(aug_X)
                y.extend(aug_y)
                aug_X = []
                aug_y = []
                gc.collect()
        
        X.extend(aug_X)
        y.extend(aug_y)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    y = to_categorical(y, num_classes=NUM_CLASSES)
    
    print(f"Final dataset shape: X: {X.shape}, y: {y.shape}")
    
    return X, y

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

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
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
    def __init__(self, num_patches, projection_dim, dropout_rate=0.1):
        super(PatchEncoder, self).__init__()
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

def build_vit_model():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_patches = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE)
    
    inputs = layers.Input(shape=input_shape)
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    augmented = inputs
    
    patches = Patches(PATCH_SIZE)(augmented)
    
    encoded_patches = PatchEncoder(num_patches, PROJECTION_DIM, DROPOUT_RATE)(patches)

    x = encoded_patches
    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=PROJECTION_DIM // NUM_HEADS,
            dropout=DROPOUT_RATE
        )(x1, x1)
        
        x2 = layers.Add()([attention_output, x])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x4 = layers.Dense(MLP_UNITS[0], activation='relu')(x3)
        x4 = layers.Dropout(DROPOUT_RATE)(x4)
        x4 = layers.Dense(MLP_UNITS[1], activation='relu')(x4)
        x4 = layers.Dropout(DROPOUT_RATE)(x4)
        
        x = layers.Add()([x4, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    representation = layers.Dropout(0.3)(representation)
    
    features = layers.Dense(256, activation='relu')(representation)
    features = layers.Dropout(0.3)(features)
    features = layers.Dense(128, activation='relu')(features)
    
    outputs = layers.Dense(
        NUM_CLASSES, 
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT_Mushroom_Classifier")
    
    return model

def train_model(X_train, y_train, X_val, y_val, fold=None):
    start_time = time.time()
    
    model_save_path = os.path.join(OUTPUT_DIR, "mushroom_model_best.h5")
    
    model = build_vit_model()
    
    train_dataset = create_data_generator(X_train, y_train, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_data_generator(X_val, y_val, batch_size=BATCH_SIZE, is_training=False)
    
    if args.use_mixed_precision:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    model.save(model_save_path)
    
    training_time = time.time() - start_time
    print(f"Total training time: {format_time(training_time)}")
    
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    return model, history

def evaluate_model(model, X_test, y_test, history=None, fold=None):
    start_time = time.time()
    
    test_dataset = create_data_generator(X_test, y_test, batch_size=BATCH_SIZE, is_training=False)
    
    print("Evaluating model...")
    results = model.evaluate(test_dataset, verbose=1)
    
    metrics = model.metrics_names
    metrics_values = dict(zip(metrics, results))
    
    print(f"Test metrics: {metrics_values}")
    
    if args.tta:
        print("Applying test-time augmentation...")
        y_pred = np.array([apply_test_time_augmentation(model, img) for img in tqdm(X_test, desc="TTA")])
    else:
        y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    print(f"Macro F1 score: {f1:.4f}")
    
    class_report = classification_report(y_true_classes, y_pred_classes)
    print(f"\nClassification Report:\n{class_report}")
    
    evaluation_time = time.time() - start_time
    print(f"Evaluation time: {format_time(evaluation_time)}")
    
    return metrics_values, y_pred_classes

def train_kfold(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    y_indices = np.argmax(y, axis=1)
    
    fold_metrics = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_indices)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
        
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        metrics, _ = evaluate_model(model, X_val, y_val, history)
        
        fold_metrics.append(metrics)
        fold_models.append(model)
        
        del model, history
        gc.collect()
        tf.keras.backend.clear_session()
    
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
    
    print(f"\nAverage metrics across {n_folds} folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    best_fold_idx = np.argmax([fold['accuracy'] for fold in fold_metrics])
    print(f"Best model is from fold {best_fold_idx+1}")
    
    return fold_models[best_fold_idx], fold_metrics

def predict_test_dataset(model, save_predictions=True):
    start_time = time.time()
    
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    test_ids = [os.path.basename(img).split('.')[0] for img in test_images]
    
    X_test_submit = []
    for img_path in tqdm(test_images, desc="Processing test images"):
        img = preprocess_image(img_path)
        if img is not None:
            X_test_submit.append(img)
    
    X_test_submit = np.array(X_test_submit, dtype=np.float32)
    
    print("Making predictions...")
    
    if args.tta:
        print("Using test-time augmentation...")
        predictions = np.array([apply_test_time_augmentation(model, img) for img in tqdm(X_test_submit, desc="TTA")])
    else:
        predictions = model.predict(X_test_submit, verbose=1)
    
    pred_classes = np.argmax(predictions, axis=1)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'type': pred_classes
    })
    
    if save_predictions:
        submission_file = os.path.join(OUTPUT_DIR, 'mushroom_predictions.csv')
        submission_df.to_csv(submission_file, index=False)
        print(f"Saved predictions to {submission_file}")
    
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

def main():
    total_start_time = time.time()
    
    print("="*50)
    print("Mushroom Classification with Vision Transformer")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"Mixed precision enabled: {args.use_mixed_precision}")
    print(f"Model size: {args.model_size}")
    print(f"Configuration: {config}")
    print("="*50)
    
    labels_df = load_labels()
    
    data_start_time = time.time()
    X, y = load_and_augment_data_enhanced(labels_df, augment=args.augment)
    data_time = time.time() - data_start_time
    print(f"Time to load and augment data: {format_time(data_time)}")
    
    if args.k_fold > 1:
        print(f"\n{'='*20} K-Fold Cross-Validation (k={args.k_fold}) {'='*20}")
        best_model, fold_metrics = train_kfold(X, y, n_folds=args.k_fold)
        
        print("\n=== Making Predictions on Test Dataset ===")
        predict_test_dataset(best_model)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=np.argmax(y, axis=1)
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Validation set size: {X_val.shape}")
        
        print("\n=== Training Vision Transformer Model ===")
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        print("\n=== Evaluating Vision Transformer Model ===")
        evaluate_model(model, X_val, y_val, history)
        
        print("\n=== Making Predictions on Test Dataset ===")
        predict_test_dataset(model)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {format_time(total_time)}")
    print(f"Process completed successfully!")

if __name__ == "__main__":
    main()