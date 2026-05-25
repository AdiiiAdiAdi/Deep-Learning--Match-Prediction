import tensorflow as tf
import os
import sys

# Need to import customs for load_model to work
MODELS_DIR = 'src/models'
# Use the same imports as app.py
sys.path.append('src')
from train import AttentionLayer, focal_loss

keras_path = os.path.join(MODELS_DIR, 'best_model.keras')
h5_path    = os.path.join(MODELS_DIR, 'best_model.h5')
model_path = keras_path if os.path.exists(keras_path) else h5_path

model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'AttentionLayer': AttentionLayer,
        'focal_loss'    : focal_loss,
    }
)

for i, inp in enumerate(model.inputs):
    print(f"Input {i}: {inp.name} - shape {inp.shape}")
