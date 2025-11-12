# model/convert_to_onnx.py
# Converts Keras model.h5 -> model.onnx and writes an input.json (NHWC, /255 preprocessing)
import tensorflow as tf
import numpy as np
import tf2onnx
import json
from PIL import Image
import os

MODEL_H5 = "model.h5"
OUT_ONNX = "model.onnx"
INPUT_JSON = "input.json"
IMG_PATH = "sample_image.jpg"

def preprocess_for_export(img_path=None):
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((224,224))
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = np.zeros((224,224,3), dtype=np.float32)
    batched = np.expand_dims(arr, axis=0)  # NHWC - (1, H, W, C)
    return batched

def convert():
    print("Loading model:", MODEL_H5)
    model = tf.keras.models.load_model(MODEL_H5, compile=False)
    sample_input = np.zeros((1,224,224,3), dtype=np.float32)
    spec = (tf.TensorSpec(sample_input.shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(OUT_ONNX, "wb") as f:
        f.write(model_proto.SerializeToString())
    print("Saved ONNX:", OUT_ONNX)

    sample = preprocess_for_export(IMG_PATH).tolist()
    with open(INPUT_JSON, "w") as f:
        json.dump({"input": sample}, f)
    print("Saved input.json:", INPUT_JSON)

if __name__ == "__main__":
    convert()
