# model/run_inference.py
# Usage:
#   python run_inference.py --image_path /path/to/image.jpg
# or
#   cat image_base64.txt | python run_inference.py --stdin 1

import argparse
import base64
import io
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import json

MODEL_PATH = "model.h5"  # place your model.h5 here (same as notebook)

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image_pil(img_pil):
    # Notebook: ImageDataGenerator(rescale=1./255), and target_size=(224,224)
    img = img_pil.convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0  # NHWC (1,224,224,3)
    batched = np.expand_dims(arr, axis=0)
    return batched

def inference_from_base64(b64str):
    if b"," in b64str:
        b64str = b64str.split(b",",1)[1]
    data = base64.b64decode(b64str)
    img = Image.open(io.BytesIO(data))
    x = preprocess_image_pil(img)
    model = load_model()
    p = float(model.predict(x)[0][0])  # sigmoid output
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="path to image file (jpg/png)")
    parser.add_argument("--stdin", type=int, default=0, help="read base64 image from stdin if 1")
    args = parser.parse_args()
    if args.stdin == 1:
        b64 = sys.stdin.buffer.read()
        p = inference_from_base64(b64)
    elif args.image_path:
        with open(args.image_path, "rb") as f:
            b64 = base64.b64encode(f.read())
            p = inference_from_base64(b64)
    else:
        print("Provide --image_path or --stdin 1", file=sys.stderr)
        sys.exit(2)
    # Print JSON output with probability p
    print(json.dumps({"p": p}))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
