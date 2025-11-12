# server/utils_preprocess.py
# Preprocessing matching the notebook: rescale=1./255, NHWC
import numpy as np
from PIL import Image
import io, base64

def preprocess_base64_image_to_nhwc(b64data):
    if "," in b64data:
        b64data = b64data.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img).astype('float32') / 255.0
    batched = np.expand_dims(arr, axis=0)  # (1,224,224,3)
    return batched
