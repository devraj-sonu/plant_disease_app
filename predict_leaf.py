from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

model = load_model('model/plant_disease_model.h5')
class_names = ['Healthy', 'Powdery Mildew', 'Leaf Spot']

with open('disease_info.json') as f:
    disease_info = json.load(f)

img = Image.open('image.jpg').resize((128, 128))
img_arr = np.array(img).reshape((1, 128, 128, 3)) / 255.0

y_pred = model.predict(img_arr)
pred_label = class_names[np.argmax(y_pred)]
pred_score = float(np.max(y_pred))
remedy = disease_info[pred_label]['remedy']

print(f"Predicted: {pred_label} (Confidence: {pred_score:.2f})")
print(f"Remedy: {remedy}")
