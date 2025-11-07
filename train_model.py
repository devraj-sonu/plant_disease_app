from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs('dataset/train/Healthy', exist_ok=True)
os.makedirs('dataset/train/Powdery Mildew', exist_ok=True)
os.makedirs('dataset/train/Leaf Spot', exist_ok=True)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

model.save('model/plant_disease_model.h5')
print("✅ Model saved as model/plant_disease_model.h5")
