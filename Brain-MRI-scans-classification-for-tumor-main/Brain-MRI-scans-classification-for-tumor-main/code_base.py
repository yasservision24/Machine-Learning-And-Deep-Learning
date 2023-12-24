import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import normalize  
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from google.colab import files

# Upload a single file
uploaded = files.upload()

# Access the uploaded file
for filename, content in uploaded.items():
    print(f"Uploaded file: {filename}, File content: {content[:100]}")  # Print the first 100 characters of the file content


!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ahmedhamada0/brain-tumor-detection

!unzip /content/brain-tumor-detection.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
import os

positive=os.listdir("/content/yes")
negative=os.listdir("/content/no")
print(positive)
print(negative)

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

features = []
labels = []

positive_dir = "/content/yes"
negative_dir = "/content/no"

for j in os.listdir(negative_dir):
    img_path = os.path.join(negative_dir, j)
    img = cv2.imread(img_path)
    img = Image.fromarray(img, "RGB")
    img = img.resize((64, 64))
    features.append(np.array(img))
    labels.append(0)

for j in os.listdir(positive_dir):
    img_path = os.path.join(positive_dir, j)
    img = cv2.imread(img_path)
    img = Image.fromarray(img, "RGB")
    img = img.resize((64, 64))
    features.append(np.array(img))
    labels.append(1)

features = np.array(features)
labels = np.array(labels)

plt.figure(figsize=(12, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(features[i+1497])
    if labels[i+1497] == 0:
        plt.title("Negative")
    else:
        plt.title("Positive")
    plt.axis('off')

print("Some sample MRI images Images")
plt.tight_layout()
plt.show()


unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(8, 6))
bars=plt.bar(unique, counts)
plt.title('Frequency of Positive or Negative for Brain Tumor')
x_labels = ['Negative ', 'Positive']
plt.ylabel('Frequency')
plt.xticks(unique, x_labels)
for bar in bars:
    height = bar.get_height()
    plt.annotate('{}'.format(height),
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')
plt.show()




features=np.array(features)
labels=np.array(labels)

X_train, X_test, Y_train, Y_test = train_test_split(features,labels, test_size=0.2, random_state=2)


scaler = StandardScaler()
X_train_std = X_train/255
X_test_std = X_test/255
tf.random.set_seed(3)


  model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(64,64,3)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=50)

loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)
print("accuracy is:",accuracy/100,"%")



# Preprocess the input image
def preprocess_input_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to the same size used during training
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Here I am Provide the path to new MRI image here
# I used the loop as I provided various MRI scans 
# So you can remove loop to give your new MRI image
for i in range(1, 4):
    input_image_path = f"/content/REAL DATA/MRI SCAN_0{str(i)}.jpeg"

    #  Make predictions on the new image
    # Preprocess the input image
    input_image = preprocess_input_image(input_image_path)

    # Add batch dimension to match the model's input shape
    input_image = np.expand_dims(input_image, axis=0)

    # Make prediction
    prediction_probs = model.predict(input_image)[0]

    # Get the predicted class (0 or 1) and probability
    predicted_class = np.argmax(prediction_probs)
    predicted_probability = prediction_probs[predicted_class]

    # Display the image
    img = Image.open(input_image_path)
    plt.imshow(img)
    plt.axis('off')

    # Display prediction results
    if predicted_class == 0:
        print(f' MRI scan is NEGATIVE for tumor with probability: {predicted_probability:.4f}')
    else:
        print(f' MRI scan is POSITIVE for tumor with probability: {predicted_probability:.4f}')

    plt.show()  # Show the image with prediction in each iteration


