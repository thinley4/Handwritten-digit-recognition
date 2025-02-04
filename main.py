import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from pymongo import MongoClient
import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Use MongoDB Atlas URL if needed
db = client["digit_recognition_db"]
collection = db["predictions"]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # Add `dim=1` for proper softmax usage

# Load the model
model = CNN()
model.load_state_dict(torch.load("models/handwritten_digit_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Streamlit App
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit, and the model will predict the digit!")

# File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()  # Get the predicted digit

    # Show the prediction
    # st.write(f"Predicted Digit: **{prediction}**")
    st.html(f"<h2 style='text-align: center; font-size: 40px'>Predicted Digit: <b>{prediction}</b></h2>")

    # Save prediction to MongoDB
    prediction_data = {
        "predicted_digit": prediction,
        "timestamp": datetime.datetime.now(),
    }
    # Convert the image to bytes
    image_bytes = uploaded_file.read()

    # Store prediction data and image together
    prediction_data["image"] = image_bytes
    collection.insert_one(prediction_data)
    st.success("Prediction saved to database!")
