# Authors: Girish Kumar Adari, Alexander Seljuk
# Extension: Digit recognition for live video feed

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from base import Network  

# Load your trained model
model = Network()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define preprocess function for frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = transform(gray_frame).unsqueeze(0)  

    # Prediction
    with torch.no_grad():
        output = model(processed_frame)
        _, pred = torch.max(output.data, 1)

    # Display prediction
    cv2.putText(frame, 'Predicted Digit: ' + str(pred.item()), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
