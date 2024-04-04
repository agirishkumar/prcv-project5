import gradio as gr
import base
import analyze
import greek
import testing
import random
import torch
from torchvision import transforms


# Assuming you have a pre-trained model for classification
# For demonstration, let's just randomly choose a category
categories = ["Cat", "Dog", "Bird"]

def classify_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load your model here
    model = base.Network().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print(image)
    #transformed_image = transform(image).unsqueeze(0)
    transformed_image = testing.preprocess_image(image)
    return testing.classify_digit(model, device, transformed_image)

def translate_to_greek(letter):
    # Translate text to Greek

    return greek.predict_image(letter)

def simulate_training(dataset_size):
    # Simulate some "training" based on dataset size. For demo, we just return a random accuracy.
    accuracy = random.uniform(0.5, 1.0)  # Random accuracy between 50% and 100%
    return f"Simulated training accuracy: {accuracy:.2%} for dataset size: {dataset_size}"

# Define Gradio interfaces
image_classifier = gr.Interface(fn=classify_image, inputs="file", outputs="label", description="Image Classifier")
text_translator = gr.Interface(fn=translate_to_greek, inputs="file", outputs="label", description="English to Greek Translator")
training_simulator = gr.Interface(fn=simulate_training, inputs="number", outputs="text", description="Train Simulator")

# Combine interfaces into a Parallel interface
app = gr.TabbedInterface([image_classifier, text_translator, training_simulator], ["Classification", "Greek", "Train"])

# Launch the app
app.launch()