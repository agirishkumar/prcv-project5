



def main():
    # ... Your existing code ...

    # Load model
    model = Network().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))

    # Process and classify each image
    for i in range(10):
        image_path = f'{i}.jpg'  # Assuming images are named '0.jpg', '1.jpg', etc.
        image_tensor = preprocess_image(image_path)
        prediction = classify_digit(model, device, image_tensor)
        print(f'Handwritten digit: {i}, Predicted digit: {prediction}')

if __name__ == "__main__":
    main()
