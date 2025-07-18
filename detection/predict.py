import os
import torch
from torchvision import transforms
import argparse
from classification.models.deep_hs_module import DeepHsModule
from core.datasets.hyperspectral_dataset import HyperspectralDataset
from types.index import MangoImage

class DeepHSPredictor:
    def __init__(self, model_path, data_path, input_size=(64, 64)):
        self.model = DeepHsModule.load_from_checkpoint(model_path)
        self.model.eval()
        self.data_path = data_path
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

    def load_images(self):
        images = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.bin'):  # Assuming hyperspectral images are in .bin format
                file_path = os.path.join(self.data_path, filename)
                images.append(MangoImage(file_path=file_path))
        return images

    def preprocess_image(self, mango_image):
        image_data = torch.load(mango_image.file_path)  # Load the hyperspectral image data
        image_tensor = self.transform(image_data)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def predict(self, mango_images):
        predictions = []
        with torch.no_grad():
            for mango_image in mango_images:
                image_tensor = self.preprocess_image(mango_image)
                output = self.model(image_tensor)
                predicted_class = output.argmax(dim=1).item()
                predictions.append((mango_image.file_path, predicted_class))
        return predictions

def get_parser():
    parser = argparse.ArgumentParser("DeepHS detector:")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing mango images.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path

    predictor = DeepHSPredictor(model_path, data_path)
    mango_images = predictor.load_images()
    results = predictor.predict(mango_images)

    for file_path, predicted_class in results:
        print(f"File: {file_path}, Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()