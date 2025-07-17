# filepath: /mango-quality-classification/mango-quality-classification/src/predict.py
import os
import torch
from torchvision import transforms
from classification.models.deep_hs_module import DeepHsModule
from core.datasets.hyperspectral_dataset import HyperspectralDataset
from types.index import MangoImage

class MangoQualityPredictor:
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
            if filename.endswith('.npy'):  # Assuming hyperspectral images are in .npy format
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

def main():
    model_path = 'path/to/your/best_model.ckpt'  # Update with the actual model path
    data_path = 'path/to/your/mango/images'  # Update with the actual data path

    predictor = MangoQualityPredictor(model_path, data_path)
    mango_images = predictor.load_images()
    results = predictor.predict(mango_images)

    for file_path, predicted_class in results:
        print(f"File: {file_path}, Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()