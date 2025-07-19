import os
import torch
from torchvision import transforms
import argparse
from classification.models.deephs_net import DeepHsModule
from core.datasets.hyperspectral_dataset import HyperspectralDataset
from classification.transformers.normalize import Normalize
from core.name_convention import CameraType

class DeepHSPredictor:
    def __init__(self, model_path, data_path, input_size=(64, 64), camera_type=CameraType.VIS):
        self.model = DeepHsModule.load_from_checkpoint(model_path)
        self.model.eval()
        self.data_path = data_path
        self.input_size = input_size

        preprocessed = [Normalize(camera_type)]
        transform = transforms.Compose(preprocessed)

        self.dataset = HyperspectralDataset(
            classification_type="ripeness",
            records=[{"bin_path": f, "hdr_path": f.replace('.bin', '.hdr')} for f in os.listdir(data_path) if f.endswith('.bin')],
            data_path=data_path,
            transform=transform,
            input_size=input_size
        )

    def predict(self):
        predictions = []
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                x, y, channel_wavelengths = self.dataset[idx]
                x = x.unsqueeze(0)  # Add batch dimension
                output = self.model(x, channel_wavelengths=[channel_wavelengths])
                predicted_class = output.argmax(dim=1).item()
                predictions.append(predicted_class)
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
    results = predictor.predict()

    for predicted_class in results:
        print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()