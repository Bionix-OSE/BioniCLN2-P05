import os
import torch
from torchvision import transforms
import argparse
from classification.model_factory import get_model
from core.datasets.hyperspectral_dataset import HyperspectralDataset
from classification.transformers.normalize import Normalize
from core.name_convention import CameraType
import core.util as util

class DeepHSPredictor:
    def __init__(self, model_path, hparams, data_path, input_size=(64, 64), camera_type=CameraType.VIS):
        self.data_path = data_path
        self.input_size = input_size

        # Instantiate model using the factory
        self.model = get_model(hparams)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Preprocessing as in trainer
        common_preprocessing = [Normalize(camera_type)]
        transform = transforms.Compose(common_preprocessing)

        self.dataset = HyperspectralDataset(
            classification_type=hparams['classification_type'],
            records=[{"bin_path": os.path.join(data_path, f), "hdr_path": os.path.join(data_path, f.replace('.bin', '.hdr'))}
                     for f in os.listdir(data_path) if f.endswith('.bin')],
            data_path=data_path,
            transform=transform,
            input_size=input_size
        )

    def predict(self):
        predictions = []
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                x, y, channel_wavelengths = self.dataset[idx]
                x = x.unsqueeze(0)
                output = self.model(x, channel_wavelengths=channel_wavelengths)
                predicted_class = output.argmax(dim=1).item()
                predictions.append(predicted_class)
        return predictions

def get_parser():
    parser = argparse.ArgumentParser("DeepHS detector:")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing mango images.')
    parser.add_argument('--model', type=str, required=True, choices=['deephs_net', 'hyve', 'resnet', 'alexnet', 'spectralnet', 'se_resnet', 'deephs_net_se'])
    parser.add_argument('--classification_type', type=str, default='ripeness')
    parser.add_argument('--camera_type', type=str, default='VIS')
    parser.add_argument("--camera_agnostic_num_gauss", default=5, type=int)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    camera_type_enum = getattr(CameraType, args.camera_type)
    hparams = {
        'model': args.model,
        'bands': len(util.get_wavelengths_for(camera_type_enum)),
        'wavelengths': util.get_wavelengths_for(camera_type_enum),
        'classification_type': args.classification_type,
        'camera_type': camera_type_enum,
        'camera_agnostic_num_gauss': args.camera_agnostic_num_gauss,
        'num_classes': 3
    }
    model_path = args.model_path
    data_path = args.data_path

    predictor = DeepHSPredictor(model_path, hparams, data_path, camera_type=hparams['camera_type'])
    results = predictor.predict()

    for predicted_class in results:
        print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()