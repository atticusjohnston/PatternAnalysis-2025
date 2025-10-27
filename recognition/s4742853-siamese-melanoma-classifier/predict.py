import torch
import argparse
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
from modules import SiameseNetwork, PretrainedSiameseNetwork


def predict(model_path: str, image_path: str, ref_csv: str, ref_img_dir: str, k: int = 10, model_type: str = 'pretrained') -> tuple:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if model_type == 'pretrained':
        network = PretrainedSiameseNetwork(pretrained=False)
    else:
        network = SiameseNetwork()

    network.load_state_dict(torch.load(model_path, map_location=device))
    network.to(device)
    network.eval()

    mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
    std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_img = Image.open(image_path).convert('RGB')
    test_img = transform(test_img).unsqueeze(0).to(device)

    ref_data = pd.read_csv(ref_csv)
    ref_by_class = {}

    for _, row in ref_data.iterrows():
        label = row['target']
        if label not in ref_by_class:
            ref_by_class[label] = []
        ref_by_class[label].append(row['image_name'])

    for label in ref_by_class:
        if len(ref_by_class[label]) > k:
            ref_by_class[label] = ref_by_class[label][:k]

    class_probs = {}

    with torch.no_grad():
        for label, img_names in ref_by_class.items():
            imgs = []
            for img_name in img_names:
                ref_path = os.path.join(ref_img_dir, f"{img_name}.jpg")
                img = Image.open(ref_path).convert('RGB')
                imgs.append(transform(img))

            ref_batch = torch.stack(imgs).to(device)
            test_batch = test_img.repeat(ref_batch.size(0), 1, 1, 1)

            probs = network(test_batch, ref_batch)

            top_k_to_use = min(3, len(probs))
            top_k_probs = probs.topk(k=top_k_to_use).values
            class_probs[label] = top_k_probs.mean().item()

    pred_class = max(class_probs, key=class_probs.get)
    pred_prob = class_probs[pred_class]

    return pred_class, pred_prob, class_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network single image prediction.")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to the saved model state dictionary (.pt file).")
    parser.add_argument('--image-path', type=str, required=True,
                        help="Path to the single image file to classify.")
    parser.add_argument('--ref-csv', type=str, required=True,
                        help="CSV file containing reference image names and labels.")
    parser.add_argument('--ref-img-dir', type=str, required=True,
                        help="Directory containing the reference image files.")
    parser.add_argument('--k', type=int, default=10,
                        help="Number of reference images per class to use for k-NN comparison.")
    parser.add_argument('--model-type', type=str, default='pretrained',
                        choices=['pretrained', 'custom'],
                        help="Specify the model architecture ('pretrained' or 'custom').")

    args = parser.parse_args()

    pred_class, pred_prob, all_probs = predict(
        args.model_path,
        args.image_path,
        args.ref_csv,
        args.ref_img_dir,
        args.k,
        args.model_type
    )

    print("--- Prediction Results ---")
    print(f"Input Image: {args.image_path}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {pred_prob:.4f}")
    print("\nClass Probabilities:")
    for label, prob in sorted(all_probs.items()):
        print(f"  Class {label}: {prob:.4f}")
