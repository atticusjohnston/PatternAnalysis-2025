import pandas as pd
import numpy as np
from pathlib import Path


def create_pairs(csv_path, output_path, seed=42):
    print(csv_path)
    df = pd.read_csv(csv_path)
    np.random.seed(seed)

    class_0 = df[df['target'] == 0]['image_name'].values
    class_1 = df[df['target'] == 1]['image_name'].values

    # Calculate numbers of each for 50/50 split (feature weights should be equal)
    # This is maximum possible pairs for each class
    # Uses NC2 formula
    # Need min so can have equal between classes
    n_same = min(len(class_0) * (len(class_0) - 1) // 2,
                 len(class_1) * (len(class_1) - 1) // 2)

    # Capped at n_same as we want it to be equal
    # Max us len_0 * len_1 (combs)
    n_diff = min(len(class_0) * len(class_1), n_same)
    print(f"n_same size: {n_same}")
    print(f"n_diff size: {n_diff}")

    # Same pairs
    same_pairs = []
    for cls, label in [(class_0, 0), (class_1, 1)]:
        # Select random img
        indices = np.random.choice(len(cls), size=len(cls), replace=False)

        # pair consecutive indices (random anyway)
        for i in range(0, len(indices) - 1, 2):
            if len(same_pairs) >= n_same:
                break
            # [image_1, image_2, label_1, label_2, pair_label]
            same_pairs.append([cls[indices[i]], cls[indices[i + 1]], label, label, 1])

    # Different pairs
    diff_pairs = []
    indices_0 = np.random.choice(len(class_0), size=n_diff, replace=True)
    indices_1 = np.random.choice(len(class_1), size=n_diff, replace=True)
    for i0, i1 in zip(indices_0, indices_1):
        diff_pairs.append([class_0[i0], class_1[i1], 0, 1, 0])

    pairs = same_pairs + diff_pairs
    np.random.shuffle(pairs)

    df_pairs = pd.DataFrame(pairs, columns=['image_1', 'image_2', 'image_1_label', 'image_2_label', 'pair_label'])
    df_pairs.to_csv(output_path, index=False)
    print("—")


if __name__ == '__main__':
    base_path = Path('../data/cleaned')

    for split in ['train', 'test', 'validation']:
        create_pairs(
            base_path / f'{split}.csv',
            base_path / f'{split}_pairs.csv'
        )