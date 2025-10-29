import pandas as pd
import numpy as np
from pathlib import Path


def create_pairs(csv_path, output_path, seed=42):
    print(csv_path)
    df = pd.read_csv(csv_path)
    np.random.seed(seed)

    class_0 = df[df['target'] == 0]['image_name'].values
    class_1 = df[df['target'] == 1]['image_name'].values

    target_appearances = 15
    n_same_per_class = len(class_0) * target_appearances // 2  # class 0 pairs
    n_same_class_1 = len(class_1) * target_appearances // 2    # class 1 pairs
    n_diff = n_same_per_class + n_same_class_1  # balance with different pairs

    print(f"Class 0 same pairs: {n_same_per_class}")
    print(f"Class 1 same pairs: {n_same_class_1}")
    print(f"Different pairs: {n_diff}")

    # Same pairs - class 1 (with replacement)
    same_pairs = []
    for _ in range(n_same_per_class):
        idx1, idx2 = np.random.choice(len(class_0), size=2, replace=True)
        while idx1 == idx2:  # avoid self-pairs
            idx2 = np.random.choice(len(class_0))
        same_pairs.append([class_0[idx1], class_0[idx2], 0, 0, 1])

    # Same pairs - class 0 (with replacement)
    for _ in range(n_same_class_1):
        idx1, idx2 = np.random.choice(len(class_1), size=2, replace=True)
        while idx1 == idx2:
            idx2 = np.random.choice(len(class_1))
        same_pairs.append([class_1[idx1], class_1[idx2], 1, 1, 1])

    # Different pairs - balanced cross-class
    diff_pairs = []
    indices_0 = np.random.choice(len(class_0), size=n_diff, replace=True)
    indices_1 = np.random.choice(len(class_1), size=n_diff, replace=True)
    for i0, i1 in zip(indices_0, indices_1):
        diff_pairs.append([class_0[i0], class_1[i1], 0, 1, 0])

    pairs = same_pairs + diff_pairs
    np.random.shuffle(pairs)

    df_pairs = pd.DataFrame(pairs, columns=['image_1', 'image_2', 'image_1_label', 'image_2_label', 'pair_label'])
    df_pairs.to_csv(output_path, index=False)
    print(f"Total pairs: {len(pairs)}")
    print(f"Pair balance: {len(same_pairs)} same, {len(diff_pairs)} different")
    print("—")


if __name__ == '__main__':
    base_path = Path('../data/cleaned')

    for split in ['train']:
        create_pairs(
            base_path / f'{split}.csv',
            base_path / f'{split}_pairs.csv'
        )

    print("\n" + "=" * 50)
    print("BALANCE CHECK")
    print("=" * 50)

    for split in ['train']:
        df = pd.read_csv(f'../data/cleaned/{split}_pairs.csv')
        counts = df['pair_label'].value_counts()
        pct = df['pair_label'].value_counts(normalize=True) * 100
        print(f"\n{split.upper()}:")
        print(f"  0 (different): {counts[0]:6d} ({pct[0]:.1f}%)")
        print(f"  1 (same):      {counts[1]:6d} ({pct[1]:.1f}%)")
        print(f"  Total:         {len(df):6d}")