import pandas as pd
import numpy as np
from pathlib import Path


def create_pairs(csv_path, output_path, seed=42):
    print(csv_path)
    df = pd.read_csv(csv_path)
    np.random.seed(seed)

    class_0 = df[df['target'] == 0]['image_name'].values
    class_1 = df[df['target'] == 1]['image_name'].values

    # Calculate numbers for 50/50 split
    max_same = min(len(class_0) * (len(class_0) - 1) // 2,
                   len(class_1) * (len(class_1) - 1) // 2)
    max_diff = len(class_0) * len(class_1)

    n_same = min(max_same, max_diff)
    n_diff = n_same

    print(f"Generating {n_same} same pairs, {n_diff} different pairs")

    # Same pairs - sample randomly from each class
    same_pairs = []
    for cls, label in [(class_0, 0), (class_1, 1)]:
        pairs_needed = n_same // 2
        for _ in range(pairs_needed):
            idx1, idx2 = np.random.choice(len(cls), size=2, replace=False)
            same_pairs.append([cls[idx1], cls[idx2], label, label, 1])

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
    print(f"Saved {len(same_pairs)} same, {len(diff_pairs)} different")
    print("—")


if __name__ == '__main__':
    base_path = Path('../data/cleaned')

    for split in ['train', 'validation']:
        create_pairs(
            base_path / f'{split}.csv',
            base_path / f'{split}_pairs.csv'
        )

    print("\n" + "=" * 50)
    print("BALANCE CHECK")
    print("=" * 50)

    for split in ['train', 'validation']:
        df = pd.read_csv(f'../data/cleaned/{split}_pairs.csv')
        counts = df['pair_label'].value_counts()
        pct = df['pair_label'].value_counts(normalize=True) * 100
        print(f"\n{split.upper()}:")
        print(f"  0 (different): {counts[0]:6d} ({pct[0]:.1f}%)")
        print(f"  1 (same):      {counts[1]:6d} ({pct[1]:.1f}%)")
        print(f"  Total:         {len(df):6d}")