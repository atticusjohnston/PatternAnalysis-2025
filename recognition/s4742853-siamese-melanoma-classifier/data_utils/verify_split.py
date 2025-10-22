import pandas as pd
import os
from tqdm import tqdm
from collections import Counter

print("Starting dataset verification...")

# --- 1. Define File Paths ---
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
ISIC_DIR = os.path.join(DATA_DIR, 'isic2020')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

GROUND_TRUTH_CSV = os.path.join(ISIC_DIR, 'all_ground-truth_v2.csv')
DUPLICATES_CSV = os.path.join(ISIC_DIR, 'duplicates.csv')

TRAIN_DIR = os.path.join(CLEANED_DIR, 'train_images')
TEST_DIR = os.path.join(CLEANED_DIR, 'test_images')
VAL_DIR = os.path.join(CLEANED_DIR, 'validation_images')

TRAIN_CSV = os.path.join(CLEANED_DIR, 'train.csv')
TEST_CSV = os.path.join(CLEANED_DIR, 'test.csv')
VAL_CSV = os.path.join(CLEANED_DIR, 'validation.csv')

errors = []
warnings = []


def verify_csv_structure(df, name):
    """Check CSV has required columns and valid data."""
    if 'image_name' not in df.columns or 'target' not in df.columns:
        errors.append(f"[CSV Structure] {name} missing required columns")
        return False

    if df['image_name'].isna().any():
        errors.append(f"[CSV Structure] {name} has null image_name values")

    if df['target'].isna().any():
        errors.append(f"[CSV Structure] {name} has null target values")

    if not df['target'].isin([0, 1]).all():
        errors.append(f"[CSV Structure] {name} has invalid target values (must be 0 or 1)")

    if df['image_name'].duplicated().any():
        dupes = df[df['image_name'].duplicated(keep=False)]['image_name'].tolist()
        errors.append(f"[CSV Structure] {name} has duplicate image_names: {dupes[:5]}")

    return True


def verify_files_match_csv(df, dir_path, partition_name):
    """Verify all CSV entries have corresponding files and vice versa."""
    csv_images = set(df['image_name'])

    if not os.path.exists(dir_path):
        errors.append(f"[Directory] {partition_name} directory does not exist: {dir_path}")
        return

    disk_files = set([f[:-4] for f in os.listdir(dir_path) if f.endswith('.jpg')])

    missing_files = csv_images - disk_files
    extra_files = disk_files - csv_images

    if missing_files:
        errors.append(f"[File Missing] {partition_name}: {len(missing_files)} images in CSV but not on disk")
        if len(missing_files) <= 10:
            errors.append(f"  Missing: {sorted(missing_files)}")

    if extra_files:
        warnings.append(f"[Extra Files] {partition_name}: {len(extra_files)} images on disk but not in CSV")
        if len(extra_files) <= 10:
            warnings.append(f"  Extra: {sorted(extra_files)}")


def verify_labels(df, partition_name, truth_data):
    """Verify labels match ground truth."""
    mismatches = []
    missing = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Verifying {partition_name} labels"):
        image_name = row['image_name']
        split_target = row['target']

        if image_name not in truth_data:
            missing.append(image_name)
        elif truth_data[image_name] != split_target:
            mismatches.append((image_name, split_target, truth_data[image_name]))

    if missing:
        errors.append(f"[Label Missing] {partition_name}: {len(missing)} images not in ground truth")
        if len(missing) <= 10:
            errors.append(f"  Missing: {missing}")

    if mismatches:
        errors.append(f"[Label Mismatch] {partition_name}: {len(mismatches)} incorrect labels")
        for img, split_val, truth_val in mismatches[:10]:
            errors.append(f"  {img}: split={split_val}, truth={truth_val}")


def verify_split_ratios(df_train, df_val, df_test):
    """Check split ratios are approximately correct."""
    total = len(df_train) + len(df_val) + len(df_test)
    train_pct = len(df_train) / total * 100
    val_pct = len(df_val) / total * 100
    test_pct = len(df_test) / total * 100

    print(f"\nSplit ratios: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")

    if not (68 <= train_pct <= 72):
        warnings.append(f"[Split Ratio] Train set is {train_pct:.1f}% (expected ~70%)")
    if not (13 <= val_pct <= 17):
        warnings.append(f"[Split Ratio] Val set is {val_pct:.1f}% (expected ~15%)")
    if not (13 <= test_pct <= 17):
        warnings.append(f"[Split Ratio] Test set is {test_pct:.1f}% (expected ~15%)")


def verify_stratification(df_train, df_val, df_test):
    """Check class distribution is similar across splits."""
    train_dist = Counter(df_train['target'])
    val_dist = Counter(df_val['target'])
    test_dist = Counter(df_test['target'])

    train_ratio = train_dist[1] / len(df_train) * 100
    val_ratio = val_dist[1] / len(df_val) * 100
    test_ratio = test_dist[1] / len(df_test) * 100

    print(f"\nClass 1 ratios: Train={train_ratio:.2f}%, Val={val_ratio:.2f}%, Test={test_ratio:.2f}%")

    max_diff = max(abs(train_ratio - val_ratio), abs(train_ratio - test_ratio), abs(val_ratio - test_ratio))
    if max_diff > 2.0:
        warnings.append(f"[Stratification] Class distribution varies by {max_diff:.2f}% across splits")


def main():
    # Load CSVs
    print("Loading CSVs...")
    try:
        df_truth = pd.read_csv(GROUND_TRUTH_CSV)
        df_dupes = pd.read_csv(DUPLICATES_CSV)
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Verify CSV structure
    print("\n1. Verifying CSV structures...")
    verify_csv_structure(df_train, "train.csv")
    verify_csv_structure(df_val, "validation.csv")
    verify_csv_structure(df_test, "test.csv")

    if errors:
        print("CSV structure errors found, stopping verification.")
        print("\n".join(errors))
        return

    # Verify uniqueness
    print("\n2. Verifying uniqueness (no overlap)...")
    train_images = set(df_train['image_name'])
    val_images = set(df_val['image_name'])
    test_images = set(df_test['image_name'])

    tv_overlap = train_images & val_images
    tt_overlap = train_images & test_images
    vt_overlap = val_images & test_images

    if tv_overlap:
        errors.append(f"[Overlap] {len(tv_overlap)} images in train AND val: {sorted(tv_overlap)[:5]}")
    if tt_overlap:
        errors.append(f"[Overlap] {len(tt_overlap)} images in train AND test: {sorted(tt_overlap)[:5]}")
    if vt_overlap:
        errors.append(f"[Overlap] {len(vt_overlap)} images in val AND test: {sorted(vt_overlap)[:5]}")

    if not (tv_overlap or tt_overlap or vt_overlap):
        print("  ✓ No overlap between splits")

    # Verify duplicate removal
    print("\n3. Verifying duplicate removal...")
    forbidden_dupes = set(df_dupes['image_name_2'])
    all_split_images = train_images | val_images | test_images
    found_forbidden = all_split_images & forbidden_dupes

    if found_forbidden:
        errors.append(
            f"[Duplicates] {len(found_forbidden)} forbidden duplicates in splits: {sorted(found_forbidden)[:10]}")
    else:
        print("  ✓ No forbidden duplicates found")

    # Verify completeness
    print("\n4. Verifying completeness...")
    truth_images = set(df_truth['image_name'])
    allowed_images = truth_images - forbidden_dupes

    missing_from_splits = allowed_images - all_split_images
    extra_in_splits = all_split_images - allowed_images

    if missing_from_splits:
        errors.append(f"[Completeness] {len(missing_from_splits)} images missing from splits")
    if extra_in_splits:
        errors.append(f"[Completeness] {len(extra_in_splits)} unauthorized images in splits")

    if not (missing_from_splits or extra_in_splits):
        print("  ✓ All valid images accounted for")

    # Verify split ratios and stratification
    print("\n5. Verifying split ratios...")
    verify_split_ratios(df_train, df_val, df_test)

    print("\n6. Verifying stratification...")
    verify_stratification(df_train, df_val, df_test)

    # Verify labels
    print("\n7. Verifying labels against ground truth...")
    truth_data = df_truth.set_index('image_name')['target'].to_dict()
    verify_labels(df_train, "train", truth_data)
    verify_labels(df_val, "validation", truth_data)
    verify_labels(df_test, "test", truth_data)

    # Verify files
    print("\n8. Verifying files match CSVs...")
    verify_files_match_csv(df_train, TRAIN_DIR, "train")
    verify_files_match_csv(df_val, VAL_DIR, "validation")
    verify_files_match_csv(df_test, TEST_DIR, "test")

    # Final report
    print("\n" + "=" * 60)
    if not errors and not warnings:
        print("  ✓ VERIFICATION SUCCESSFUL - ALL CHECKS PASSED")
    else:
        if errors:
            print(f"  ✗ VERIFICATION FAILED - {len(errors)} ERROR(S)")
            for err in errors:
                print(f"    {err}")
        if warnings:
            print(f"\n  ⚠ {len(warnings)} WARNING(S)")
            for warn in warnings:
                print(f"    {warn}")
    print("=" * 60)


if __name__ == "__main__":
    main()