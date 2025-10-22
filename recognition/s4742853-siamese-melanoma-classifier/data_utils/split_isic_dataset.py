import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For a nice progress bar

print("Starting dataset preprocessing...")

# --- 1. Define File Paths ---
# Base directory is the current working directory
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
ISIC_DIR = os.path.join(DATA_DIR, 'isic2020')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

# Input paths
ALL_IMAGES_DIR = os.path.join(ISIC_DIR, 'all_images')
GROUND_TRUTH_CSV = os.path.join(ISIC_DIR, 'all_ground-truth_v2.csv')
DUPLICATES_CSV = os.path.join(ISIC_DIR, 'duplicates.csv')

# Output path
TRAIN_DIR = os.path.join(CLEANED_DIR, 'train')
TEST_DIR = os.path.join(CLEANED_DIR, 'test')
VAL_DIR = os.path.join(CLEANED_DIR, 'validation')

TRAIN_CSV = os.path.join(CLEANED_DIR, 'train.csv')
TEST_CSV = os.path.join(CLEANED_DIR, 'test.csv')
VAL_CSV = os.path.join(CLEANED_DIR, 'validation.csv')

# --- 2. Create Output Directories ---
print(f"Creating directories at {CLEANED_DIR}...")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# --- 3. Load Data and Remove Duplicates ---
print("Loading CSVs...")
try:
    df_truth = pd.read_csv(GROUND_TRUTH_CSV)
    df_dupes = pd.read_csv(DUPLICATES_CSV)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'all_ground-truth_v2.csv' and 'duplicates.csv' are in 'data/isic2020/'.")
    exit()

print(f"Original images in ground truth: {len(df_truth)}")

# Get the set of images to remove (we'll keep image_name_1 and remove image_name_2)
images_to_remove = set(df_dupes['image_name_2'])
print(f"Found {len(images_to_remove)} duplicate images to remove.")

# Filter the main dataframe
df_cleaned = df_truth[~df_truth['image_name'].isin(images_to_remove)].copy()
print(f"Total images after removing duplicates: {len(df_cleaned)}")

# Select only the columns we need
df_final = df_cleaned[['image_name', 'target']].copy()

# --- 4. Perform Stratified Split (70/15/15) ---
print("Performing stratified 70/15/15 split...")

# First split: 70% train, 30% temp (for test/val)
df_train, df_temp = train_test_split(
    df_final,
    test_size=0.30,
    stratify=df_final['target'],
    random_state=42  # for reproducibility
)

# Second split: 50% of temp -> 15% validation, 15% test
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,  # 50% of 30% = 15%
    stratify=df_temp['target'],
    random_state=42  # for reproducibility
)

print("\nSplit complete:")
print(f"  Training set:   {len(df_train)} images")
print(f"  Validation set: {len(df_val)} images")
print(f"  Test set:       {len(df_test)} images")

# --- 5. Save New CSVs ---
print(f"Saving new CSVs to {CLEANED_DIR}...")
df_train.to_csv(TRAIN_CSV, index=False)
df_val.to_csv(VAL_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)
print("CSVs saved.")

# --- 6. Copy Image Files ---
print("Copying image files... This may take a while.")


def copy_files(df, dest_dir_path):
    """Helper function to copy files with a progress bar."""
    copied_count = 0
    not_found_count = 0

    # Use tqdm for a progress bar
    for image_name in tqdm(df['image_name'], desc=f"Copying to {os.path.basename(dest_dir_path)}"):
        image_filename = f"{image_name}.jpg"
        src_path = os.path.join(ALL_IMAGES_DIR, image_filename)
        dest_path = os.path.join(dest_dir_path, image_filename)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            copied_count += 1
        else:
            print(f"\nWarning: Source file not found, skipping: {src_path}")
            not_found_count += 1

    print(f"Finished copying to {os.path.basename(dest_dir_path)}.")
    print(f"  Copied: {copied_count}, Not Found: {not_found_count}")


# Run the copy function for each split
copy_files(df_train, TRAIN_DIR)
copy_files(df_val, VAL_DIR)
copy_files(df_test, TEST_DIR)

print("\n--- All done! ---")
print("Dataset successfully cleaned, split, and copied.")
