# Siamese Network Melanoma Classifier

Binary melanoma classification using a Siamese network on ISIC 2020 data. Achieves 71.32% test accuracy with soft top-k voting.

## Overview

This project addresses binary classification of skin lesion images (benign vs melanoma) on the highly imbalanced ISIC 2020 Kaggle Challenge dataset, where melanoma represents only 1.8% of cases. Rather than training a standard classifier, we train a Siamese neural network to discriminate between same-class and different-class image pairs. This approach learns a robust similarity metric, enabling the network to determine whether two images belong to the same diagnostic class. At test time, we use soft top-k voting where each test image is compared against k=10 reference images per class, taking the top-3 highest similarity scores for each class and averaging them. The class with the higher mean similarity is predicted. The twin network architecture uses shared weights across both branches and is joined by a learned weighted L1 distance metric that maps to a probability via sigmoid activation.

## How It Works

Siamese networks learn to map similar images close together and dissimilar images far apart in a learned feature space. The architecture consists of twin networks with shared parameters that process two input images simultaneously through identical convolutional and fully-connected layers. Each branch produces a feature embedding vector, and these embeddings are compared using a weighted L1 distance metric where the weights are learned during training. The distance is passed through a sigmoid activation to produce a probability that the two images belong to the same class. During training, the network sees balanced pairs of same-class images (labeled 1) and different-class images (labeled 0), optimising a binary cross-entropy loss. At test time, rather than making a direct classification, we use soft top-k voting where each test image is compared against k reference images from each class in the validation set. The network produces similarity scores for all comparisons, and we take the mean of the top-5 highest scores for each class. The class with the higher mean similarity is predicted as the final classification.

```
        ┌─────────────┐
        │   Image 1   │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Conv. Net  │
        │  (shared)   │
        └──────┬──────┘
               │
        ┌──────▼──────┐     ┌─────────────┐
        │ Embedding   ├────►│   Weighted  │
        │     h₁      │     │  L1 Distance│──► Sigmoid ──► Probability
        └─────────────┘     │             │
                            └──────▲──────┘
        ┌─────────────┐            │
        │ Embedding   ├────────────┘
        │     h₂      │
        └──────▲──────┘
               │
        ┌──────┴──────┐
        │  Conv. Net  │
        │  (shared)   │
        └──────▲──────┘
               │
        ┌──────┴──────┐
        │   Image 2   │
        └─────────────┘
```

## Model Architecture

### Pretrained Siamese Network
The implemented model uses a ResNet18 feature extractor pretrained on ImageNet as the shared convolutional network, followed by a custom fully-connected head that produces 128-dimensional embeddings. The architecture consists of the ResNet18 backbone (all layers except the final classification layer), followed by a linear layer (512→256), LeakyReLU activation, another linear layer (256→128), and dropout (p=0.5) for regularisation. The network uses learnable alpha weights (128-dimensional vector) to compute a weighted L1 distance between the twin embeddings. The final prediction is computed as `p = sigmoid(sum(alpha_i * |h1^(i) - h2^(i)|))`, where h1 and h2 are the 128-dimensional embeddings from each twin branch.

**Note:** A custom Siamese network architecture (without pretrained weights) was designed but not fully implemented for this project. The pretrained model provides superior performance through transfer learning from ImageNet.

## Dataset & Preprocessing

**ISIC 2020 Kaggle Challenge** - Dermoscopic skin lesion images
- Total: 33,126 images (binary labels)
- Class distribution: 584 melanoma (1.8%), 32,542 benign (98.2%)
- Image size: arbitrarily sized RGB

**Preprocessing Pipeline:**
1. Remove duplicates (provided `duplicates.csv`)
2. 70/15/15 stratified split on `target` label
3. Generate balanced pairs for training only (50/50 same/different), with equal representations of each image throughout all pairs
4. Resize to 224×224 with LANCZOS resampling
5. Normalize: mean=[0.806, 0.620, 0.590], std=[0.085, 0.098, 0.110]
   - Mean and standard deviation calculated from entire training set (22,890 images)

**Data Augmentation (training only):**
- Random horizontal flip
- Random vertical flip
- Random rotation (±20°)
- Random affine (translate=0.1, scale=0.8-1.2, shear=10)
- Random perspective (distortion=0.2, p=0.3)
- Color jitter: brightness=0.2, contrast=0.2, saturation=0.2
- Gaussian blur (kernel=3, sigma=0.1-2.0)
- Random erasing (p=0.3, scale=0.02-0.15)

**Training Pair Generation:**
- 343,348 pairs (171,674 different, 171,674 same)
- Each image aims to appear 15 times on average across all pairs

**Data Utility Scripts:**
- `split_isic_dataset.py` - Split dataset into train, val, test; remove duplicates
- `create_image_pairs.py` - Generate balanced pairs (train only)
- `resize_images.py` - Resize all images to 224×224
- `verify_split.py` - Validate splits (optional)

## Installation

**Requirements:** Python 3.8+, CUDA (optional)

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- torch==2.9.0
- torchvision==0.24.0
- pandas==2.3.3
- pillow==12.0.0
- scikit-learn==1.7.2
- matplotlib==3.10.7
- tqdm==4.67.1

## Usage

### Data Preparation

```bash
# 1. Split dataset and remove duplicates
python split_isic_dataset.py

# 2. Create balanced pairs (train only)
python create_image_pairs.py

# 3. Resize images to 224×224
python resize_images.py

# 4. Verify splits (optional)
python verify_split.py
```

Expected directory structure after preprocessing:
```
data/cleaned/
├── train.csv & train_pairs.csv
├── validation.csv
├── test.csv
├── train_images_224/
├── validation_images_224/
└── test_images_224/
```

### Training

**General/Local:**
```bash
python train.py \
    --mode train \
    --model pretrained \
    --batch-size 512 \
    --epochs 10 \
    --lr 1e-3 \
    --k 10 \
    --save-dir models \
    --results-dir results
```

**All Arguments:**
- `--mode`: train | test | both
- `--model`: pretrained | custom
- `--batch-size`: Training batch size (default: 512)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--k`: Reference images per class for testing (default: 10)
- `--train-csv`: Path to training pairs CSV
- `--train-img-dir`: Training images directory
- `--val-csv`: Validation images CSV
- `--val-img-dir`: Validation images directory
- `--test-csv`: Test images CSV
- `--test-img-dir`: Test images directory
- `--save-dir`: Model save directory
- `--results-dir`: Results save directory
- `--model-timestamp`: Timestamp of saved model (for testing)
- `--log-file`: Log file path

#### Rangpur/SLURM Users

Submit batch job:
```bash
# Training
sbatch submit_job.sh train 512 10 1e-3 pretrained "" 10

# Testing (after training)
sbatch submit_job.sh test 512 10 1e-3 pretrained <TIMESTAMP> 10

# Both (train then test)
sbatch submit_job.sh both 512 10 1e-3 pretrained "" 10
```

Arguments: MODE BATCH_SIZE EPOCHS LR MODEL MODEL_TIMESTAMP K

### Testing

```bash
python train.py \
    --mode test \
    --model pretrained \
    --model-timestamp 1761465943 \
    --k 10
```

Generates `results/predictions_<TIMESTAMP>.csv` with predictions.

### Single Image Prediction

```bash
python predict.py \
    --model-path models/siamese_melanoma_classifier_1761465943.pt \
    --image-path data/cleaned/test_images/ISIC_0015719.jpg \
    --ref-csv data/cleaned/validation.csv \
    --ref-img-dir data/cleaned/validation_images_224 \
    --k 10 \
    --model-type pretrained
```

**Output:**
```
--- Prediction Results ---
Input Image: path/to/lesion.jpg
Predicted Class: 0
Confidence: 0.8234

Class Probabilities:
  Class 0 (benign): 0.8234
  Class 1 (melanoma): 0.1766
```

## Training Details

**Optimization:**
- Loss: Binary cross-entropy (BCELoss) on training pairs
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ExponentialLR (gamma=0.99, per-epoch decay)
- Gradient clipping: max_norm=1.0
- Early stopping: patience=5 epochs based on validation balanced accuracy

**Regularization:**
- L2 weight decay: 1e-5
- Dropout: 0.5 (custom FC layers)
- Data augmentation (training only)

**Training Loop:**
- Train on balanced pairs with BCE loss
- Validate using soft top-5 voting (same as testing)
- Monitor validation balanced accuracy each epoch
- Save best model by validation balanced accuracy
- Stop if no improvement for 5 epochs

**Device Detection:** Automatic (CUDA, MPS, CPU)

## Testing Protocol

**Soft Top-k Voting Inference:**
The testing procedure uses a soft top-k voting scheme rather than direct classification. For each test image, we compare it against k=10 reference images per class selected from the validation set. The trained Siamese network produces similarity scores for all pairs. For each class, we identify the top-5 highest similarity scores and compute their mean. The final prediction is the class with the higher average similarity. This soft voting approach provides more robust predictions than hard nearest-neighbor classification. The similarity scores naturally handle imbalanced distributions because the decision is based on learned metric distances rather than frequency-based priors.

**Inference Steps:**
1. Select k=10 reference images per class from validation set
2. For each test image, compute similarity scores against all 20 references
3. Identify top-5 highest scores for class 0 (benign)
4. Identify top-5 highest scores for class 1 (melanoma)
5. Compute mean of top-5 scores for each class
6. Predict class with higher mean similarity

## Results

**Test Set Performance (4,906 images):**
- **Accuracy: 71.32%** (3,499/4,906 correct)
- **Balanced Accuracy: 60.01%**
- Model: PretrainedSiameseNetwork (ResNet18)
- Training: 8 epochs (early stopped), batch_size=512, lr=1e-4, k=10

**Confusion Matrix:**

|               | Predicted Benign | Predicted Melanoma |
|---------------|------------------|--------------------|
| **Actual Benign**    | 3,457 (TN)       | 1,362 (FP)           |
| **Actual Melanoma**  | 45 (FN)          | 42 (TP)            |

**Per-class Recall:**
- Class 0 (benign): 71.74%
- Class 1 (melanoma): 48.28%

**Analysis:**
The model demonstrates moderate performance on the highly imbalanced ISIC 2020 dataset. While achieving 71.32% raw accuracy, the balanced accuracy of 60.01% reveals the model's challenge in handling class imbalance. The melanoma recall of 48.28% indicates the model correctly identifies approximately half of melanoma cases.

**Sensitivity to Hyperparameters:**

Training is highly sensitive to both learning rate and the K value used in soft top-k voting:

- **Learning Rate Sensitivity**: The choice of learning rate critically impacts convergence and final performance. Early experiments with lr=1e-3 showed unstable training and overfitting, with the model oscillating between predicting predominantly benign (epoch 2: 97.45% accuracy, 50.17% balanced accuracy) or predominantly melanoma (epoch 1: 5.55% accuracy, 51.36% balanced accuracy). Reducing to lr=1e-4 provided more stable convergence, though the model still exhibited significant epoch-to-epoch variation in class predictions. This sensitivity stems from the Siamese architecture's reliance on learning subtle distance metrics - too high a learning rate causes the embedding space to collapse or diverge, while too low prevents adequate separation between classes.

- **K Value Sensitivity**: The number of reference images per class (K=10) and the soft top-k voting mechanism (using top-5 scores) directly affect prediction robustness. With only 87 melanoma cases in the validation set, K=10 means predictions rely on just 11.5% of available melanoma references. Smaller K values risk overfitting to non-representative exemplars, while larger K values may dilute discriminative signals. The top-5 averaging within K=10 provides some robustness against outliers, but preliminary testing showed substantial variance in melanoma recall (±15-20%) when K varied between 5 and 20.

**Limitations:**

The model falls short of the target 80% accuracy specified for this task, indicating room for improvement through hyperparameter tuning, architecture modifications, or alternative training strategies such as focal loss to better handle class imbalance.

## Project Structure

```
s4742853-siamese-melanoma-classifier/
├── README.md                         
├── modules.py                        # Network architectures
├── dataset.py                        # Data loaders & preprocessing
├── train.py                          # Training/testing pipeline
├── predict.py                        # Single image inference
├── requirements.txt                  # Python dependencies
├── scratch.ipynb                     # Development notebook
├── useful_commands.txt               # Common commands reference
├── data/
│   ├── isic2020/                     # Raw ISIC 2020 dataset
│   └── cleaned/                      # Preprocessed splits
│       ├── train.csv & train_pairs.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── train_images_224/
│       ├── validation_images_224/
│       └── test_images_224/
├── data_utils/                       # Preprocessing scripts
│   ├── split_isic_dataset.py         # Dataset splitting
│   ├── create_image_pairs.py         # Pair generation
│   ├── resize_images.py              # Image resizing
│   └── verify_split.py               # Split validation
├── scripts/                          # RANGPUR SLURM job scripts
│   ├── submit_job.sh                 # Training/testing job
│   ├── submit_job_test.sh            # Testing-only job
│   └── submit_predict.sh             # Single prediction job
├── models/                           # Saved checkpoints
│   └── siamese_melanoma_classifier_<TIMESTAMP>.pt
├── results/                          # Test predictions
│   └── predictions_<TIMESTAMP>.csv
└── logs/                             # Training/testing logs
    ├── <JOBID>.out
    ├── <JOBID>.err
    └── <TIMESTAMP>.log
```

## Implementation Notes

**Pairs**
Training on balanced pairs of same-class and different-class images naturally handles severe class imbalance. Rather than learning frequency-based priors that would bias toward the majority class, the network learns a similarity metric that generalizes across distributions. This pair-based approach also enables the network to leverage the full dataset more effectively since we can generate many training pairs from a small number of images. The procedure of setting a target number of times an image appears is used to expose the model to images from the minority class, aiming to aid the model in identifying similar characteristics between the undersampled class at test time.

**Soft Top-k Voting**
Using soft top-k voting at test time provides more robust predictions than direct classification or hard nearest-neighbor voting. By averaging the top-5 similarity scores per class rather than taking a single maximum, the method reduces sensitivity to outliers and the sampled reference images, and produces more stable decisions.

**Pre-trained ResNet18**
ResNet18 provides a strong pretrained feature extractor from ImageNet. With 11M parameters, it offers a good balance between model capacity and generalisation while remaining computationally efficient for training and inference. The pretrained weights give the network a head start on learning discriminative visual features, which is critical when working with limited medical imaging data.

**Balanced Accuracy**
Balanced accuracy is the mean of recall across both classes: (recall_benign + recall_melanoma) / 2. Unlike standard accuracy, which can be misleadingly high on imbalanced datasets by simply predicting the majority class, balanced accuracy treats both classes equally regardless of their prevalence. With melanoma representing only 1.8% of cases, a naive classifier predicting all images as benign would achieve 98.2% accuracy but 50% balanced accuracy (100% recall on benign, 0% on melanoma). We use balanced accuracy for validation-based early stopping and final evaluation because it ensures the model learns to identify both classes effectively rather than exploiting class imbalance.

**Addressing Class Imbalances and Overfitting**
The extreme class imbalance is intrinsically addressed in the Siamese architecture, which learns similarity rather than classification. Balanced training pairs, with equal expected instances of each image in these pairs, ensure the network never sees the imbalance during training, while soft top-k voting at inference uses learned metric distances instead of probabilities, avoiding majority-class bias.
Overfitting is mitigated through aggressive regularisation: dropout (0.5), L2 weight decay (1e-5), extensive data augmentation, early stopping on validation balanced accuracy, and gradient clipping.

**Weighted L1 Distance:**
The learnable alpha vector weights each dimension of the embedding space, allowing the network to emphasize the most discriminative features for melanoma classification. This weighted metric is more flexible than fixed distance functions like Euclidean or cosine similarity, and the weights are learned end-to-end through backpropagation during training.

## References

1. **Koch et al. (2015)** - Siamese Neural Networks for One-shot Image Recognition. [Paper](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf)
2. **ISIC 2020 Challenge Dataset** - Skin Lesion Analysis Towards Melanoma Detection. [Dataset](https://challenge2020.isic-archive.com)

## Acknowledgements

**AI Assistance:** Claude Sonnet 4.5 (Anthropic) was used to generate `data_utils/split_isic_dataset.py` and `data_utils/verify_split.py`. Additional AI assistance was used for documentation and code comments in Python files.