# Model Card: Steel-Surface Defect CNN

## Model Details
- **Architecture:** ResNet50 (ImageNet pretrained)
- **Framework:** PyTorch
- **License:** MIT
- **Version:** 1.0
- **Author:** Ramadhan Adam Zome

## Intended Use
**Primary Use:** Automated visual quality control in steel manufacturing plants, particularly in resource constrained African industrial contexts.

**Target Users:** Quality assurance engineers, factory floor supervisors, steel manufacturers.

**Deployment Scenario:** Edge devices (CPU/GPU) for real-time inspection or batch processing of production line images.

## Training Data
- **Dataset:** Severstal Steel Defect Detection (public Kaggle competition)
- **Classes:** 4 defect types (Crack, Scratch, Dent, OK)
- **Images:** 4,000+ high-resolution steel sheet images (256×1600)
- **Class Distribution:** Highly imbalanced (Class 3: 5,043 samples; Class 2: 210 samples)
- **Preprocessing:** RLE mask decoding, resized to 224×224, ImageNet normalization

## Performance Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| OK | 0.37 | 0.19 | 0.25 | 90 |
| Crack | **0.04** | **1.00** | **0.08** | 21 |
| Scratch | 0.98 | 0.11 | 0.20 | 505 |
| Dent | 0.32 | 0.27 | 0.29 | 51 |
| **Macro Avg** | **0.43** | **0.39** | **0.20** | 667 |

- **ROC-AUC Macro:** 0.652
- **Training Epochs:** 20 (early stopping at epoch 16)
- **Best Val Macro-F1:** 0.206

## Limitations & Biases
1. **Class Imbalance:** Model is biased toward majority class (Class 3). Rare defects (Class 2) may be under-detected.
2. **Lighting Sensitivity:** Performance degrades ~6% under extreme glare conditions (ΔF1 = 0.062 &gt; goal of 0.03).
3. **Resolution Dependency:** Trained on high-res images; performance may drop with lower-quality factory cameras.
4. **Single-Label Simplification:** Images with multiple defects are assigned only the first defect class.

## Ethical Considerations
- **Labor Impact:** May reduce need for manual inspectors; recommend human-in-the-loop for final decisions.
- **Fairness:** Dataset may not represent all steel types/finishes used in underrepresented regions.
- **Safety-Over-Reliance:** High automation confidence could reduce expert oversight, risking missed novel defects.
- **Data Privacy:** Ensure factory camera data complies with local privacy regulations.

## Recommendations for Deployment
1. Use **contrast enhancement** preprocessing for low-light factory conditions.
2. Implement **human review queue** for predictions with confidence &lt; 0.7.
3. **Continuous monitoring:** Log false positives/negatives for model retraining.
4. Consider **semi-supervised learning** to augment rare defect classes.
5. **Quantize model** (INT8) for edge deployment without GPUs.

## How to Use
```bash
# Load model
import torch
model = torch.load("outputs/best_steel_resnet50.pt")
model.eval()

# Predict
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([...])
image = transform(Image.open("steel_sheet.jpg")).unsqueeze(0)
prediction = model(image).argmax().item()
```
## Contact
For questions or support, please contact Ramadhan Adam Zome at ramadhanzome4@gmail.com