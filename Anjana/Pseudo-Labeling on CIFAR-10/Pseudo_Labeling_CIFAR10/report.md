# Accuracy Improvement Report

## Setup
- Dataset: CIFAR-10
- Model: ResNet-18
- Initial labeled data: 10%

## Results (Typical)
| Model | Training Data | Test Accuracy |
|------|---------------|---------------|
| Initial Model | 10% labeled | ~55–60% |
| Pseudo-Labeled Model | 100% (pseudo) | ~65–70% |

## Conclusion
Pseudo-labeling significantly improves performance by leveraging unlabeled data,
demonstrating the effectiveness of semi-supervised learning.
