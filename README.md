## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Usage

1. Update the `ROOT_PATH` variable in the `main()` function to point to your dataset
2. Run the training script:
```bash
python run_code.py
```

## Training Configuration

- **Image Size**: 256×256 pixels
- **Batch Size**: 8
- **Learning Rate**: 0.001 (Cosine Annealing)
- **Optimizer**: AdamW with weight decay (1e-4)
- **Loss Function**: Binary Cross Entropy
- **Epochs**: 50
- **Gradient Clipping**: Max norm 1.0
