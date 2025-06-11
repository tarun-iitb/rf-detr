"""
Train RF-DETR on the BCCD dataset.

Instructions:
1. Install dependencies:
   pip install rfdetr pylabel supervision

This script downloads the BCCD dataset (Pascal VOC format),
converts it to COCO format as expected by RF-DETR,
trains the model and saves the resulting weights.
"""

import os
import subprocess
from pathlib import Path

# clone BCCD dataset if not already present
if not Path('BCCD_Dataset').exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Shenggan/BCCD_Dataset.git'], check=True)

# paths
VOC_ROOT = Path('BCCD_Dataset/BCCD')
IMG_DIR = VOC_ROOT / 'JPEGImages'
ANN_DIR = VOC_ROOT / 'Annotations'
SPLIT_DIR = VOC_ROOT / 'ImageSets/Main'

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# --- Convert VOC to COCO ----------------------------------------------------
from pylabel import importer
from pylabel.dataset import Dataset

print('Importing VOC annotations...')
# images are referenced relative to annotations
dataset = importer.ImportVOC(str(ANN_DIR), path_to_images='../JPEGImages/')

# Read split files
splits = {
    'train': [line.strip() for line in open(SPLIT_DIR / 'train.txt')],
    'val':   [line.strip() for line in open(SPLIT_DIR / 'val.txt')],
    'test':  [line.strip() for line in open(SPLIT_DIR / 'test.txt')]
}

# assign split to each row in dataframe
from pathlib import Path as _P

def _assign_split(filename: str) -> str:
    stem = _P(filename).stem
    for split, names in splits.items():
        if stem in names:
            return split
    return 'train'

dataset.df['split'] = dataset.df['img_filename'].apply(_assign_split)

# export each split to COCO and copy images
for split, names in splits.items():
    split_dir = DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    df_split = dataset.df[dataset.df['split'] == split].reset_index(drop=True)
    subset = Dataset(df_split)
    subset.export.ExportToCoco(output_path=split_dir / '_annotations.coco.json',
                               cat_id_index=0)

    for name in names:
        src = IMG_DIR / f'{name}.jpg'
        dst = split_dir / f'{name}.jpg'
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

print('Dataset prepared in COCO format.')

# --- Training ---------------------------------------------------------------
from rfdetr import RFDETRBase

# classes in BCCD dataset
CLASS_NAMES = ['RBC', 'WBC', 'Platelets']

# instantiate model with the number of classes
model = RFDETRBase(num_classes=len(CLASS_NAMES))

# recommended training parameters from the README
model.train(
    dataset_dir=str(DATA_DIR),
    epochs=10,              # adjust as needed
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='output/bccd',
    class_names=CLASS_NAMES,
)

# best weights will be saved inside output/bccd
BEST_CHECKPOINT = Path('output/bccd/checkpoint_best_total.pth')
MODEL_DIR = Path('trained_model')
MODEL_DIR.mkdir(exist_ok=True)
if BEST_CHECKPOINT.exists():
    (MODEL_DIR / 'bccd_best.pth').write_bytes(BEST_CHECKPOINT.read_bytes())
    print(f'Saved trained model to {MODEL_DIR / "bccd_best.pth"}')
else:
    print('Training finished but checkpoint not found.')
