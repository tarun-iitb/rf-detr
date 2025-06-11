# Required packages:
# pip install rfdetr lxml

import os
import json
import shutil
import subprocess
from pathlib import Path
from lxml import etree
from tqdm import tqdm

from rfdetr import RFDETRBase


def download_dataset(target_dir: Path):
    """Clone the BCCD dataset from GitHub if it doesn't exist."""
    if target_dir.exists():
        return
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/Shenggan/BCCD_Dataset.git",
        str(target_dir)
    ], check=True)


def parse_voc_xml(xml_path: Path):
    """Parse Pascal VOC annotation and return image info and objects."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.findtext("xmin")))
        ymin = int(float(bndbox.findtext("ymin")))
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))
        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "area": (xmax - xmin) * (ymax - ymin)
        })

    return filename, width, height, objects


def prepare_split(src_dir: Path, dst_dir: Path, ids: list, categories: dict):
    """Prepare one dataset split in COCO format."""
    images = []
    annotations = []
    ann_id = 1
    for idx, img_id in enumerate(tqdm(ids, desc=f"Preparing {dst_dir.name}")):
        xml_path = src_dir / "Annotations" / f"{img_id}.xml"
        img_file, width, height, objs = parse_voc_xml(xml_path)

        src_img = src_dir / "JPEGImages" / img_file
        dst_img = dst_dir / img_file
        shutil.copy(src_img, dst_img)

        images.append({
            "id": idx + 1,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        for obj in objs:
            annotations.append({
                "id": ann_id,
                "image_id": idx + 1,
                "category_id": categories[obj["name"]],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for name, cid in categories.items()
        ]
    }

    with open(dst_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)


def prepare_dataset(src_root: Path, dst_root: Path):
    """Convert VOC dataset to COCO format structure."""
    splits = {
        "train": "train.txt",
        "valid": "val.txt",
        "test": "test.txt"
    }

    categories = {"RBC": 1, "WBC": 2, "Platelets": 3}
    dst_root.mkdir(parents=True, exist_ok=True)

    for split, txt in splits.items():
        split_dir = dst_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(src_root / "ImageSets" / "Main" / txt) as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]
        prepare_split(src_root, split_dir, ids, categories)


if __name__ == "__main__":
    # Paths
    raw_dataset_path = Path("BCCD_Dataset") / "BCCD"
    coco_dataset_path = Path("bccd_coco")
    output_dir = Path("bccd_output")

    # Download and prepare dataset
    download_dataset(Path("BCCD_Dataset"))
    prepare_dataset(raw_dataset_path, coco_dataset_path)

    # Load model and train following the README recommendations
    model = RFDETRBase()
    model.train(
        dataset_dir=str(coco_dataset_path),
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=str(output_dir)
    )

    print(f"Training complete. Weights saved to {output_dir}")
