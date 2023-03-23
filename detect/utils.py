import random
import os
from types import FunctionType
import cv2
import pandas as pd
import yaml

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from logzero import logger

from pathlib import Path
import shutil
from collections import defaultdict
from tqdm import tqdm
import xml.etree.ElementTree as ET


def balanced_split(imgs_dir, ann_dir, train_ratio):
    obj_imgs = defaultdict(list)
    for img_path in tqdm(imgs_dir.iterdir()):
        ann_file = ann_dir / img_path.name.replace(img_path.suffix, '.xml')
        if not ann_file.exists():
            continue

        for obj in get_annotations(ann_file).keys():
            obj_imgs[obj].append(img_path.name.replace('.jpg', ''))

    sorted_objs = sorted(obj_imgs.keys(), key=lambda obj: len(obj_imgs[obj]))

    train_imgs = set()
    val_imgs = set()
    used_imgs = set()
    for obj in tqdm(sorted_objs):
        imgs = set(obj_imgs[obj])
        left_imgs = imgs.difference(used_imgs)

        left_imgs = list(left_imgs)
        n_imgs = len(imgs)
        obj_train_imgs = left_imgs[:int(train_ratio * n_imgs)]
        obj_val_imgs = left_imgs[int(train_ratio * n_imgs):] + list(used_imgs)

        train_imgs.update(obj_train_imgs)
        val_imgs.update(obj_val_imgs)
        used_imgs.update(val_imgs)
    return list(train_imgs), list(val_imgs)


@dataclass(frozen=True)
class BBox(object):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float = field(
        default=1.0, compare=False, hash=False, repr=False
    )


    def union(self, other):
        return BBox(
            min(self.xmin, other.xmin),
            min(self.ymin, other.ymin),
            max(self.xmax, other.xmax),
            max(self.ymax, other.ymax)
        )

    def inside(self, x, y):
        return (x >= self.xmin and x <= self.xmax) and (y >= self.ymin and y <= self.ymax)

    def contains(self, other, overlap_ratio=0.9):
        if not self.intersected(other):
            return False
        intersect = self.intersect(other)
        if intersect.area / other.area > overlap_ratio:
            return True

    def intersected(self, other):
        return all((
            self.xmin < other.xmax,
            other.xmin < self.xmax,
            self.ymin < other.ymax,
            other.ymin < self.ymax
        ))

    def intersect(self, other):
        assert self.intersected(other)
        return BBox(
            max(self.xmin, other.xmin),
            max(self.ymin, other.ymin),
            min(self.xmax, other.xmax),
            min(self.ymax, other.ymax)
        )

    @property
    def area(self):
        return self.height * self.width

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin


def get_annotations(xml_path: Path, cat_mapper: Dict[str, str] = None) -> Dict[str, List[BBox]]:
    annotations = defaultdict(list)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            rectangle = []
            for v in ['xmin', 'ymin', 'xmax', 'ymax']:
                rectangle.append(int(bbox.find(v).text))
            if cat_mapper is not None:
                name = cat_mapper.get(name, name)
            annotations[name].append(BBox(*rectangle))
    except Exception as e:
        print('{} cannot be parsed'.format(xml_path))
        print(e)
    return annotations


def get_summary(xml_dir: Path, split_file: Path = None):
    imgs_name = None
    if split_file is not None:
        with open(split_file, 'r') as f:
            imgs_name = list(line.strip() for line in f.readlines())

    if isinstance(xml_dir, str):
        xml_dir = Path(xml_dir)
    df = []
    for xml_path in tqdm(xml_dir.iterdir()):
        if imgs_name is not None and xml_path.stem not in imgs_name:
            continue
        anns = get_annotations(xml_path)
        for obj, bboxes in anns.items():
            for bbox in bboxes:
                df.append([xml_path.name, obj, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
    df = pd.DataFrame(df, columns=['path', 'category', 'xmin', 'ymin', 'xmax', 'ymax'])
    return df


def read_classes(yolo_data_path):
    with open(yolo_data_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data['names']


def read_yolo_labels(yolo_label_path, yolo_data_path, cat_mapper: Dict[str, str] = None, conf_threshold=0.5):
    classes = read_classes(yolo_data_path)
    id2class = dict(zip(range(len(classes)), classes))
    label_dir = Path(yolo_label_path)
    labels = {}

    logger.info('read yolo results')
    for label_path in tqdm(label_dir.glob('*.txt')):
        annos = defaultdict(list)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                items = [float(x) for x in line.split()]
                class_id, x, y, w, h = items[:5]
                conf = 1.0
                if len(items) == 6:
                    conf = items[5]
                if conf < conf_threshold:
                    continue
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
                class_name = id2class[int(class_id)]
                if cat_mapper is not None:
                    class_name = cat_mapper.get(class_name, class_name)
                annos[class_name].append(BBox(x_min, y_min, x_max, y_max, conf))
        labels[label_path.stem] = annos
    return labels



def read_voc_labels(gt_label_path):
    label_dir = Path(gt_label_path)
    labels = {}
    logger.info('read ground-truth')
    for label_path in tqdm(label_dir.glob('*.xml')):
        annos = get_annotations(label_path)
        labels[label_path.stem] = annos
    return labels


def draw_bboxes(img, annos: List[BBox]):
    for bbox in annos:
        cv2.rectangle(img, (bbox.xmin, bbox.ymin),
                      (bbox.xmax, bbox.ymax),
                      (255, 0, 0), 2)
    return img
