import argparse
import enum
import cv2
import itertools

from logzero import logger
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from anomaly_detect.components import PositionStructure
from anomaly_detect.utils import BBox, balanced_split, read_voc_labels


train_ratio = 0.8

CATS = ['fatigue_block', 'stripping_off_block', 'poor_light_band', 'corrugation', 'abnormal_rail_gap']


def parse_args():
    parser = ArgumentParser(
        description='detect anomaly from yolo results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--annos', type=str, required=True, help='path to annotation file')
    parser.add_argument('--imgs', type=str, required=True, help='path to image folder')
    parser.add_argument('--crop', type=str, required=True, help='name of cropped component')
    parser.add_argument('--output', type=str, required=True, help='path to output folder')
    return parser.parse_args()


def crop_img(img_path, annos, crop):
    ret = []
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f'failed to read image {img_path}')
        return ret
    ps = PositionStructure(annos)
    for cropped in ps.get_components_by_cat(crop):
        cropped_annos = {}
        bbox = cropped.bbox
        xmin = int(bbox.xmin)
        ymin = int(bbox.ymin)
        xmax = int(bbox.xmax)
        ymax = int(bbox.ymax)
        cropped_img = img[ymin:ymax, xmin:xmax, :]
        h, w = cropped_img.shape[:2]
        for child in cropped.get_children():
            if child.cat not in CATS:
                continue
            cropped_annos[child.cat] = BBox(
                max(child.bbox.xmin - bbox.xmin, 0),
                max(child.bbox.ymin - bbox.ymin, 0),
                min(child.bbox.xmax - bbox.xmin, w),
                min(child.bbox.ymax - bbox.ymin, h)
            )
        ret.append([cropped_img, cropped_annos])
    return ret


def create_dataset(annos, train, val, args):
    img_dir = Path(args.imgs)
    output_dir = Path(args.output)
    if output_dir.exists():
        raise ValueError(f'output dir {output_dir} already exists')
    output_dir.mkdir(parents=True)
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir()
    output_label_dir.mkdir()
    for split, img_names in zip(['train', 'val'], [train, val]):
        split_img_dir = output_img_dir / split
        split_label_dir = output_label_dir / split
        split_img_dir.mkdir()
        split_label_dir.mkdir()
        for img_name in tqdm(img_names, desc=f'processing {split} images'):
            img_path = img_dir / f'{img_name}.jpg'
            if not img_path.exists():
                img_path = img_dir / f'{img_name}.png'
            if not img_path.exists():
                logger.error(f'cannot find image {img_name}')
            label = annos[img_name]
            cropped_imgs_annos = crop_img(img_path, label, args.crop)
            if cropped_imgs_annos is None:
                logger.error(f'failed to crop image {img_name}')
                continue
            for idx, (img, cropped_annos) in enumerate(cropped_imgs_annos):
                h, w = img.shape[:2]
                new_img_name = f'{img_name}_{idx}'
                cv2.imwrite(str(split_img_dir / f'{new_img_name}.jpg'), img)
                for cat, bbox in cropped_annos.items():
                    center_x = (bbox.xmin + bbox.xmax) / 2 / w
                    center_y = (bbox.ymin + bbox.ymax) / 2 / h
                    bbox_w = (bbox.xmax - bbox.xmin) / w
                    bbox_h = (bbox.ymax - bbox.ymin) / h
                    with open(split_label_dir / f'{new_img_name}.txt', 'w') as f:
                        f.write(f'{CATS.index(cat)} {center_x} {center_y} {bbox_w} {bbox_h}')
    yaml_file = output_dir / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(f"""\
train: {output_img_dir / 'train'}
val: {output_img_dir / 'val'}

nc: {len(CATS)}
names: {CATS}
                """)


def main(args):
    annos = read_voc_labels(args.annos)
    annos = dict(
        (k, v) for k, v in annos.items() if args.crop in v
    )
    cropped_annos = {}
    for img, anno in tqdm(annos.items()):
        cropped_anno = defaultdict(list)
        ps = PositionStructure(anno)
        cropped_components = ps.get_components_by_cat(args.crop)
        for component in cropped_components:
            cropped_anno[args.crop].append(component.bbox)
            for child in component.get_children():
                cropped_anno[child.cat].append(child.bbox)
        cropped_annos[img] = cropped_anno
    train_imgs, val_imgs = balanced_split(cropped_annos, train_ratio)
    create_dataset(cropped_annos, train_imgs, val_imgs, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
