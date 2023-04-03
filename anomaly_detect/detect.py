import pickle
import shutil
from typing import Dict, List
import cv2
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import logging
import logzero
import pandas as pd

from logzero import logger
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from scipy.spatial import KDTree, kdtree
from components import PositionStructure

from utils import BBox, draw_bboxes, get_annotations, read_voc_labels, read_yolo_labels, BBox
from detector import detectors, detectors_factory

extra_conf_thres = {
    'turnout': 0.4,
    'sleeper': 0.7
}


cat_mapper = {
    'normal_spike_w': 'spike_8B'
}


def parse_args():
    parser = ArgumentParser(
        description='detect anomaly from yolo results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--yolo-data', type=str, required=True, help='path to yolo data file')
    parser.add_argument('--yolo-detect-labels', required=True, type=str, help='path to dir that contains detected labels')
    parser.add_argument('--gt-labels', required=True, type=str, help='path to dir that contains ground truth labels')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--imgs', nargs='+', type=str, default=None, help='imgs to be detected, just for debug')
    parser.add_argument('--save', type=str, default=None, help='save detected labels to this dir')
    parser.add_argument('--cache', action='store_true', help='use cache')
    return parser.parse_args()


def merge_labels(preds, gts):
    merged = defaultdict(list)
    for img, annos in gts.items():
        if img not in preds:
            continue
        merged[img] = [annos, preds[img]]
    return merged

def main(args):
    # contains all detected images
    labeled_preds_dir = Path(args.yolo_detect_labels).parent

    if args.cache:
        with open('./pred.pkl', 'rb') as f:
            preds = pickle.load(f)
        with open('./gt.pkl', 'rb') as f:
            gts = pickle.load(f)
    else:
        preds = read_yolo_labels(args.yolo_detect_labels, args.yolo_data, cat_mapper,
                                 conf_threshold=args.conf_thres,
                                 extra_conf_thres=extra_conf_thres)
        gts = read_voc_labels(args.gt_labels)
    if not Path('./pred.pkl').exists():
        with open('./pred.pkl', 'wb') as f:
            pickle.dump(preds, f)
        with open('./gt.pkl', 'wb') as f:
            pickle.dump(gts, f)
    merged_labels = merge_labels(preds, gts)
    if args.imgs is not None:
       merged_labels = {img: merged_labels[img] for img in args.imgs}

    save_paths = dict()
    if args.save:
        save_dir = Path(args.save)
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        for cat in detectors.keys():
            path = save_dir / cat
            path.mkdir(exist_ok=True, parents=True)
            (path / 'fn').mkdir(exist_ok=True, parents=True)
            (path / 'fp').mkdir(exist_ok=True, parents=True)
            save_paths[cat] = path

    tp = dict()
    fp = dict()
    fn = dict()
    for img_name, (gt, pred) in tqdm(merged_labels.items()):
        ps = PositionStructure(pred)
        for cat in detectors.keys():
            if cat not in tp:
                tp[cat] = 0
                fp[cat] = 0
                fn[cat] = 0
            detector = detectors_factory[cat]
            anomaly_gt = gt.get(cat)
            anomaly_pred = detector.detect(ps, img_name)
            if anomaly_gt and anomaly_pred:
                tp[cat] += 1
                logger.debug('detect %s tp %s', cat, img_name)
            if anomaly_gt and not anomaly_pred:
                fn[cat] += 1
                logger.debug('detect %s fn %s', cat, img_name)
                if args.save:
                    img_path = labeled_preds_dir / f'{img_name}.jpg'
                    if not img_path.exists():
                        img_path = labeled_preds_dir / f'{img_name}.png'
                    img = cv2.imread(str(img_path))
                    img = draw_bboxes(img, anomaly_gt)
                    cv2.imwrite(str(save_paths[cat] / 'fn' / img_path.name), img)
            if not anomaly_gt and anomaly_pred:
                fp[cat] += 1
                logger.debug('detect %s fp %s', cat, img_name)
                if args.save:
                    img_path = labeled_preds_dir / f'{img_name}.jpg'
                    if not img_path.exists():
                        img_path = labeled_preds_dir / f'{img_name}.png'
                    shutil.copy2(str(img_path), str(save_paths[cat] / 'fp' / img_path.name))

    df = []
    for cat in tp.keys():
        df.append([cat, tp[cat], fp[cat], fn[cat], tp[cat] / (tp[cat] + (fp[cat] + fn[cat] / 2))])
    df = pd.DataFrame(df, columns=['category', 'true positive', 'false positive', 'false negative', 'f1'])
    logger.info('Results:\n%s', df.to_markdown())


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(level=logging.INFO)
    main(args)
