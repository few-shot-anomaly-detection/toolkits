from typing import Dict, List
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
from tqdm import tqdm

from utils import BBox, get_annotations, read_voc_labels, read_yolo_labels, BBox
from detector import detectors, detectors_factory


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
    parser.add_argument('--cats', nargs='+', type=str, required=True, help='categories to be calculated')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--imgs', nargs='+', type=str, default=None, help='imgs to be detected, just for debug')
    return parser.parse_args()


def merge_labels(preds, gts):
    merged = defaultdict(list)
    for img, annos in gts.items():
        if img not in preds:
            logger.warning(f'img {img} not in preds')
            continue
        merged[img] = [annos, preds[img]]
    return merged

def main(args):
    preds = read_yolo_labels(args.yolo_detect_labels, args.yolo_data, cat_mapper, conf_threshold=args.conf_thres)
    gts = read_yolo_labels(args.gt_labels, args.yolo_data, cat_mapper)
    merged_labels = merge_labels(preds, gts)
    if args.imgs is not None:
       merged_labels = {img: merged_labels[img] for img in args.imgs}

    tp = dict()
    fp = dict()
    fn = dict()
    for img, (gt, pred) in tqdm(merged_labels.items()):
        for cat in args.cats:
            if cat not in tp:
                tp[cat] = 0
                fp[cat] = 0
                fn[cat] = 0
            anomaly_gt = gt.get(cat)
            anomaly_pred = pred.get(cat)
            if anomaly_gt and anomaly_pred:
                tp[cat] += 1
                # logger.info('detect %s tp %s', cat, img)
            if anomaly_gt and not anomaly_pred:
                fn[cat] += 1
                logger.info('detect %s fn %s', cat, img)
            if not anomaly_gt and anomaly_pred:
                fp[cat] += 1
                logger.info('detect %s fp %s', cat, img)

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
