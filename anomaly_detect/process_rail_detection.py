from anomaly_detect.utils import BBox, read_yolo_labels
import xml.etree.cElementTree as ET
from pathlib import Path
from logzero import logger

annos = read_yolo_labels('../yolov5/runs/detect/rail_rail/labels/', '../yolov5/data/rail_rail.yaml', conf_threshold=0.4)

def create_obj(bbox):
    obj = ET.Element('object')
    ET.SubElement(obj, 'name').text = 'rail'
    box = ET.SubElement(obj, 'bndbox')
    ET.SubElement(box, 'xmin').text = str(int(bbox.xmin))
    ET.SubElement(box, 'ymin').text = str(int(bbox.ymin))
    ET.SubElement(box, 'xmax').text = str(int(bbox.xmax))
    ET.SubElement(box, 'ymax').text = str(int(bbox.ymax))
    return obj

def merge_rail(bboxes):
    if len(bboxes) == 1:
        return BBox(bboxes[0].xmin, 0, bboxes[0].xmax, 1)
    if len(bboxes) == 2:
        if abs(bboxes[0].center[0] - bboxes[1].center[0]) < 0.1:
            return BBox(min(bboxes[0].xmin, bboxes[1].xmin), 0,
                        max(bboxes[0].xmax, bboxes[1].xmax), 1)
    return None

anno_dir = Path('/home/lucien/datasets/koujian_data/1k_annotations')
out_dir = Path('/home/lucien/datasets/koujian_data/1k_annotations_with_rail')
if out_dir.exists():
    raise ValueError('out_dir already exists')
out_dir.mkdir(parents=True)

no_detect = 0
no_rail = 0
for fp in anno_dir.iterdir():
    anno = annos.get(fp.stem)
    if not anno:
        logger.error(f'cannot find anything in {fp}')
        no_detect += 1
        continue
    rail_bbox = merge_rail(anno.get('normal_rail'))
    if not rail_bbox:
        logger.error(f'cannot find any rail in {fp}')
        no_rail += 1
        continue
    rail_bbox = BBox(rail_bbox.xmin * 1024, rail_bbox.ymin * 1024, rail_bbox.xmax * 1024, rail_bbox.ymax * 1024)
    obj = create_obj(rail_bbox)
    tree = ET.parse(fp)
    root = tree.getroot()
    root.insert(0, obj)
    out = out_dir / f'{fp.stem}.xml'
    tree.write(out)
logger.info(f'no detect: {no_detect}, no rail: {no_rail}')
