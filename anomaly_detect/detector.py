
from abc import ABC
from typing import List

from class_registry.registry import abstract_method
from logzero import logger

from components import PositionStructure
from class_registry import ClassRegistry, ClassRegistryInstanceCache

from utils import BBox


detectors = ClassRegistry()
detectors_factory = ClassRegistryInstanceCache(detectors)


class Detector(ABC):
    @abstract_method
    def detect(self, ps: PositionStructure, img_name: str) -> List[BBox]:
        pass


@detectors.register('ballastbed_crack')
class FasteningOppositeDetector(Detector):
    def detect(self, ps: PositionStructure, img_name: str) -> List[BBox]:
        ret = []
        cracks = ps.get_components_by_cat('crack')
        if not cracks:
            return ret

        for crack in cracks:
            # if crack.parent is None:
            #     ret.append(crack.bbox)
            if crack.parent is None or crack.parent.cat != 'sleeper':
                ret.append(crack.bbox)
        return ret


# @detectors.register('fastening_oppsite')
class FasteningOppositeDetector(Detector):
    def detect(self, ps: PositionStructure, img_name: str) -> List[BBox]:
        ret = []
        rails = ps.get_components_by_cat('rail')
        if not rails:
            logger.debug('rail not found: {}'.format(img_name))
            return ret
        if len(rails) != 1:
            logger.debug('found more than one rail: {}'.format(img_name))
            return ret
        if ps.get_components_by_cat('turnout'):
            logger.debug('turnout found: {}'.format(img_name))
            return ret
        rail = rails[0]

        bolt_ins = ps.get_components_by_cat('bolt_in')
        for bolt_in in bolt_ins:
            for opening in ps.get_components_by_cat('opening'):
                if abs(opening.bbox.center[0] - bolt_in.bbox.center[0]) > 0.05:
                    continue
                logger.debug(opening.bbox.center)
                logger.debug(min(bolt_in.bbox.center[0], rail.bbox.center[0]))
                logger.debug(max(bolt_in.bbox.center[0], rail.bbox.center[0]))
                if opening.bbox.center[0] < min(bolt_in.bbox.center[0], rail.bbox.center[0]) \
                        or opening.bbox.center[0] > max(bolt_in.bbox.center[0], rail.bbox.center[0]):
                    ret.append(1)
        return ret


# @detectors.register('fastening_lack')
class FasteningLackDetector(Detector):
    def detect(self, ps: PositionStructure, img_name: str) -> List[BBox]:
        ret = []
        rails = ps.get_components_by_cat('rail')
        if not rails:
            logger.debug('rail not found: {}'.format(img_name))
            return ret
        if len(rails) != 1:
            logger.debug('found more than one rail: {}'.format(img_name))
            return ret
        if ps.get_components_by_cat('turnout'):
            logger.debug('turnout found: {}'.format(img_name))
            return ret

        rail = rails[0]
        rail_center = rail.bbox.center
        spikes = ps.get_components_by_cat('spike_8B')
        for spike in spikes:
            # if spike.ymin < 0.1 or spike.ymax > 0.9:
            #     continue
            spike_x, spike_y = spike.bbox.center
            height = spike.bbox.height
            missed = True
            right = True
            if spike_x < rail_center[0]:
                right = False
            region = BBox(
                0 if right else rail_center[0],
                spike_y - height if right else spike_y - height - 30,
                rail_center[0] if right else 1,
                spike_y + height + 30 if right else spike_y + height
            )
            for opposite_component in ps.get_components_by_bbox(region):
                logger.debug('%s: %s', spike, opposite_component.cat)
                if opposite_component.cat == 'spike_8B' or \
                        opposite_component.cat == 'component' or \
                        opposite_component.cat == 'opening' or \
                        opposite_component.cat == 'bolt_in':
                    missed = False
                    break
            if missed:
                ret.append(BBox(
                    2 * rail_center[0] - spike.bbox.xmin,
                    spike.bbox.ymin,
                    2 * rail_center[0] - spike.bbox.xmax,
                    spike.bbox.ymax
                ))

        def check_exists(comps):
            for cat in ['spike_8B', 'bolt_in', 'component', 'opening', 'a', 'b', 'plate']:
                if len([c for c in comps if c.cat == cat]) > 0:
                    return True
            return False

        sleepers = ps.get_components_by_cat('sleeper')
        for sleeper in sleepers:
            if not sleeper.bbox.intersected(rail.bbox):
                continue
            left_comps = ps.get_components_by_bbox(BBox(
                sleeper.bbox.xmin,
                sleeper.bbox.ymin,
                rail.bbox.center[0],
                sleeper.bbox.ymax
            ))
            right_comps = ps.get_components_by_bbox(BBox(
                rail.bbox.center[0],
                sleeper.bbox.ymin,
                sleeper.bbox.xmax,
                sleeper.bbox.ymax
            ))
            logger.debug('%s\n%s', left_comps, right_comps)
            if not check_exists(left_comps) or not check_exists(right_comps):
                ret.append(1)
        return ret
