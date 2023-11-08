from collections import defaultdict, OrderedDict
from pycocotools.cocoeval import COCOeval as _COCOeval
import mmcv
from .builder import DATASETS
from . import coco
from mmdet.core import eval_map, eval_recalls
from mmcv.utils import print_log
import numpy as np
import logging

class UAVDTEval(_COCOeval):
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params

        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # store ignored/iscrowd anns of gts in a dictionary 
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0 # complete 'ignore' key to all entries
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd'] # match 'ignore' with 'iscrowd'
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        
        self._igs = defaultdict(list)       # mask for evaluation
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        
        #store masks
        for gt in gts:
            if gt['ignore']:
                self._igs[gt['image_id']].append(gt)
            else:
                gt['category_id'] = 0
                self._gts[gt['image_id'], gt['category_id']].append(gt)
    

        #remove the detections that fall in the ignored regions
        for dt in dts:
            if dt['image_id'] in self._igs:
                if all([dt['bbox'][0] <= ig['bbox'][0] or\
                        dt['bbox'][1] <= ig['bbox'][1] or\
                        dt['bbox'][0] + dt['bbox'][2] >= ig['bbox'][0] + ig['bbox'][2] or\
                        dt['bbox'][1] + dt['bbox'][3] >= ig['bbox'][1] + ig['bbox'][3]
                        for ig in self._igs[dt['image_id']]]):
                    dt['category_id'] = 0
                    self._dts[dt['image_id'], dt['category_id']].append(dt)
            else:
                dt['category_id'] = 0
                self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        #VOC evaluate for coco dataset

        self._prepare()

        gts_voc = OrderedDict()
        dts_voc = OrderedDict()

        for key, gts_img in self._gts.items():
            img, cat = key
            img_gt = defaultdict(list)
            bboxes_gt = []
            bboxes_dt = []
            cats_gt = []
            bboxes_ignore_gt = []
            cats_ignore_gt = []
            for gt in gts_img:
                bboxes_gt.append([gt['bbox'][0], 
                                  gt['bbox'][1], 
                                  gt['bbox'][2] + gt['bbox'][0], 
                                  gt['bbox'][3] + gt['bbox'][1],])
                cats_gt.append(cat)

            for dt in self._dts[img, cat]:
                bboxes_dt.append([dt['bbox'][0], 
                                  dt['bbox'][1], 
                                  dt['bbox'][2] + dt['bbox'][0], 
                                  dt['bbox'][3] + dt['bbox'][1], dt['score']])

            # If there is no prediction, add an all-zero list
            if not bboxes_dt:
                bboxes_dt.append([0, 0, 0, 0, 0])

            img_gt['bboxes'] = np.array(bboxes_gt)
            img_gt['labels'] = np.array(cats_gt)

            if self.include_ignore and img in self._igs:
                for ig in self._igs[img]:
                    bboxes_ignore_gt.append([ig['bbox'][0], 
                                    ig['bbox'][1], 
                                    ig['bbox'][2] + ig['bbox'][0], 
                                    ig['bbox'][3] + ig['bbox'][1],])
                    cats_ignore_gt.append(0)
                img_gt['bboxes_ignore'] = np.array(bboxes_ignore_gt)
                img_gt['labels_ignore'] = np.array(cats_ignore_gt)

            gts_voc[img] = img_gt
            dts_voc[img] = np.array([bboxes_dt,])

        annotations = list(gts_voc.values())
        results = list(dts_voc.values())


        metric = 'mAP'
        allowed_metrics = ['mAP', 'recall']
        iou_thr = .7
        proposal_nums=(100, 300, 1000)

        eval_results = OrderedDict()

        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            ds_name = 'uavdt'
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger = None,
                    use_legacy_coordinate=False)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 4)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)

        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger = None,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

# Uncomment to enable UAVDT evaluation
# coco.COCOeval = UAVDTEval

@DATASETS.register_module()
class UavdtDataset(coco.CocoDataset):

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 include_ignore=False):
        if isinstance(metric, list):
            assert len(metric) == 1, "Only 'bbox' is supported"
            metric = metric[0]
        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, _ = self.format_results(results, jsonfile_prefix)

        try:
            predictions = mmcv.load(result_files[metric])
            coco_det = coco_gt.loadRes(predictions)
        except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)

        iou_type = metric
        cocoEval = coco.COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        cocoEval.include_ignore = include_ignore

        return cocoEval.evaluate()
