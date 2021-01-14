import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures import Instances, Boxes


class FRCNN_FPN_Detectron2(DefaultPredictor):
    def __init__(self, cfg, model_path, softmax_only_person=True):
        # Prints wayyy too many things when logger is set to info.
        logging.getLogger('detectron2.checkpoint.c2_model_loading').setLevel(
            logging.WARNING)
        if isinstance(cfg, (str, Path)):
            cfg_obj = get_cfg()
            cfg_obj.merge_from_file(cfg)
            cfg_obj.MODEL.WEIGHTS = model_path
            cfg = cfg_obj
        super().__init__(cfg)
        self.softmax_only_person = softmax_only_person
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if self.num_classes == 80:  # COCO
            self.person_class = 0
        elif self.num_classes == 1230:  # LVIS
            self.person_class = 804
        else:
            raise NotImplementedError(
                f'Unknown number of classes {self.num_classes}.')

    def cuda(self):
        return self.model.cuda()

    def eval(self):
        self.model.eval()

    def to_numpy(self, img):
        # Ugly, but easiest to implement
        assert img.shape[0] == 1
        return np.asarray(to_pil_image(img[0]))

    def detect(self, img):
        img = self.to_numpy(img)[:, :, ::-1]  # to BGR
        detections = self(img)

        # Only use labels for person
        instances = detections['instances']
        instances = instances[instances.pred_classes == self.person_class]
        boxes = instances.pred_boxes.tensor.detach()
        scores = instances.scores.detach()
        return boxes, scores

    def load_pretrained(self):
        raise NotImplementedError

    def _transform_image_and_boxes(self, img, boxes):
        """
        Args:
            images (Tensor)
            boxes (Tensor)
        """
        transform = self.transform_gen.get_transform(img)
        transformed_img = transform.apply_image(img)
        scale_y = transform.new_h / transform.h
        scale_x = transform.new_w / transform.w
        new_boxes = Instances((transform.new_h, transform.new_w))
        new_boxes.proposal_boxes = Boxes(boxes)
        new_boxes.proposal_boxes.scale(scale_x, scale_y)
        new_boxes.objectness_logits = torch.ones(len(boxes))
        transformed_img = torch.as_tensor(
            transformed_img.astype("float32").transpose(2, 0, 1))
        return transformed_img, new_boxes

    def predict_boxes(self, images, boxes):
        assert images.shape[0] == 1
        img = images
        img = self.to_numpy(img)[:, :, ::-1]  # to BGR
        original_size = img.shape[:2]
        try:
            img, proposals = self._transform_image_and_boxes(img, boxes)
        except BaseException as e:
            print(e)
            import ipdb; ipdb.set_trace()

        device = self.model.device
        img = img.to(device)
        proposals = [proposals.to(device)]
        inputs = [{
            'image': img,
            'proposals': proposals,
            'height': original_size[0],
            'width': original_size[1]
        }]

        model = self.model
        roi_heads = self.model.roi_heads

        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        features = [features[f] for f in roi_heads.in_features]
        box_features = roi_heads.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = roi_heads.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = roi_heads.box_predictor(
            box_features)

        outputs = FastRCNNOutputs(
            roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            roi_heads.smooth_l1_beta,
        )
        pred_boxes = outputs.predict_boxes()[0]

        c = self.person_class

        if self.softmax_only_person:
            scores = pred_class_logits[:, [c, -1]].detach()
            scores = F.softmax(scores, -1)[:, 0]
        else:
            scores = F.softmax(pred_class_logits, -1)
            scores = scores[:, c].detach()
        boxes = pred_boxes[:, c * 4:(c + 1) * 4].detach()

        scale_y = original_size[0] / img.shape[1]
        scale_x = original_size[1] / img.shape[2]
        boxes = Boxes(boxes)
        boxes.scale(scale_x, scale_y)
        boxes = boxes.tensor
        return boxes, scores

    def load_image(self, img):
        pass
