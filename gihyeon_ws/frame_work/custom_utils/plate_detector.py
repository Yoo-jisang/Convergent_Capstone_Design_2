import torch, cv2
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


class detector():
    def __init__(self, device='cpu', conf_thres=0.25, iou_thres=0.45, margin_gain = 0.1):
        device = select_device(device)

        # yolo model
        weights = './yolo_data/model_weights/best.pt'
        data = './yolo_data/class_yaml/collection.yaml'
        dnn = False
        half = False
        self.augment = False
        self.visualize = False
        imgsz = (640, 640)

        # nms
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = [0]
        self.agnostic_nms = False # class nms
        self.max_det = 1000
        self.margin_gain = margin_gain

        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # warm up
        self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *imgsz))  # warmup

    def np_to_tensor(self, im):
        im = cv2.resize(im, (640, 640))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.FloatTensor(im.transpose(2, 0, 1)/255).unsqueeze(0)
        return im
    
    def extract_plate_region(self, origin_im, pred):
        h, w = origin_im.shape[0], origin_im.shape[1]

        det_list = []
        for det in pred[0]:
            x1, y1, x2, y2, _, _ = det

            x1, x2 = int(x1.item()/640*w), int(x2.item()/640*w)
            y1, y2 = int(y1.item()/640*h), int(y2.item()/640*h)
            det_list.append([x1, y1, x2, y2])

        return det_list

    def crop_plate_with_margin(self, origin_im, det_list):
        result = []

        for i, det in enumerate(det_list):
            x1, y1, x2, y2 = det

            w = origin_im.shape[0]
            h = origin_im.shape[1]

            w_local, h_local = (x2 - x1), (y2 - y1)

            x1 = int(x1 - w_local*self.margin_gain)
            x2 = int(x2 + w_local*self.margin_gain)

            y1 = int(y1 - h_local*self.margin_gain)
            y2 = int(y2 + h_local*self.margin_gain)

            # 예외 처리(인덱스를 넘어갈 수 도 있으므로...)
            # x1 = max(x1, 0)
            # y1 = max(y1, 0)

            # x2 = min(x2, w-1)
            # y2 = min(y2, h-1)

            result.append(origin_im[y1:y2, x1:x2, :])
            # cv2.imwrite('./test'+str(i)+'.jpg', origin_im[y1:y2, x1:x2, :])
        return result
    
    def predict(self, im):
        origin_im = im.copy()

        im = self.np_to_tensor(im)
        
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        det_list = self.extract_plate_region(origin_im, pred)

        result = self.crop_plate_with_margin(origin_im, det_list)

        return result
    