"""
Evaluation detection model
Author: noitq
"""

import torch
import mlflow
import mlflow.onnx
import argparse
import yaml
from tqdm import tqdm 
from pathlib import Path
import numpy as np

from model import DetectionModel
from dataset import create_dataloader
from utils import select_device, scale_coords, clip_coords, box_iou, xywh2xyxy, xyxy2xywh, batch_stats, metrics_from_preds

mlflow.set_tracking_uri('http://mlflow-tracking.vinbrain.net:8899')
DETECTION_EXPERIMENT_ID = 3
RUN_NAME = "Eval detection"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolov5_onnx', help='Model name')
    parser.add_argument('--model_version', type=int, default=1, help='Model version registered on Mlflow server')
    parser.add_argument('--data', type=str, default='/u01/data/bdd100k/det/bdd100k.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--device", type=str, default= 'cpu')
    parser.add_argument("--batch_size", type=int, default= 32)
    parser.add_argument("--input_size", type=int, default= 640)
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_txt', action='store_true')
    parser.add_argument('--save_conf', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--single_cls', action='store_true')
    opt = parser.parse_args()
    device = select_device(opt.device)
    
    """
    Init model
    """
    od = DetectionModel(opt.model_name, opt.model_version, opt)
    
    """
    Init data
    """
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    path = data['test']
    nc = 1 if opt.single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    dataloader = create_dataloader(path, opt.input_size, opt.batch_size, 32, opt, pad=0.5, rect=False)[0]
    
    """
    Make inference
    """
    with mlflow.start_run(experiment_id=DETECTION_EXPERIMENT_ID, run_name=RUN_NAME):
        seen = 0
        stats = []
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
            img, targets = img.to(device), targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            
            # run model
            output = od.predict(img)
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            
            # batch scores
            batch_results = batch_stats(img, output, targets, paths, shapes, iouv, niou, device)
            seen += len(output)
            stats.extend(batch_results)
    
            
        # get metrics
        nt, mp, mr, map50, map, f1, ap_class = metrics_from_preds(stats)

        # Print results
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        
        # log_metric to server
        mlflow.log_metric('mp', mp)
        mlflow.log_metric('mr', mr)
        mlflow.log_metric('map50', map50)
        mlflow.log_metric('map', map)


                
            
                
    
    