import os
import cv2
import numpy as np
import torch
import onnxruntime

import mlflow
import mlflow.onnx
from utils import change_input_dim, non_max_suppression
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "onnx"

class DetectionModel:
    def __init__(self, model_name, model_version, opt):
        self.opt = opt

        local_model_path = _download_artifact_from_uri(artifact_uri=f"models:/{model_name}/{model_version}")
        flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
        onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
        
        self.model = onnxruntime.InferenceSession(onnx_model_artifacts_path)

    def predict(self, img):
        """
        Make inference
        """
        onnx_input = self.preprocess(img)
        ort_inputs = {self.model.get_inputs()[0].name: onnx_input}
        out = self.model.run(None, ort_inputs)
        
        return self.postprocess(out)
    
    def preprocess(self, img):
        """
        pre processing
        """
        # pass your image preprocessing here
        img = img.cpu().numpy().astype(np.float32)
        img /= 255.0
        
        return img

    def postprocess(self, out):
        """
        post processing
        """
        # pass your image postprocessing here
        out = torch.from_numpy(out[0])
        output = non_max_suppression(out, conf_thres=self.opt.conf_thres, iou_thres=self.opt.iou_thres)
        
        return out