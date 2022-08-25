# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import json

dataset = "/home/raiyaan/VisDrone/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations"
images = "/home/raiyaan/VisDrone/VisDrone2019-DET-train/VisDrone2019-DET-train/images"

def load_visdrone(name, thing_classes):
    data = []
    
    count = 0
    
    for file in os.listdir(dataset):
        file_path=os.path.join(dataset, file)
        
        annotation_file= open(file_path,'r')
        #print (file)
        lines = annotation_file.readlines()
        image_file = file[:-3]+"jpg"
        image_file_path=os.path.join(images, image_file)
        
        
        im = cv2.imread(image_file_path)

        image_annotation = {
            "file_name" : image_file_path, # full path to image
            "image_id" :  count, # image unique ID
            "height" : im.shape[0], # height of image
            "width" : im.shape[1], # width of image 
        }
        
        annotations = []
        
        for line in lines:
            
            array=line.strip().split(",")
            #print(array)
            if len(array) > 8: #some annotations have an extra comma at the end
                array.pop()
            array = [int(i) for i in array]
   
            #print(array)
            
            if array[5] != 0:

                single_annotation = {
                    "category_id" : array[5]-1,#ignored regions are omitted so -1
                    "bbox" : [array[0], array[1], array[2], array[3]], # bbox coordinates
                    "bbox_mode" : BoxMode.XYWH_ABS, # bbox mode, depending on your format
                    
                }
            annotations.append(single_annotation)    
        image_annotation["annotations"] = annotations
            
        #ignored regions are omitted
        data.append(image_annotation)
        count += 1
    # the json file where the output is stored
    out_file = open("visdrone-meta-file.json", "w")
    json.dump(data, out_file, indent = 6)
    out_file.close()
    return data

def register_meta_visdrone(name, thing_classes, metadata):
    # register dataset (step 1)
    DatasetCatalog.register(
        name, # name of dataset, this will be used in the config file
        lambda: load_visdrone( # this calls your dataset loader to get the data
            name, thing_classes, # inputs to your dataset loader
        ),
    )

    # register meta information (step 2)
    MetadataCatalog.get(name).set(
        thing_classes=metadata["thing_classes"], # all classes
        base_classes=metadata["base_classes"], # base classes
        novel_classes=metadata["novel_classes"], # novel classes
    )
    MetadataCatalog.get(name).evaluator_type = "visdrone" # set evaluator
    

    



