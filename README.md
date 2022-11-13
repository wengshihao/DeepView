# DeepView

This repository is an implementation of the paper: **Prioritizing Testing Instances to Enhance the Robustness of
Object Detection Systems**

**DeepView** is an instance-level test prioritization tool for object detection models to reduce data annotation costs.

![overview](./data/overview.Png) 

## Installation
`pip install -r requirements.txt`

## Usage
We prepare a demo for **DeepView**:
+ `python demo.py`

In order to run this demo, you should first download [coco2017val](https://cocodataset.org/#download) dataset in the data folder

## Output
`deepview_result.json` is the output of **DeepView**, which 
is a prioritized set of instances that you can 
map top-$k$ instances back to the original image 
to provide to the annotator based on your annotation 
budget

Some examples are shown below: