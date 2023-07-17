# DeepView

This repository is an implementation of the paper: **Prioritizing Testing Instances to Enhance the Robustness of
Object Detection Systems**

**DeepView** is an instance-level test prioritization tool for object detection models to reduce data annotation costs.

![overview](./data/overview.Png) 

Object detection models have been widely deployed in military and life-related intelligent software systems. However, along with the outstanding success of object detection, it may exhibit abnormal behavior and lead to severe accidents and losses. During the development and evaluation process, training and evaluating an object detection model are computationally intensive, while preparing annotated tests requires extremely heavy manual labor. Therefore, reducing the annotation-budget of test data collection becomes a challenging and necessary task. Although many test prioritization approaches for DNN-based systems have been proposed, the large differences between classification and object detection make them difficult to be applied to the testing of object detection models.

In this paper, we propose **DeepView**, a novel instance-level test prioritization tool for object detection models to reduce data annotation costs. **DeepView** is based on splitting object detection results into instances and calculating the capability of locating and classifying an instance, respectively. We further designed a test prioritization tool that enables testers to improve model performance by focusing on instances that may cause model errors from a large unlabeled dataset. To evaluate DeepView, we conduct an extensive empirical study on two kinds of object detection model architectures and two commonly used datasets. The experimental results show that **DeepView** outperforms existing test prioritization approaches regarding effectiveness and diversity. Also, we observe that using **DeepView** can effectively improve the accuracy and robustness of object detection models.


## Installation
`pip install -r requirements.txt`

## Usage
We prepare a demo for **DeepView**:
+ `python demo.py`

In order to run this demo, you should first download [coco2017val](https://cocodataset.org/#download) dataset in the data folder

## Output and Annotation
`deepview_result.json` is the output of **DeepView**, 
which is a prioritized set of instances that you can map 
top-k instances back to the original image according to your 
annotation budget and highlight them to provide to the annotator

The annotator only needs to focus on the highlighted 
areas and annotate them 

Some examples are shown below:

<div><table frame=void>	<!--用了<div>进行封装-->
	<tr>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./data/example_1.png"
                 alt="example_1"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        		<!--标题1-->
        </center></div></td>    
     	<td><div><center>	<!--第二张图片-->
    		<img src="./data/example_2.png"
                 alt="example_2"
                 height="120"/>	
    		<br>
    		 <!--标题1-->
        </center></div></td>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./data/example_3.png"
                 alt="example_3"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        		<!--标题1-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./data/example_4.png"
                 alt="example_4"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        		<!--标题1-->
        </center></div></td> 

</table></div>

You can modify the code in the `demo.py` to run **DeepView** on other datasets and models.
