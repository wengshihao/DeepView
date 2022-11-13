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