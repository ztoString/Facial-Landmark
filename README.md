# Facial-Landmark
Ubantu-18.04.1虚拟机下基于caffe和opencv实现人脸五个关键点检测  

首先，满怀敬意地附上原作者的源码地址  [在这里](https://github.com/luoyetx/deep-landmark)  
下载数据集 [在这里](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)   

## 人脸关键点检测
### 目的：检测人脸的5个关键点：左眼、右眼、鼻子、左嘴角、右嘴角  

源码结构：
>facial-Landmark  
>>common  
>>>__init__.py  
>>>cnns.py  
>>>level.py  
>>>utils.py  

>>data  
>>>level1.py  
>>>level2.py  
>>>level3.py  
>>>utils.py  

>>test  
>>>draw_landmarkfortest.py  

#### 记得：  
1.下载数据集  
2.制作hdf5数据源(每8000-10000个数据做一个hdf5数据源)  
3.通过myprototxt脚本文件制作各关键点所需的deploy.prototxt、solver.prototxt、train.prototxt和train.sh文件  
4.修改文件路径  

#### 注意：在numpy>1.11.0版本中，程序会报错：TypeError: slice indices must be integers or None or have an __index__ method   
#### 因为numpy1.11.0以上版本不支持浮点数作为索引，需要对相应变量进行int()类型强制转换

### 中心思想：将人脸图像分3层level训练
#### 1.全局 -- 眼睛和鼻子 -- 鼻子和嘴巴
#### 2.对每一个关键点生成0.16和0.18的padding，进行微调
#### 3.继续对每一个关键点生成0.11和0.12的padding，进行微调

### Level1：
![Level1.jpg](https://github.com/ztoString/Facial-Landmark/raw/master/result-folder/level1-Aaron_Eckhart_0001.jpg.jpg)

### Level2：
![Level2.jpg](https://github.com/ztoString/Facial-Landmark/raw/master/result-folder/level1-Aaron_Peirsol_0001.jpg-level2-.jpg)

### Level3：
![Level3.jpg](https://github.com/ztoString/Facial-Landmark/raw/master/result-folder/level1-Aaron_Peirsol_0001.jpg-level2--level3.jpg)

