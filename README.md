# 一次简单的人脸识别
    
## 1. opencv的简单介绍

### 1.1. 如何获取opencv

[官方网址]("http://opencv.org")
  
使用pip安装(python2.7)  
`pip install opencv`

使用编译完成的wheel安装(python3)  
http://www.lfd.uci.edu/~gohlke/pythonlibs/


### 1.2. opencv是什么
  
OpenCV是一个基于BSD许可（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows、Android和Mac OS操作系统上。实现了图像处理和计算机视觉方面的很多通用算法

![OPENCV](http://opencv.org/assets/theme/logo.png)
  
### 1.3  为什么要使用opencv
  
1.基于c++接口封装,运行速度优越
2.__强大__ __应用领域广__
3.API接口稳定高效
  
### 1.4. 基于人工神经网络算法(机器学习) 提取haar特征值

利用opencv自带的haar training程序训练一个分类器，需要经过以下几个步骤：  

（1）收集训练样本：  

        `训练样本包括正样本和负样本。正样本，通俗点说，就是图片中只有你需要的目标。而负样本的图片只要其中不含有目标就可以了。但需要说明的是，负样本也并非随便选取的。例如，你需要检测的目标是汽车，那么正样本就应该是仅仅含有汽车的图片，而负样本显然不能是一些包含天空的，海洋的，风景的图片。因为你最终训练分类器的目的是检测汽车，而汽车应该出现在马路上。也就是说，分类器最终检测的图片应该是那些包含马路，交通标志，建筑物，广告牌，汽车，摩托车，三轮车，行人，自行车等在内的图片。很明显，这里的负样本应该是包含摩托车、三轮车、自行车、行人、路面、灌木丛、花草、交通标志、广告牌等。

        另外，需要提醒的是，adaboost方法也是机器学习中的一个经典算法，而机器学习算法的前提条件是，测试样本和训练样本独立同分布。所谓的独立同分布，可以简单理解为：训练样本要和最终的应用场合非常接近或者一致。否则，基于机器学习的算法并不能保证算法的有效性。此外，足够的训练样本（至少得几千张正样本、几千张负样本）也是保证训练算法有效性的一个前提条件。

        这里，假设所有的正样本都放在f:/pos文件夹下，所有的负样本都放在f:/neg文件夹下；`

（2）对所有的正样本进行尺寸归一化：  

`上一步收集到的正样本，有很多的尺寸大小，有的是200*300，有的是500*800...尺寸归一化的目的，就是把所有的图片都缩放到同一大小。比如，都缩放到50*60的大小。`

（3）生成正样本描述文件：  

        `所谓的正样本描述文件，其实就是一个文本文件，只不过，很多人喜欢将这个文件的后缀改成.dat而已。正样本描述文件中的内容包括：文件名 目标个数 目标在图片中的位置（x,y,width,height）`

典型的正样本描述文件如下所示：  
  
0.jpg 1 0 0 30 40  
  
1.jpg 1 0 0 30 40  
  
2.jpg 1 0 0 30 40  
  
.....

        `不难发现，正样本描述文件中，每一个正样本占一行，每一行以正样本图片开头，后面紧跟着该图片中正样本的数量（通常为1），以及正样本在图片中的位置

        假如，f:\pos文件夹下有5000个正样本图片，每个图片中仅有一个目标。那么，我们可以写程序（遍历文件夹中的所有图片文件，将文件名写入到文件中，将正样本在图片中的位置，大小都写入文件中）生成一个pos.dat文件作为正样本描述文件`

（4）创建正样本vec文件

    `由于haarTraining训练的时候需要输入的正样本是vec文件，所以需要使用createsamples程序来将正样本转换为vec文件。打开OpenCV安装目录下bin文件夹里面的名为createSamples（新版本的OpenCV里面改名为opencv_createSamples）的可执行程序。需要提醒的是，该程序应该通过命令行启动。并设置正样本所在的路径以及生成的正样本文件保存路劲（例如：f:\pos\pos.vec）。`

Createsamples程序的命令行参数：  
命令行参数：  
－vec <vec_file_name>  
`训练好的正样本的输出文件名。`  
－img<image_file_name>  
`源目标图片（例如：一个公司图标）`  
－bg<background_file_name>  
`背景描述文件。`  
－num<number_of_samples>    
`要产生的正样本的数量，和正样本图片数目相同。`    
－bgcolor<background_color>    
`背景色（假定当前图片为灰度图）。背景色制定了透明色。对于压缩图片，颜色方差量由bgthresh参数来指定。则在bgcolor－bgthresh和bgcolor＋bgthresh中间的像素被认为是透明的。`  
－bgthresh<background_color_threshold>  
－inv  
`如果指定，颜色会反色`  
－randinv  
`如果指定，颜色会任意反色`  
－maxidev<max_intensity_deviation>   
`背景色最大的偏离度。`  
－maxangel<max_x_rotation_angle>  
－maxangle<max_y_rotation_angle>    
－maxzangle<max_x_rotation_angle>  
`最大旋转角度，以弧度为单位。`  
－show  
`如果指定，每个样本会被显示出来，按下"esc"会关闭这一开关，即不显示样本图片，而创建过程继续。这是个有用的debug选项。`  
－w<sample_width>    
`输出样本的宽度（以像素为单位）`   
－h《sample_height》  
`输出样本的高度，以像素为单位`    
   
（5） 创建负样本描述文件   

`在保存负样本的文件夹下生成一个负样本描述文件，具体步骤同（3），此处不再赘叙`   

（6）进行样本训练    
`该步骤通过调用OpenCV\bin目录下的haartraining程序(新版本的opencv改名为opencv_haartraining)来完成。其中，Haartraining的命令行参数为：`
－data<dir_name>   
`存放训练好的分类器的路径名。`  
－vec<vec_file_name>    
`正样本文件名（由trainingssamples程序或者由其他的方法创建的）`    
－bg<background_file_name>   
`背景描述文件。`   
－npos<number_of_positive_samples>     
－nneg<number_of_negative_samples>    
`用来训练每一个分类器阶段的正/负样本。合理的值是：nPos = 7000;nNeg = 3000 `  
－nstages<number_of_stages>   
`训练的级联分类器层数。  `  
－nsplits<number_of_splits>   
`决定用于阶段分类器的弱分类器。如果1，则一个简单的stump classifier被使用。如果是2或者更多，则带有number_of_splits个内部节点的CART分类器被使用。`    
－mem<memory_in_MB>     
`预先计算的以MB为单位的可用内存。内存越大则训练的速度越快。`  
－sym（default）   
－nonsym   
`指定训练的目标对象是否垂直对称。垂直对称提高目标的训练速度。例如，正面部是垂直对称的。`   
－minhitrate《min_hit_rate》   
`每个阶段分类器需要的最小的命中率。总的命中率为min_hit_rate的number_of_stages次方。`    
－maxfalsealarm<max_false_alarm_rate>   
`没有阶段分类器的最大错误报警率。总的错误警告率为max_false_alarm_rate的number_of_stages次方。`    
－weighttrimming<weight_trimming>    
`指定是否使用权修正和使用多大的权修正。一个基本的选择是0.9`    
－eqw   
－mode<basic(default)|core|all>   
`选择用来训练的haar特征集的种类。basic仅仅使用垂直特征。all使用垂直和45度角旋转特征。`   
－w《sample_width》   
－h《sample_height》   
`训练样本的尺寸，（以像素为单位）。必须和训练样本创建的尺寸相同。`   
一个训练分类器的例子：   
"D:\Program Files\OpenCV\bin\haartraining.exe"   -data data\cascade -vec data\pos.vec -bg negdata\negdata.dat -npos 49 -nneg 49 -mem 200 -mode ALL -w 20 -h 20   
  
训练结束后，会在目录data下生成一些子目录，即为训练好的分类器。   
  
（7） 生成xml文件   
  
`上一步在进行haartraining的时候，会在data目录下生成一些目录及txt文件，我们需要调用opencv\bin\haarconv.exe将这些txt文件转换为xml文件，也就是所谓的分类器。`   


 

