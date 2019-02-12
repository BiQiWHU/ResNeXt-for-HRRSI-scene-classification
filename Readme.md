Using ResNeXt or ResNet for image scene classification
In this project, we offer the code of ResNet18\34\50\101\152 and ResNeXt50 for image scene classification.
Note that ResNeXt101 is similar to ResNeXt50 and all you need to do is just change several numbers in my code.

Settings：
（1）Python3.5
（2）Tensorflow1.6
（3）OpenCV3

The operation is listed as follows.
Step1: Paste your data in the folder. (Images of a same class should be placed in one subfolder)
Step2: run tfdata.py, generate the training and testing dataset.
Step3: run Res.py to train your model (you can select ResNet18\34\50\101\152) or run ResNeXtb.py/ResNeXtc.py to train your ResNeXt model
Step4: use test.py to obtain the performance on your own dataset.(Please change the loops according to your number of testing samples)(Our dataset has 420 test samples and our batch size is 20 when debuging, please change into your own settings)
Enjoy!

Note that in ResNeXtb.py we implement Type B of ResNeXt structure and in ResNeXtc.py we implement Type C of ResNeXt structure. 
Also note that there are three types of ResNeXt block and these three structures are equal.


During the coding, I also refer to the following project:
https://github.com/taki0112/ResNet-Tensorflow/blob/master/ops.py