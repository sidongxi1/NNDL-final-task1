# NNDL-final-task1
对比监督学习和自监督学习在图像分类任务上的性能表现

(1) 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；

**使用  SimCLR.py  训练ResNet-18，然后使用 LinearClassification.py 对其性能进行评测**

(2) 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；

**使用  ImageNet.py  训练并对比，**

(3) 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

