# NNDL-final-task1
对比监督学习和自监督学习在图像分类任务上的性能表现

------

### Task1 

(1) 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；

(2) 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；

(3) 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

------

### Set UP

训练基于如下应用程序版本：

<table style="width:50%">
  <tr>
    <th>Tool</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>11.6</td>
  </tr>
  <tr>
    <td>torch</td>
    <td>1.12.0</td>
  </tr>
  <tr>
    <td>torchvision</td>
    <td>0.13.0</td>
  </tr>
  <tr>
    <td>Tensorboard</td>
    <td>2.11.2</td>
  </tr>
</table>
可以使用以下命令安装所需的Python包：

```bash
pip install torch torchvision tensorboard
```

------

### 训练和测试

### Train&Test

#### 自监督学习训练

使用以下命令在CIFAR-10数据集上训练ResNet-18，其中CIFAR-10的下载由程序自动完成：

```bash
python SimCLR.py
```

有关SimCLR算法的详细内容，可参见 [SimCLR算法](https://github.com/google-research/simclr)。

#### 性能评测

在训练完成后，使用以下命令对其性能进行评测：

```bash
python LinearClassification.py
```

#### 监督学习训练

使用以下命令在ImageNet数据集上采用监督学习训练并得到表征：

```bash
python ImageNet.py
```

然后，使用以下命令在CIFAR-100数据集上从零开始以监督学习方式进行训练，并进行对比：

```bash
python Supervised.py
```

------

### Visualization

我们使用Tensorboard进行训练过程和结果的可视化。启动Tensorboard服务器可运行以下命令：

```bash
tensorboard --logdir=/path/to/your/logdir/
```

其中 `/path/to/your/logdir/` 是日志的保存路径。在本实验中，日志保存于 `runs` 文件夹中，模型权重保存于 `models` 文件夹中。可以通过以下命令启动Tensorboard并查看训练日志：

```bash
tensorboard --logdir=runs/
```

### 目录结构

项目目录结构如下所示：

```
NNDL-final-task1/
├── SimCLR.py                  # 自监督学习训练脚本
├── LinearClassification.py    # 性能评测脚本
├── ImageNet.py                # 监督学习训练脚本
├── Supervised.py              # 从零开始监督学习训练脚本
├── runs/                      # Tensorboard日志目录
└── models/                    # 保存的模型权重
```
