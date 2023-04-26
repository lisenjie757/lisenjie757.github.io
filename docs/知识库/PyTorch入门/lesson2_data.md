# lesson2. 数据集和数据加载

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

    /home/lsj/.conda/envs/pt12/lib/python3.8/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## 加载预加载数据集

- root ：训练集/测试集存储路径
- train ：确定是训练集还是测试集
- download=True ：如果root不可用 从互联网上下载数据
- transform / target_transform ：指定特征和标签的转换


```python
training_data = datasets.FashionMNIST(
    root = "../data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "../data",
    train = False,
    download = True,
    transform = ToTensor()
)
```

## 迭代和可视化数据集


```python
labels_map = {
    0:"T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    img,label =  training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()
```


![result](https://i.imgur.com/Ui7pdoU.png)


## 创建自定义数据集

- \_\_init__ ：创建类对象时运行一次 包括图像 标注 和 变换
- \_\_len__ ：返回数据集样本数
- \_\_getitem__ ：根据索引加载和返回数据样本 包括图像 标注 和 变换


```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label
```

## 使用DataLoaders训练Data
- DataLoader的作用：minibatches采样，每epoch reshuffle data，多线程加速


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)
```

## 通过DataLoader迭代


```python
train_features,train_labels = next(iter(train_dataloader))
print(train_features.size())
print(train_labels.size())
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img,cmap="gray")
plt.show()
print(label)
```

    torch.Size([64, 1, 28, 28])
    torch.Size([64])



![result](https://i.imgur.com/UEoa2W0.png)


    tensor(1)

