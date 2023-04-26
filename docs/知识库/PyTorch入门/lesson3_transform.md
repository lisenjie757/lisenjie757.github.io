# lesson3. 数据预处理

## 图像和标签的转换函数

- ToTensor() ：转换PIL图像和NumPy数组为FloatTensor，并归一化图像像素值为[0.,1.]
- Lamda Transforms ：此函数为用户自定义lamda函数；此处函数为将整数转换为one-hot编码，首先创建zero张量，长度取决于类别数，然后调用scatter_函数通过整数y对张量进行1填充
- scatter_函数填充原理(dim=0) ：self[ index[i][j] ] [j] = src[i][j]


```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)
```
