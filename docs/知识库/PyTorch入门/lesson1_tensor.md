# lesson1. 张量操作

```python
import torch
import numpy as np
```

## 初始化张量

```python
##直接数据创建
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

##通过numpy转换
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

##保持特定属性创建
x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data,dtype=torch.float)
print(x_rand)

##以特定的维度构造特定属性的张量
shape = (2,3,)
rand_tensor = torch.rand(shape)
print(rand_tensor)
ones_tensor = torch.ones(shape)
print(ones_tensor)
zeros_tensor = torch.zeros(shape)
print(zeros_tensor)
```


        tensor([[1, 2],
                [3, 4]])
        tensor([[1, 2],
                [3, 4]])
        tensor([[1, 1],
                [1, 1]])
        tensor([[0.8245, 0.8069],
                [0.2119, 0.9145]])
        tensor([[0.3151, 0.8133, 0.7510],
                [0.8190, 0.9030, 0.1693]])
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
        tensor([[0., 0., 0.],
                [0., 0., 0.]])


## 张量的属性

```python
tensor = torch.rand(3,4)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
```

    tensor([[0.9527, 0.5502, 0.1051, 0.5587],
            [0.7882, 0.5084, 0.8422, 0.6339],
            [0.1735, 0.7458, 0.3827, 0.7142]])
    torch.Size([3, 4])
    torch.float32
    cpu


## 张量运算


```python
##张量在GPU运算
print(torch.__version__)
print(torch.cuda.is_available())
tensor = tensor.to('cuda')
print(tensor)

##标准索引和切片运算
tensor = torch.ones(4,4)
print(tensor[0])
print(tensor[:,0])
print(tensor[...,-1])
tensor[...,1] = 0
print(tensor)

##张量连接运算
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

##矩阵乘法（下面三种写法是一样的）
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor,tensor.T,out=y3)
print(y1)
print(y2)
print(y3)

##矩阵对应元素点乘（下面三种写法是一样的）
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(z1)
print(z2)
print(z3)

##单元素张量（可转换为Python元素）
agg = tensor.sum()
print(agg,type(agg))
agg_item = agg.item()
print(agg_item,type(agg_item))

##原地张量运算（计算微分时会丢失历史信息，不建议使用）
print(tensor)
tensor.add_(5)
print(tensor)
```

    1.12.0
    True
    tensor([[0.9527, 0.5502, 0.1051, 0.5587],
            [0.7882, 0.5084, 0.8422, 0.6339],
            [0.1735, 0.7458, 0.3827, 0.7142]], device='cuda:0')
    tensor([1., 1., 1., 1.])
    tensor([1., 1., 1., 1.])
    tensor([1., 1., 1., 1.])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
    tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])
    tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])
    tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    tensor(12.) <class 'torch.Tensor'>
    12.0 <class 'float'>
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]])


## 与Numpy互相转换


```python
## tensor to numpy
t = torch.ones(5)
print(t)
n = t.numpy()
print(n)

## numpy to tensor
n = np.ones(5)
print(n)
t = torch.from_numpy(n)
print(t)
```

    tensor([1., 1., 1., 1., 1.])
    [1. 1. 1. 1. 1.]
    [1. 1. 1. 1. 1.]
    tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

