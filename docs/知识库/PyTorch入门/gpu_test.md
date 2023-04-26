# lesson0. GPU性能测试

```python
import torch
import time

print(torch.__version__)        # 返回pytorch的版本
print(torch.cuda.is_available())        # 当CUDA可用时返回True

a = torch.randn(10000, 1000)    # 返回10000行1000列的张量矩阵
b = torch.randn(1000, 2000)     # 返回1000行2000列的张量矩阵

t0 = time.time()        # 记录时间
c = torch.matmul(a, b)      # 矩阵乘法运算
t1 = time.time()        # 记录时间
print(a.device, t1 - t0, c.norm(2))     # c.norm(2)表示矩阵c的二范数

device = torch.device('cuda')       # 用GPU来运行
a = a.to(device)
b = b.to(device)

# 初次调用GPU，需要数据传送，因此比较慢
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

# 这才是GPU处理数据的真实运行时间，当数据量越大，GPU的优势越明显
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
```


    1.12.0
    True
    cuda:0 0.00018405914306640625 tensor(1414297.6250, device='cuda:0')
    cuda:0 0.00024271011352539062 tensor(1414297.6250, device='cuda:0')

