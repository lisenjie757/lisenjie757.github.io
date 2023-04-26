# lesson5. 自动求导机制

```python
import torch
```

## 定义最简单的单层神经网络


```python
## input tensor
x = torch.ones(5)
## expected output
y = torch.zeros(3)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
z = torch.matmul(x,w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
```

## 反向传播函数的引用存储在张量的grad_fn属性中


```python
print("z = "+str(z.grad_fn))
print("loss = "+str(loss.grad_fn))
```

    z = <AddBackward0 object at 0x7f05c46fa040>
    loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f05c4845f70>


## 计算梯度


```python
loss.backward()
print(w.grad)
print(b.grad)
```

    tensor([[0.2915, 0.1360, 0.1072],
            [0.2915, 0.1360, 0.1072],
            [0.2915, 0.1360, 0.1072],
            [0.2915, 0.1360, 0.1072],
            [0.2915, 0.1360, 0.1072]])
    tensor([0.2915, 0.1360, 0.1072])


## 禁用梯度跟踪
- 有时已有训练好的模型，我们只想推理，不想更新参数，可以使用torch.no_grad()或detach()来禁用梯度计算
- 以下是可能要禁用梯度跟踪的几点理由：1.做微调时可能需要冻结某些参数 2.只做推理时提高计算效率


```python
z = torch.matmul(x,w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w) + b
print(z.requires_grad)

z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)
```

    True
    False
    False


## 张量梯度和Jacobian Products
1. PyTorch backward grad实际计算保存的是Jacobian Products=vT*J，默认v=tensor(1.0)
2. PyTorch的机制会累积梯度，所以需要清零，实际训练时优化器会自动帮助我们清零


```python
inp = torch.eye(5,requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp),retain_graph=True)
print("First Call\n"+str(inp.grad))
out.backward(torch.ones_like(inp),retain_graph=True)
print("\nSecond Call\n"+str(inp.grad))
inp.grad.zero_()
out.backward(torch.ones_like(inp),retain_graph=True)
print("\nCall after zeroing gradients\n"+str(inp.grad))
```

    First Call
    tensor([[4., 2., 2., 2., 2.],
            [2., 4., 2., 2., 2.],
            [2., 2., 4., 2., 2.],
            [2., 2., 2., 4., 2.],
            [2., 2., 2., 2., 4.]])
    
    Second Call
    tensor([[8., 4., 4., 4., 4.],
            [4., 8., 4., 4., 4.],
            [4., 4., 8., 4., 4.],
            [4., 4., 4., 8., 4.],
            [4., 4., 4., 4., 8.]])
    
    Call after zeroing gradients
    tensor([[4., 2., 2., 2., 2.],
            [2., 4., 2., 2., 2.],
            [2., 2., 4., 2., 2.],
            [2., 2., 2., 4., 2.],
            [2., 2., 2., 2., 4.]])

