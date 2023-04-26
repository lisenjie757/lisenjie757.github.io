# lesson6. 优化机制

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
```

## 前置代码
### 数据集&数据加载器&构建模型


```python
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
        
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to('cuda')
```

### 超参数


```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 优化循环
### 损失函数


```python
## 初始化损失函数
loss_fn = nn.CrossEntropyLoss().to('cuda')
```

### 优化器


```python
## 此处使用SGD，还有ADAM RMSProp
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
```

## 完整实现


```python
def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X = X.to('cuda')
        y = y.to('cuda')
        ## 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred,y)
        
        ## 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss,current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0,0
    
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to('cuda')
            y = y.to('cuda')
            
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Ang loss: {test_loss:>8f}\n")
    
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
print("Done")
```

    Epoch 1
    ----------------------------
    loss: 2.304493  [    0/60000]
    loss: 2.283905  [ 6400/60000]
    loss: 2.284196  [12800/60000]
    loss: 2.263063  [19200/60000]
    loss: 2.251981  [25600/60000]
    loss: 2.237573  [32000/60000]
    loss: 2.227256  [38400/60000]
    loss: 2.204539  [44800/60000]
    loss: 2.182078  [51200/60000]
    loss: 2.162598  [57600/60000]
    Test Error: 
     Accuracy: 49.6%, Ang loss: 2.161115
    
    Done

