### .ipynb->.py
```bash
jupyter nbconvert --to script your_notebook.ipynb
```
在相同目录下生成一个名为 your_notebook.py 的文件。

### tensorboard 常用配置
```bash
tensorboard --logdir=logs/ant_bee
```

### loss与model的关系
一个loss对象可以应用于多个模型。例如：
```python
tudui1 = Tudui() # 模型
tudui2 = Tudui()
optim1 = torch.optim.SGD(tudui1.parameters(), lr=0.01)
optim2 = torch.optim.SGD(tudui2.parameters(), lr=0.01)

for data in dataloader:
    imgs, targets = data

    # Forward pass for tudui1
    outputs1 = tudui1(imgs)
    loss1 = loss(outputs1, targets)
    
    # Forward pass for tudui2
    outputs2 = tudui2(imgs)
    loss2 = loss(outputs2, targets)

    # Backward pass for tudui1
    optim1.zero_grad()
    loss1.backward()
    optim1.step()

    # Backward pass for tudui2
    optim2.zero_grad()
    loss2.backward()
    optim2.step()
```
在PyTorch中，result_loss.backward()知道要将梯度信息传播回哪个模型，是因为张量（tensors）和模型参数之间的连接在前向传播时就已经建立。具体的机制和流程：

#### 梯度传播机制
1. 前向传播建立计算图：
    当执行outputs = tudui(imgs)时，PyTorch会动态构建一个计算图（computation graph），记录下每个操作以及涉及的张量和参数。
在计算损失result_loss = loss(outputs, targets)时，这个计算图也包含了outputs的所有前向操作（即模型Tudui的所有层和参数）。

2. 计算图中的节点：
    计算图中的每个节点代表一个张量操作。节点间的边表示这些操作之间的数据流。
tudui.parameters()返回的参数张量会作为计算图中的叶子节点（leaf nodes），这些叶子节点是需要计算梯度的。

3. 梯度传播：
    调用result_loss.backward()时，PyTorch会沿着计算图进行反向传播，计算每个叶子节点（即模型参数）的梯度，并将梯度存储在这些张量的.grad属性中。

    