import{_ as n,Y as e,Z as a,a0 as s}from"./framework-c2b1cf81.js";const t={},p=s(`<h1 id="lesson4-构建神经网络" tabindex="-1"><a class="header-anchor" href="#lesson4-构建神经网络" aria-hidden="true">#</a> lesson4. 构建神经网络</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> os 
<span class="token keyword">import</span> torch
<span class="token keyword">from</span> torch <span class="token keyword">import</span> nn
<span class="token keyword">from</span> torch<span class="token punctuation">.</span>utils<span class="token punctuation">.</span>data <span class="token keyword">import</span> DataLoader
<span class="token keyword">from</span> torchvision <span class="token keyword">import</span> datasets<span class="token punctuation">,</span> transforms
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>/home/lsj/.conda/envs/pt12/lib/python3.8/site-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at  /opt/conda/conda-bld/pytorch_1656352465323/work/c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() &gt; 0
</code></pre><h2 id="获取训练设备" tabindex="-1"><a class="header-anchor" href="#获取训练设备" aria-hidden="true">#</a> 获取训练设备</h2><ul><li>查看有无可用GPU</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>device <span class="token operator">=</span> <span class="token string">&quot;cuda&quot;</span> <span class="token keyword">if</span> torch<span class="token punctuation">.</span>cuda<span class="token punctuation">.</span>is_available<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">else</span> <span class="token string">&quot;cpu&quot;</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>device<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>cpu
</code></pre><h2 id="定义神经网络类" tabindex="-1"><a class="header-anchor" href="#定义神经网络类" aria-hidden="true">#</a> 定义神经网络类</h2><ul><li>super(NeuralNetwork,self) ：查找NeuralNetwork的父类，对self实施父类的方法</li><li>nn.Flatten(x,[start=1,end=-1]) ：对输入张量进行指定维数降维，此处将(1,28,28)降成(1,28*28)</li><li>nn.Sequential() ：序列容器，将神经网络模块按顺序添加到容器中</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">NeuralNetwork</span><span class="token punctuation">(</span>nn<span class="token punctuation">.</span>Module<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token builtin">super</span><span class="token punctuation">(</span>NeuralNetwork<span class="token punctuation">,</span>self<span class="token punctuation">)</span><span class="token punctuation">.</span>__init__<span class="token punctuation">(</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>flatten <span class="token operator">=</span> nn<span class="token punctuation">.</span>Flatten<span class="token punctuation">(</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>linear_relu_stack <span class="token operator">=</span> nn<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span>
            nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">28</span><span class="token operator">*</span><span class="token number">28</span><span class="token punctuation">,</span><span class="token number">512</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
            nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
            nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">512</span><span class="token punctuation">,</span><span class="token number">512</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
            nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
            nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">512</span><span class="token punctuation">,</span><span class="token number">10</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
        <span class="token punctuation">)</span>
        
    <span class="token keyword">def</span> <span class="token function">forward</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
        x <span class="token operator">=</span> self<span class="token punctuation">.</span>flatten<span class="token punctuation">(</span>x<span class="token punctuation">)</span>
        logits <span class="token operator">=</span> self<span class="token punctuation">.</span>linear_relu_stack<span class="token punctuation">(</span>x<span class="token punctuation">)</span>
        <span class="token keyword">return</span> logits
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">##将模型移入GPU并打印其网络结构</span>
model <span class="token operator">=</span> NeuralNetwork<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>to<span class="token punctuation">(</span>device<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>model<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
</code></pre><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">##输入数据到模型模块进行推理，不要直接调用model.forward()!!!</span>
X <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">,</span>device<span class="token operator">=</span>device<span class="token punctuation">)</span>
logits <span class="token operator">=</span> model<span class="token punctuation">(</span>X<span class="token punctuation">)</span>
pred_probab <span class="token operator">=</span> nn<span class="token punctuation">.</span>Softmax<span class="token punctuation">(</span>dim<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">(</span>logits<span class="token punctuation">)</span>
y_pred <span class="token operator">=</span> pred_probab<span class="token punctuation">.</span>argmax<span class="token punctuation">(</span>dim<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>logits<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>pred_probab<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>y_pred<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[ 0.1835,  0.0703,  0.0762, -0.0534, -0.0084,  0.0368,  0.1318, -0.0841,
         -0.0383, -0.0130]], grad_fn=&lt;AddmmBackward0&gt;)
tensor([[0.1162, 0.1037, 0.1044, 0.0917, 0.0959, 0.1003, 0.1103, 0.0889, 0.0931,
         0.0955]], grad_fn=&lt;SoftmaxBackward0&gt;)
tensor([0])
</code></pre><h2 id="模型层解构" tabindex="-1"><a class="header-anchor" href="#模型层解构" aria-hidden="true">#</a> 模型层解构</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>input_image <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>input_image<span class="token punctuation">.</span>size<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>torch.Size([3, 28, 28])
</code></pre><h2 id="nn-flatten" tabindex="-1"><a class="header-anchor" href="#nn-flatten" aria-hidden="true">#</a> nn.Flatten</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>flatten <span class="token operator">=</span> nn<span class="token punctuation">.</span>Flatten<span class="token punctuation">(</span><span class="token punctuation">)</span>
flat_image <span class="token operator">=</span> flatten<span class="token punctuation">(</span>input_image<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>flat_image<span class="token punctuation">.</span>size<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>torch.Size([3, 784])
</code></pre><h2 id="nn-linear" tabindex="-1"><a class="header-anchor" href="#nn-linear" aria-hidden="true">#</a> nn.Linear</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>layer1 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span>in_features<span class="token operator">=</span><span class="token number">28</span><span class="token operator">*</span><span class="token number">28</span><span class="token punctuation">,</span>out_features<span class="token operator">=</span><span class="token number">20</span><span class="token punctuation">)</span>
hidden1 <span class="token operator">=</span> layer1<span class="token punctuation">(</span>flat_image<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>hidden1<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[-0.3914,  0.9743,  0.3574,  0.0032, -0.0105, -0.3178, -0.6143,  0.0487,
          0.0233, -0.4386,  0.3226,  0.0912,  0.0098,  0.0723, -0.1843, -0.5586,
         -0.0618, -0.0330,  0.6477,  0.4035],
        [-0.5303,  0.4330,  0.5043,  0.3772, -0.3653, -0.2800, -0.3662,  0.0570,
          0.3869,  0.0945, -0.2175, -0.0924, -0.1414, -0.1828,  0.0621, -0.3528,
         -0.2910, -0.0231,  0.1191,  0.2671],
        [-0.1596,  0.5198,  0.3571,  0.0806, -0.2248, -0.2083, -0.3483,  0.0522,
         -0.0583, -0.0232,  0.0035, -0.3093,  0.0038,  0.0386,  0.2241, -0.2543,
         -0.2830,  0.0570,  0.2809,  0.0586]], grad_fn=&lt;AddmmBackward0&gt;)
</code></pre><h2 id="nn-relu" tabindex="-1"><a class="header-anchor" href="#nn-relu" aria-hidden="true">#</a> nn.ReLu</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;Before ReLU: &quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>hidden1<span class="token punctuation">)</span><span class="token punctuation">)</span>
hidden1 <span class="token operator">=</span> nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">(</span>hidden1<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;After ReLU: &quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>hidden1<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>Before ReLU: tensor([[-0.3914,  0.9743,  0.3574,  0.0032, -0.0105, -0.3178, -0.6143,  0.0487,
          0.0233, -0.4386,  0.3226,  0.0912,  0.0098,  0.0723, -0.1843, -0.5586,
         -0.0618, -0.0330,  0.6477,  0.4035],
        [-0.5303,  0.4330,  0.5043,  0.3772, -0.3653, -0.2800, -0.3662,  0.0570,
          0.3869,  0.0945, -0.2175, -0.0924, -0.1414, -0.1828,  0.0621, -0.3528,
         -0.2910, -0.0231,  0.1191,  0.2671],
        [-0.1596,  0.5198,  0.3571,  0.0806, -0.2248, -0.2083, -0.3483,  0.0522,
         -0.0583, -0.0232,  0.0035, -0.3093,  0.0038,  0.0386,  0.2241, -0.2543,
         -0.2830,  0.0570,  0.2809,  0.0586]], grad_fn=&lt;AddmmBackward0&gt;)
After ReLU: tensor([[0.0000, 0.9743, 0.3574, 0.0032, 0.0000, 0.0000, 0.0000, 0.0487, 0.0233,
         0.0000, 0.3226, 0.0912, 0.0098, 0.0723, 0.0000, 0.0000, 0.0000, 0.0000,
         0.6477, 0.4035],
        [0.0000, 0.4330, 0.5043, 0.3772, 0.0000, 0.0000, 0.0000, 0.0570, 0.3869,
         0.0945, 0.0000, 0.0000, 0.0000, 0.0000, 0.0621, 0.0000, 0.0000, 0.0000,
         0.1191, 0.2671],
        [0.0000, 0.5198, 0.3571, 0.0806, 0.0000, 0.0000, 0.0000, 0.0522, 0.0000,
         0.0000, 0.0035, 0.0000, 0.0038, 0.0386, 0.2241, 0.0000, 0.0000, 0.0570,
         0.2809, 0.0586]], grad_fn=&lt;ReluBackward0&gt;)
</code></pre><h2 id="nn-sequential" tabindex="-1"><a class="header-anchor" href="#nn-sequential" aria-hidden="true">#</a> nn.Sequential</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>seq_modules <span class="token operator">=</span> nn<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span>
    flatten<span class="token punctuation">,</span>
    layer1<span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">20</span><span class="token punctuation">,</span><span class="token number">10</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
input_image <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">,</span><span class="token number">28</span><span class="token punctuation">)</span>
logits <span class="token operator">=</span> seq_modules<span class="token punctuation">(</span>input_image<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>logits<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[ 0.1165, -0.2980, -0.1271,  0.1471,  0.1197, -0.0370, -0.1217, -0.0424,
          0.1851,  0.1187],
        [ 0.2034, -0.2883, -0.2599,  0.1343,  0.0700, -0.1013, -0.1442, -0.0667,
          0.2362,  0.1714],
        [ 0.1659, -0.2946, -0.1774,  0.1805,  0.1837, -0.1381, -0.2138, -0.0489,
          0.1290,  0.1409]], grad_fn=&lt;AddmmBackward0&gt;)
</code></pre><h2 id="nn-softmax" tabindex="-1"><a class="header-anchor" href="#nn-softmax" aria-hidden="true">#</a> nn.Softmax</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>softmax <span class="token operator">=</span> nn<span class="token punctuation">.</span>Softmax<span class="token punctuation">(</span>dim<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>
pred_probab <span class="token operator">=</span> softmax<span class="token punctuation">(</span>logits<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>pred_probab<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[0.1105, 0.0730, 0.0866, 0.1139, 0.1108, 0.0948, 0.0871, 0.0943, 0.1183,
         0.1107],
        [0.1211, 0.0741, 0.0762, 0.1130, 0.1060, 0.0893, 0.0855, 0.0924, 0.1251,
         0.1173],
        [0.1171, 0.0739, 0.0831, 0.1188, 0.1192, 0.0864, 0.0801, 0.0945, 0.1128,
         0.1142]], grad_fn=&lt;SoftmaxBackward0&gt;)
</code></pre><h2 id="模型参数" tabindex="-1"><a class="header-anchor" href="#模型参数" aria-hidden="true">#</a> 模型参数</h2><ul><li>nn.Moudle会自动跟踪保存模型参数，使用parameters()或named_parameters()获取</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">print</span><span class="token punctuation">(</span>model<span class="token punctuation">)</span>
<span class="token keyword">for</span> name<span class="token punctuation">,</span>param <span class="token keyword">in</span> model<span class="token punctuation">.</span>named_parameters<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>name<span class="token punctuation">)</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>param<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
linear_relu_stack.0.weight
Parameter containing:
tensor([[ 0.0348,  0.0040,  0.0076,  ..., -0.0357,  0.0206,  0.0259],
        [-0.0055,  0.0342, -0.0239,  ...,  0.0177, -0.0028,  0.0001],
        [-0.0087,  0.0276, -0.0036,  ...,  0.0066, -0.0192,  0.0107],
        ...,
        [ 0.0304,  0.0203, -0.0344,  ..., -0.0248, -0.0153, -0.0235],
        [-0.0091,  0.0115,  0.0318,  ...,  0.0144, -0.0122,  0.0103],
        [ 0.0242,  0.0351,  0.0152,  ...,  0.0321,  0.0250,  0.0157]],
       requires_grad=True)
linear_relu_stack.0.bias
Parameter containing:
tensor([ 0.0086,  0.0261, -0.0205, -0.0040, -0.0326,  0.0316,  0.0330,  0.0151,
         0.0222,  0.0155,  0.0138,  0.0226, -0.0340,  0.0284,  0.0142,  0.0227,
        -0.0184,  0.0282,  0.0177, -0.0084, -0.0119,  0.0096,  0.0266, -0.0295,
         0.0015, -0.0128,  0.0264,  0.0321,  0.0178, -0.0252,  0.0236,  0.0214,
        -0.0174,  0.0289,  0.0141, -0.0312,  0.0098,  0.0023,  0.0334,  0.0266,
        -0.0066, -0.0073, -0.0353, -0.0099,  0.0220,  0.0309, -0.0066,  0.0306,
        -0.0093, -0.0084, -0.0095, -0.0249,  0.0078, -0.0064,  0.0199,  0.0271,
        -0.0092,  0.0084, -0.0321,  0.0004,  0.0080, -0.0073,  0.0245, -0.0002,
        -0.0249, -0.0292, -0.0079,  0.0330, -0.0068,  0.0158,  0.0122, -0.0097,
         0.0253,  0.0310, -0.0032,  0.0106,  0.0018, -0.0012,  0.0039, -0.0119,
        -0.0156, -0.0053,  0.0002,  0.0014,  0.0014,  0.0159, -0.0092,  0.0293,
         0.0052, -0.0214,  0.0215,  0.0116,  0.0255,  0.0205, -0.0012, -0.0256,
         0.0120, -0.0325,  0.0134, -0.0281, -0.0290,  0.0121, -0.0133, -0.0083,
         0.0181,  0.0198,  0.0058,  0.0216, -0.0294, -0.0038,  0.0191,  0.0347,
        -0.0143,  0.0352,  0.0141, -0.0328, -0.0030,  0.0131,  0.0042,  0.0300,
         0.0344,  0.0171,  0.0259,  0.0226, -0.0004,  0.0050, -0.0064, -0.0164,
         0.0203, -0.0159, -0.0110,  0.0344,  0.0011, -0.0209,  0.0086,  0.0233,
        -0.0299,  0.0152, -0.0084, -0.0026, -0.0344, -0.0111,  0.0086,  0.0349,
         0.0332, -0.0176,  0.0083, -0.0140,  0.0187,  0.0048,  0.0086,  0.0030,
         0.0277, -0.0129,  0.0217,  0.0349,  0.0280,  0.0318, -0.0106, -0.0288,
         0.0162, -0.0272, -0.0072, -0.0309, -0.0355, -0.0083, -0.0280,  0.0040,
        -0.0351,  0.0036,  0.0146,  0.0040, -0.0322,  0.0223, -0.0252, -0.0342,
        -0.0173,  0.0159, -0.0172,  0.0283,  0.0034, -0.0343, -0.0092,  0.0318,
        -0.0109, -0.0017,  0.0117, -0.0121,  0.0198,  0.0045,  0.0195,  0.0203,
         0.0027,  0.0307,  0.0342,  0.0058,  0.0137, -0.0250,  0.0323, -0.0203,
        -0.0175, -0.0336,  0.0225,  0.0174,  0.0133, -0.0023,  0.0227,  0.0159,
         0.0103, -0.0025, -0.0069,  0.0192, -0.0229, -0.0128,  0.0086, -0.0022,
         0.0079,  0.0272,  0.0226,  0.0276,  0.0306, -0.0331, -0.0246, -0.0043,
        -0.0060, -0.0044, -0.0346,  0.0211, -0.0071, -0.0253,  0.0276, -0.0204,
        -0.0324,  0.0193, -0.0099, -0.0121,  0.0290, -0.0217,  0.0343, -0.0314,
        -0.0292, -0.0337,  0.0151,  0.0295,  0.0176, -0.0045,  0.0142,  0.0304,
         0.0330, -0.0073,  0.0148,  0.0149,  0.0192, -0.0222,  0.0071,  0.0125,
         0.0131,  0.0269,  0.0344, -0.0205,  0.0218, -0.0177, -0.0269,  0.0308,
        -0.0307, -0.0215,  0.0141, -0.0032, -0.0103, -0.0041, -0.0056,  0.0134,
         0.0209, -0.0150,  0.0095,  0.0140, -0.0091, -0.0120,  0.0293,  0.0271,
        -0.0248,  0.0034, -0.0329, -0.0193,  0.0074,  0.0277,  0.0201,  0.0309,
         0.0164, -0.0086,  0.0351,  0.0066,  0.0134, -0.0169,  0.0097, -0.0147,
        -0.0202,  0.0163, -0.0352, -0.0045,  0.0349,  0.0263, -0.0148, -0.0227,
        -0.0271,  0.0343,  0.0116, -0.0238, -0.0317,  0.0028, -0.0039,  0.0135,
        -0.0292, -0.0170,  0.0183,  0.0149, -0.0118, -0.0347,  0.0133,  0.0243,
        -0.0031, -0.0055, -0.0007,  0.0086,  0.0182,  0.0312,  0.0135,  0.0247,
        -0.0009, -0.0114,  0.0334,  0.0033,  0.0345,  0.0009,  0.0325,  0.0345,
         0.0130, -0.0173, -0.0304,  0.0315, -0.0152,  0.0342,  0.0344,  0.0159,
        -0.0345, -0.0127, -0.0041,  0.0154,  0.0021, -0.0109, -0.0194, -0.0281,
        -0.0313,  0.0304,  0.0296, -0.0010,  0.0145, -0.0013,  0.0225, -0.0129,
        -0.0117,  0.0243,  0.0114,  0.0268,  0.0355,  0.0287,  0.0215, -0.0161,
        -0.0352, -0.0282, -0.0211, -0.0301, -0.0174,  0.0089, -0.0218, -0.0023,
        -0.0317,  0.0042,  0.0058, -0.0156, -0.0101,  0.0149,  0.0078,  0.0137,
        -0.0260, -0.0297,  0.0091,  0.0093, -0.0114,  0.0023, -0.0234, -0.0002,
        -0.0168,  0.0292, -0.0079, -0.0051,  0.0270, -0.0315, -0.0071,  0.0253,
         0.0168,  0.0220,  0.0239, -0.0155,  0.0092,  0.0175, -0.0040, -0.0141,
        -0.0194,  0.0099,  0.0291, -0.0104, -0.0010, -0.0028,  0.0270, -0.0121,
        -0.0240, -0.0177,  0.0315, -0.0061,  0.0183,  0.0273, -0.0118, -0.0030,
         0.0263, -0.0175, -0.0066, -0.0259, -0.0101, -0.0285,  0.0177, -0.0302,
         0.0235, -0.0129, -0.0354, -0.0338, -0.0323,  0.0244,  0.0228, -0.0277,
        -0.0251, -0.0111,  0.0082, -0.0015,  0.0052,  0.0273, -0.0055, -0.0343,
        -0.0202, -0.0139,  0.0105,  0.0304, -0.0068,  0.0223, -0.0314, -0.0344,
         0.0260, -0.0021,  0.0234, -0.0201, -0.0235, -0.0280,  0.0195,  0.0007,
        -0.0124,  0.0133, -0.0023, -0.0111, -0.0275,  0.0120, -0.0128, -0.0184,
        -0.0173, -0.0179, -0.0357, -0.0295,  0.0036, -0.0305,  0.0249,  0.0217,
         0.0213,  0.0177, -0.0226,  0.0214, -0.0117,  0.0243,  0.0051, -0.0346,
        -0.0152, -0.0278,  0.0193, -0.0311, -0.0318,  0.0307,  0.0086,  0.0304,
        -0.0007,  0.0357,  0.0219, -0.0269, -0.0110, -0.0060, -0.0256,  0.0340,
         0.0111, -0.0119, -0.0019,  0.0254,  0.0167, -0.0046,  0.0224,  0.0296],
       requires_grad=True)
linear_relu_stack.2.weight
Parameter containing:
tensor([[ 0.0345,  0.0181,  0.0369,  ..., -0.0131,  0.0136,  0.0314],
        [ 0.0394,  0.0078, -0.0396,  ...,  0.0083, -0.0370,  0.0369],
        [ 0.0008,  0.0354, -0.0103,  ...,  0.0071,  0.0435,  0.0437],
        ...,
        [-0.0400, -0.0052, -0.0206,  ..., -0.0104, -0.0068, -0.0242],
        [ 0.0065, -0.0042,  0.0153,  ...,  0.0032, -0.0207, -0.0188],
        [-0.0385,  0.0161, -0.0351,  ..., -0.0256,  0.0053, -0.0024]],
       requires_grad=True)
linear_relu_stack.2.bias
Parameter containing:
tensor([ 4.1813e-02, -1.6619e-02,  3.9113e-02, -1.5093e-02,  1.5017e-02,
         5.2896e-03, -7.0315e-03,  9.5963e-03, -3.3275e-02, -1.9160e-02,
        -3.4745e-02,  1.0509e-02,  1.0498e-04, -7.1548e-03, -1.5839e-02,
        -3.1533e-02, -1.3287e-02,  2.3534e-02, -1.6398e-02,  4.8202e-03,
        -1.0436e-02,  3.4014e-02,  2.8655e-02,  2.8397e-02,  2.7178e-02,
         3.9391e-02,  3.0124e-02,  3.3509e-03,  6.1992e-03,  2.6582e-02,
        -3.9359e-02, -3.0841e-02,  2.8772e-02,  1.2272e-02,  3.5646e-02,
         2.2063e-02, -1.4506e-02,  8.3983e-03,  3.6239e-02, -3.6312e-02,
         4.0445e-02,  3.2031e-02,  5.8938e-03,  1.1676e-02,  1.2338e-02,
         4.0429e-02,  3.2177e-02, -1.9051e-02, -3.5229e-02, -1.8315e-02,
        -3.6294e-02,  2.0468e-02, -4.3678e-02,  1.2375e-02, -2.1652e-02,
        -2.6055e-03,  1.9871e-02, -2.9539e-02, -4.4110e-03, -4.0434e-02,
         3.8815e-02,  2.5248e-02,  3.2590e-02,  4.4631e-04, -3.0369e-02,
         1.7029e-02, -5.3398e-03,  1.9067e-02, -4.1852e-02,  9.3174e-03,
         3.2190e-02,  6.5696e-03,  3.1825e-03, -6.6836e-04,  1.0921e-02,
         3.2889e-02,  1.3549e-02,  1.1673e-03,  4.2575e-02,  2.5849e-02,
        -3.9895e-02, -3.9511e-02, -2.1672e-02,  3.7632e-02,  1.7327e-02,
         2.3956e-02, -2.5438e-02,  3.1431e-02, -2.6375e-03,  2.0853e-02,
        -2.4985e-02, -1.1729e-02, -1.8595e-02,  1.9006e-02,  1.5368e-03,
        -3.5385e-02, -3.6201e-02,  9.3275e-03,  1.2355e-02,  3.5495e-02,
        -3.3091e-02,  6.2980e-03,  4.5804e-03, -4.3357e-02, -9.3052e-03,
        -1.4889e-02,  3.6015e-02, -6.1881e-03,  5.8576e-03,  3.3089e-02,
         1.5759e-02, -1.9032e-02, -1.1096e-02, -6.8615e-03,  1.7547e-03,
         2.7654e-02,  7.6464e-03, -4.0611e-02, -1.9589e-02,  1.9037e-02,
        -7.7626e-03, -3.3604e-02, -2.5285e-02,  4.0471e-02,  3.6573e-02,
        -3.9312e-02, -3.3128e-02,  1.4771e-02,  1.0701e-02, -3.1122e-02,
         3.6833e-02, -3.6208e-02,  7.8927e-03, -2.9675e-02,  3.9354e-02,
        -3.0588e-02, -3.5297e-02,  3.1088e-02,  1.7613e-02,  3.1319e-02,
         2.7442e-02,  3.8756e-02,  4.4021e-02,  4.3242e-02, -5.1761e-03,
         3.4909e-02,  3.3177e-02, -2.4528e-02,  3.8147e-02, -1.9509e-02,
        -2.1462e-02,  1.5008e-02, -3.2534e-02, -3.9613e-02, -3.7725e-02,
         3.1532e-02,  1.9861e-02,  3.8157e-02,  1.7813e-02,  6.0684e-03,
         3.6414e-03,  1.7636e-02,  5.8332e-03,  3.4099e-02, -3.4436e-02,
         2.4158e-02, -2.9897e-02,  3.6654e-02,  7.4221e-03,  1.5306e-02,
        -8.5132e-03,  8.1645e-03, -2.7132e-02, -1.4036e-02,  2.8793e-02,
         4.2096e-02, -1.4138e-02,  3.3185e-02, -3.6140e-02,  2.7398e-02,
        -1.5582e-02,  3.5993e-02,  3.0235e-02, -1.9122e-02, -3.2258e-02,
         5.2566e-03, -1.7969e-02,  3.2155e-02, -4.3666e-02,  2.1930e-02,
         1.4098e-02,  4.9657e-03, -3.7629e-02,  4.2928e-02,  3.9507e-02,
        -1.5557e-02, -3.8715e-02, -7.4666e-04,  6.8257e-03,  3.9410e-02,
         2.7932e-02, -7.3785e-03,  3.5149e-02, -2.1111e-03,  4.2002e-02,
        -6.6258e-03, -1.2529e-02,  2.5985e-02, -5.3836e-03,  3.4099e-02,
        -1.9472e-02,  1.4900e-02,  5.3838e-03,  1.7148e-02,  3.6593e-02,
        -1.3598e-02,  2.1629e-02,  2.9592e-02, -1.4871e-02,  1.7056e-02,
         2.5576e-02,  2.2679e-02,  9.2657e-03,  2.7061e-02, -1.4918e-04,
        -7.0879e-04,  2.1378e-02,  3.2623e-02, -3.2693e-02, -6.9890e-03,
         1.2475e-02,  2.1180e-02, -2.5963e-02,  1.1538e-02, -3.1687e-02,
        -3.3825e-02,  6.3065e-03, -2.2391e-02, -1.6993e-02,  2.9761e-02,
        -1.7584e-02,  2.6158e-02, -3.8398e-02,  9.5393e-03,  1.0308e-02,
        -2.7005e-02,  2.7423e-02, -3.0228e-02, -7.5275e-04,  2.9244e-03,
        -3.2164e-02,  3.6587e-02,  3.0417e-02, -2.9701e-02,  3.8880e-02,
        -3.0179e-02,  8.1338e-04,  1.9973e-02, -2.1000e-02, -1.2114e-02,
         2.6584e-02,  7.1286e-03,  8.1980e-03,  3.8927e-02, -1.0494e-02,
        -3.5321e-02,  3.1413e-03, -7.3663e-03, -1.4615e-02, -2.9388e-02,
        -1.0254e-02,  3.6683e-02,  1.9666e-02,  1.0081e-02, -3.3764e-02,
        -1.3077e-02, -1.1296e-02,  1.9023e-02,  5.0457e-03,  3.8632e-02,
        -3.8144e-02,  4.1357e-02,  1.7847e-02,  3.6878e-02,  3.9748e-02,
        -6.0793e-03,  2.1098e-02, -3.4776e-03,  3.3519e-02,  1.9769e-02,
        -3.3734e-03, -2.9008e-02, -4.2866e-02, -2.6344e-02,  1.6083e-02,
        -4.1019e-02, -3.9586e-02,  3.0554e-02,  1.0268e-02,  2.8977e-02,
         3.8883e-02, -1.5359e-02, -2.8558e-02,  4.0887e-03,  4.0116e-02,
         3.4093e-02,  3.2030e-03, -2.7915e-02,  1.0666e-02,  3.0899e-02,
        -2.1109e-02, -6.3490e-03, -4.2149e-02, -3.4760e-02, -2.5595e-02,
        -3.3401e-02, -5.2975e-03, -3.4152e-02,  3.5972e-02, -4.2104e-02,
        -2.7873e-02, -1.3702e-02, -5.8416e-03,  2.8854e-02,  2.5872e-02,
        -2.6067e-02, -2.7059e-02,  7.1851e-03,  2.8099e-02, -1.0737e-02,
         1.9886e-02,  2.8195e-02,  3.1350e-02, -2.6669e-02,  2.2479e-02,
        -4.3147e-03, -3.5953e-02, -1.9973e-04, -2.4630e-03,  2.7178e-02,
        -2.7818e-02,  2.6631e-02,  8.9729e-03, -2.2624e-02, -2.4536e-02,
         3.8296e-02,  2.4300e-02,  3.1020e-02,  2.5661e-02,  3.2956e-02,
         2.6426e-02, -3.3200e-02, -2.5431e-02,  3.7043e-02, -2.5536e-02,
        -2.3622e-02,  2.5614e-02,  2.9049e-02,  5.2677e-03,  3.0301e-02,
        -1.4990e-02,  1.3833e-02, -2.7951e-02,  1.0994e-02, -1.7039e-02,
        -3.8425e-02, -3.3476e-02, -3.6594e-02,  3.7877e-02, -2.3660e-02,
        -2.7774e-02, -2.5421e-02,  3.1451e-02,  1.8529e-02, -1.8345e-02,
         3.9190e-02, -3.0978e-02,  5.2639e-03,  2.9981e-02,  9.4218e-03,
        -4.0102e-02, -1.6685e-02, -2.9653e-02,  9.4134e-04, -3.6672e-02,
         2.8482e-02, -1.8619e-02,  2.3092e-02,  1.6687e-02,  3.6474e-03,
         2.7813e-02, -1.0635e-02,  6.3452e-03,  2.8750e-02, -2.9257e-02,
         4.9299e-03, -9.1479e-03,  3.1707e-02, -6.7157e-03,  2.3438e-04,
         1.7786e-02,  1.8846e-02,  7.4835e-03,  2.5446e-02,  1.7958e-02,
        -2.3121e-02,  3.3062e-02,  3.7613e-02, -4.3311e-02, -1.0183e-02,
         2.8330e-02, -4.0246e-02, -3.4520e-02,  3.7923e-02, -1.8100e-02,
         1.2737e-02, -1.2114e-02,  6.5649e-03, -5.9071e-03,  7.7365e-03,
        -7.4515e-03, -2.2001e-02, -2.3253e-02, -2.9996e-02,  1.1063e-03,
         1.4057e-03,  1.3215e-03, -3.7861e-02, -1.5023e-02, -7.1092e-03,
         1.2387e-02, -4.4001e-02,  3.0360e-02, -2.8778e-02,  3.1841e-02,
        -1.2114e-02,  4.1058e-02, -1.2814e-02,  3.9077e-02,  6.9941e-03,
         1.7139e-02,  3.9288e-02,  2.1338e-02,  2.7192e-02,  4.3974e-02,
         1.9053e-02,  9.1687e-04, -1.3872e-02,  2.6999e-02,  2.7631e-02,
        -2.2441e-02, -2.1365e-02, -3.3507e-02,  5.4755e-03,  2.4524e-02,
        -9.1414e-03,  3.3348e-02, -3.6404e-02, -1.5845e-02, -1.7853e-02,
        -2.6763e-02,  2.4977e-02, -2.2549e-02, -2.6055e-03, -1.4956e-02,
        -2.4629e-02, -2.9500e-02, -3.6520e-02, -3.1318e-02,  2.5301e-02,
        -4.1560e-02,  4.0596e-02, -3.5743e-02, -9.9402e-03, -3.8203e-02,
         1.8469e-02, -7.2222e-03,  8.7008e-03, -1.6842e-02,  1.3508e-03,
         2.8061e-02, -3.6841e-02, -4.1263e-02,  1.8805e-02, -4.1805e-02,
        -3.8119e-04, -1.6620e-02, -2.8509e-02, -4.1276e-02, -4.1390e-02,
        -1.6600e-02,  1.1227e-02, -9.8479e-03,  8.0011e-03, -3.1407e-02,
        -3.2109e-02,  3.0424e-02, -3.1924e-02, -3.0520e-05, -1.3426e-02,
        -3.4665e-02, -3.7141e-02, -8.8735e-03,  2.1064e-02,  1.1333e-02,
         3.4191e-02, -8.8482e-03,  1.2196e-02, -2.9521e-02, -7.7659e-03,
         2.9205e-02, -4.3007e-02], requires_grad=True)
linear_relu_stack.4.weight
Parameter containing:
tensor([[-0.0176, -0.0254,  0.0367,  ..., -0.0318, -0.0283,  0.0168],
        [-0.0305,  0.0373,  0.0211,  ..., -0.0173,  0.0382,  0.0340],
        [-0.0214, -0.0138,  0.0270,  ...,  0.0156, -0.0321, -0.0142],
        ...,
        [-0.0366,  0.0046, -0.0345,  ..., -0.0114, -0.0277, -0.0087],
        [-0.0342, -0.0080, -0.0343,  ...,  0.0110, -0.0043,  0.0092],
        [-0.0192,  0.0405, -0.0111,  ..., -0.0192, -0.0165,  0.0208]],
       requires_grad=True)
linear_relu_stack.4.bias
Parameter containing:
tensor([ 0.0355,  0.0054,  0.0438, -0.0149, -0.0191,  0.0248,  0.0221, -0.0388,
         0.0360,  0.0128], requires_grad=True)
</code></pre>`,36),o=[p];function c(i,l){return e(),a("div",null,o)}const r=n(t,[["render",c],["__file","lesson4_build.html.vue"]]);export{r as default};