import{_ as n,Y as s,Z as a,a0 as e}from"./framework-c2b1cf81.js";const t={},p=e(`<h1 id="lesson7-保存和加载模型" tabindex="-1"><a class="header-anchor" href="#lesson7-保存和加载模型" aria-hidden="true">#</a> lesson7. 保存和加载模型</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> torch
<span class="token keyword">import</span> torchvision<span class="token punctuation">.</span>models <span class="token keyword">as</span> models
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="保存和加载模型权重" tabindex="-1"><a class="header-anchor" href="#保存和加载模型权重" aria-hidden="true">#</a> 保存和加载模型权重</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>model <span class="token operator">=</span> models<span class="token punctuation">.</span>vgg16<span class="token punctuation">(</span>pretrained<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
torch<span class="token punctuation">.</span>save<span class="token punctuation">(</span>model<span class="token punctuation">.</span>state_dict<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span><span class="token string">&#39;model_weights.pth&#39;</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>/home/lsj/.conda/envs/pt12/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter &#39;pretrained&#39; is deprecated since 0.13 and will be removed in 0.15, please use &#39;weights&#39; instead.
  warnings.warn(
/home/lsj/.conda/envs/pt12/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or \`None\` for &#39;weights&#39; are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing \`weights=VGG16_Weights.IMAGENET1K_V1\`. You can also use \`weights=VGG16_Weights.DEFAULT\` to get the most up-to-date weights.
  warnings.warn(msg)
</code></pre><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">## 创建相同的模型结构，但不加载预训练权重</span>
model <span class="token operator">=</span> models<span class="token punctuation">.</span>vgg16<span class="token punctuation">(</span><span class="token punctuation">)</span>
model<span class="token punctuation">.</span>load_state_dict<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>load<span class="token punctuation">(</span><span class="token string">&#39;model_weights.pth&#39;</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
<span class="token comment">## 加载权重做推理前需要调用此函数将dropout和batch normalization层设为评估模式，否则将会导致不一致的推理结果</span>
model<span class="token punctuation">.</span><span class="token builtin">eval</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
</code></pre><h2 id="保存和加载整个模型结构和权重" tabindex="-1"><a class="header-anchor" href="#保存和加载整个模型结构和权重" aria-hidden="true">#</a> 保存和加载整个模型结构和权重</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>torch<span class="token punctuation">.</span>save<span class="token punctuation">(</span>model<span class="token punctuation">,</span><span class="token string">&quot;model.pth&quot;</span><span class="token punctuation">)</span>
model <span class="token operator">=</span> torch<span class="token punctuation">.</span>load<span class="token punctuation">(</span><span class="token string">&quot;model.pth&quot;</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="保存和加载训练时的断点" tabindex="-1"><a class="header-anchor" href="#保存和加载训练时的断点" aria-hidden="true">#</a> 保存和加载训练时的断点</h2><h3 id="_1-导入必要的库" tabindex="-1"><a class="header-anchor" href="#_1-导入必要的库" aria-hidden="true">#</a> 1.导入必要的库</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> torch
<span class="token keyword">import</span> torch<span class="token punctuation">.</span>nn <span class="token keyword">as</span> nn
<span class="token keyword">import</span> torch<span class="token punctuation">.</span>nn<span class="token punctuation">.</span>functional <span class="token keyword">as</span> F
<span class="token keyword">import</span> torch<span class="token punctuation">.</span>optim <span class="token keyword">as</span> optim
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_2-定义和初始化神经网络" tabindex="-1"><a class="header-anchor" href="#_2-定义和初始化神经网络" aria-hidden="true">#</a> 2.定义和初始化神经网络</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Net</span><span class="token punctuation">(</span>nn<span class="token punctuation">.</span>Module<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token builtin">super</span><span class="token punctuation">(</span>Net<span class="token punctuation">,</span>self<span class="token punctuation">)</span><span class="token punctuation">.</span>__init__<span class="token punctuation">(</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>conv1 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Conv2d<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">6</span><span class="token punctuation">,</span><span class="token number">5</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>pool <span class="token operator">=</span> nn<span class="token punctuation">.</span>MaxPool2d<span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>conv2 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Conv2d<span class="token punctuation">(</span><span class="token number">6</span><span class="token punctuation">,</span><span class="token number">16</span><span class="token punctuation">,</span><span class="token number">5</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>fc1 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">16</span><span class="token operator">*</span><span class="token number">5</span><span class="token operator">*</span><span class="token number">5</span><span class="token punctuation">,</span><span class="token number">120</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>fc2 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">120</span><span class="token punctuation">,</span><span class="token number">84</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>fc3 <span class="token operator">=</span> nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">84</span><span class="token punctuation">,</span><span class="token number">10</span><span class="token punctuation">)</span>
    
    <span class="token keyword">def</span> <span class="token function">foward</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
        x <span class="token operator">=</span> self<span class="token punctuation">.</span>pool<span class="token punctuation">(</span>F<span class="token punctuation">.</span>relu<span class="token punctuation">(</span>self<span class="token punctuation">.</span>conv1<span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
        x <span class="token operator">=</span> self<span class="token punctuation">.</span>pool<span class="token punctuation">(</span>F<span class="token punctuation">.</span>relu<span class="token punctuation">(</span>self<span class="token punctuation">.</span>conv2<span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
        <span class="token comment">## view相当于reshape</span>
        x <span class="token operator">=</span> view<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">16</span><span class="token operator">*</span><span class="token number">5</span><span class="token operator">*</span><span class="token number">5</span><span class="token punctuation">)</span>
        X <span class="token operator">=</span> F<span class="token punctuation">.</span>relu<span class="token punctuation">(</span>self<span class="token punctuation">.</span>fc1<span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">)</span>
        x <span class="token operator">=</span> F<span class="token punctuation">.</span>relu<span class="token punctuation">(</span>self<span class="token punctuation">.</span>fc2<span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">)</span>
        x <span class="token operator">=</span> self<span class="token punctuation">.</span>fc3<span class="token punctuation">(</span>x<span class="token punctuation">)</span>
        <span class="token keyword">return</span> x

net <span class="token operator">=</span> Net<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>net<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</code></pre><h3 id="_3-初始化优化器" tabindex="-1"><a class="header-anchor" href="#_3-初始化优化器" aria-hidden="true">#</a> 3.初始化优化器</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>optimizer <span class="token operator">=</span> optim<span class="token punctuation">.</span>SGD<span class="token punctuation">(</span>net<span class="token punctuation">.</span>parameters<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>lr<span class="token operator">=</span><span class="token number">0.001</span><span class="token punctuation">,</span>momentum<span class="token operator">=</span><span class="token number">0.9</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_4-保存一般断点" tabindex="-1"><a class="header-anchor" href="#_4-保存一般断点" aria-hidden="true">#</a> 4.保存一般断点</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">## 附加信息</span>
EPOCH <span class="token operator">=</span> <span class="token number">5</span>
PATH <span class="token operator">=</span> <span class="token string">&quot;model.pt&quot;</span>
LOSS <span class="token operator">=</span> <span class="token number">0.4</span>

torch<span class="token punctuation">.</span>save<span class="token punctuation">(</span><span class="token punctuation">{</span>
    <span class="token string">&#39;epoch&#39;</span><span class="token punctuation">:</span>EPOCH<span class="token punctuation">,</span>
    <span class="token string">&#39;model_state_dict&#39;</span><span class="token punctuation">:</span>net<span class="token punctuation">.</span>state_dict<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    <span class="token string">&#39;optimizer_state_dict&#39;</span><span class="token punctuation">:</span>optimizer<span class="token punctuation">.</span>state_dict<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    <span class="token string">&#39;loss&#39;</span><span class="token punctuation">:</span>LOSS<span class="token punctuation">,</span>
    <span class="token punctuation">}</span><span class="token punctuation">,</span>PATH<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_5-加载一般断点" tabindex="-1"><a class="header-anchor" href="#_5-加载一般断点" aria-hidden="true">#</a> 5.加载一般断点</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">## 加载前首先初始化模型和优化器</span>
model <span class="token operator">=</span> Net<span class="token punctuation">(</span><span class="token punctuation">)</span>
optimizer <span class="token operator">=</span> optim<span class="token punctuation">.</span>SGD<span class="token punctuation">(</span>net<span class="token punctuation">.</span>parameters<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>lr<span class="token operator">=</span><span class="token number">0.001</span><span class="token punctuation">,</span>momentum<span class="token operator">=</span><span class="token number">0.9</span><span class="token punctuation">)</span>

checkpoint <span class="token operator">=</span> torch<span class="token punctuation">.</span>load<span class="token punctuation">(</span>PATH<span class="token punctuation">)</span>
model<span class="token punctuation">.</span>load_state_dict<span class="token punctuation">(</span>checkpoint<span class="token punctuation">[</span><span class="token string">&#39;model_state_dict&#39;</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
optimizer<span class="token punctuation">.</span>load_state_dict<span class="token punctuation">(</span>checkpoint<span class="token punctuation">[</span><span class="token string">&#39;optimizer_state_dict&#39;</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
epoch <span class="token operator">=</span> checkpoint<span class="token punctuation">[</span><span class="token string">&#39;epoch&#39;</span><span class="token punctuation">]</span>
loss <span class="token operator">=</span> checkpoint<span class="token punctuation">[</span><span class="token string">&#39;loss&#39;</span><span class="token punctuation">]</span>

<span class="token comment">## training mode or evaluation mode</span>
model<span class="token punctuation">.</span><span class="token builtin">eval</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token comment"># - or -</span>
model<span class="token punctuation">.</span>train<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</code></pre><h3 id="_6-迁移学习下的热启动模式" tabindex="-1"><a class="header-anchor" href="#_6-迁移学习下的热启动模式" aria-hidden="true">#</a> 6.迁移学习下的热启动模式</h3><ul><li>只需设置strict=False来忽略非匹配的模型层参数</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>torch<span class="token punctuation">.</span>save<span class="token punctuation">(</span>model<span class="token punctuation">.</span>state_dict<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>PATH<span class="token punctuation">)</span>

model <span class="token operator">=</span> Net<span class="token punctuation">(</span><span class="token punctuation">)</span>
model<span class="token punctuation">.</span>load_state_dict<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>load<span class="token punctuation">(</span>PATH<span class="token punctuation">)</span><span class="token punctuation">,</span>strict<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>&lt;All keys matched successfully&gt;
</code></pre>`,26),o=[p];function i(c,l){return s(),a("div",null,o)}const r=n(t,[["render",i],["__file","lesson7_saveload.html.vue"]]);export{r as default};
