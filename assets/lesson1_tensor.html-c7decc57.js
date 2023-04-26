import{_ as n,Y as s,Z as a,a0 as t}from"./framework-c2b1cf81.js";const p={},e=t(`<h1 id="lesson1-张量操作" tabindex="-1"><a class="header-anchor" href="#lesson1-张量操作" aria-hidden="true">#</a> lesson1. 张量操作</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> torch
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="初始化张量" tabindex="-1"><a class="header-anchor" href="#初始化张量" aria-hidden="true">#</a> 初始化张量</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">##直接数据创建</span>
data <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span><span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">]</span><span class="token punctuation">]</span>
x_data <span class="token operator">=</span> torch<span class="token punctuation">.</span>tensor<span class="token punctuation">(</span>data<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>x_data<span class="token punctuation">)</span>

<span class="token comment">##通过numpy转换</span>
np_array <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>data<span class="token punctuation">)</span>
x_np <span class="token operator">=</span> torch<span class="token punctuation">.</span>from_numpy<span class="token punctuation">(</span>np_array<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>x_np<span class="token punctuation">)</span>

<span class="token comment">##保持特定属性创建</span>
x_ones <span class="token operator">=</span> torch<span class="token punctuation">.</span>ones_like<span class="token punctuation">(</span>x_data<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>x_ones<span class="token punctuation">)</span>
x_rand <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand_like<span class="token punctuation">(</span>x_data<span class="token punctuation">,</span>dtype<span class="token operator">=</span>torch<span class="token punctuation">.</span><span class="token builtin">float</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>x_rand<span class="token punctuation">)</span>

<span class="token comment">##以特定的维度构造特定属性的张量</span>
shape <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token punctuation">)</span>
rand_tensor <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand<span class="token punctuation">(</span>shape<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>rand_tensor<span class="token punctuation">)</span>
ones_tensor <span class="token operator">=</span> torch<span class="token punctuation">.</span>ones<span class="token punctuation">(</span>shape<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>ones_tensor<span class="token punctuation">)</span>
zeros_tensor <span class="token operator">=</span> torch<span class="token punctuation">.</span>zeros<span class="token punctuation">(</span>shape<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>zeros_tensor<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>    tensor([[1, 2],
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
</code></pre><h2 id="张量的属性" tabindex="-1"><a class="header-anchor" href="#张量的属性" aria-hidden="true">#</a> 张量的属性</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>tensor <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">.</span>shape<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">.</span>dtype<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">.</span>device<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[0.9527, 0.5502, 0.1051, 0.5587],
        [0.7882, 0.5084, 0.8422, 0.6339],
        [0.1735, 0.7458, 0.3827, 0.7142]])
torch.Size([3, 4])
torch.float32
cpu
</code></pre><h2 id="张量运算" tabindex="-1"><a class="header-anchor" href="#张量运算" aria-hidden="true">#</a> 张量运算</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">##张量在GPU运算</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>torch<span class="token punctuation">.</span>__version__<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>torch<span class="token punctuation">.</span>cuda<span class="token punctuation">.</span>is_available<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
tensor <span class="token operator">=</span> tensor<span class="token punctuation">.</span>to<span class="token punctuation">(</span><span class="token string">&#39;cuda&#39;</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>

<span class="token comment">##标准索引和切片运算</span>
tensor <span class="token operator">=</span> torch<span class="token punctuation">.</span>ones<span class="token punctuation">(</span><span class="token number">4</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">[</span><span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">,</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
tensor<span class="token punctuation">[</span><span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token number">0</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>

<span class="token comment">##张量连接运算</span>
t1 <span class="token operator">=</span> torch<span class="token punctuation">.</span>cat<span class="token punctuation">(</span><span class="token punctuation">[</span>tensor<span class="token punctuation">,</span>tensor<span class="token punctuation">,</span>tensor<span class="token punctuation">]</span><span class="token punctuation">,</span>dim<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>t1<span class="token punctuation">)</span>

<span class="token comment">##矩阵乘法（下面三种写法是一样的）</span>
y1 <span class="token operator">=</span> tensor @ tensor<span class="token punctuation">.</span>T
y2 <span class="token operator">=</span> tensor<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>tensor<span class="token punctuation">.</span>T<span class="token punctuation">)</span>
y3 <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand_like<span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>tensor<span class="token punctuation">,</span>tensor<span class="token punctuation">.</span>T<span class="token punctuation">,</span>out<span class="token operator">=</span>y3<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>y1<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>y2<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>y3<span class="token punctuation">)</span>

<span class="token comment">##矩阵对应元素点乘（下面三种写法是一样的）</span>
z1 <span class="token operator">=</span> tensor <span class="token operator">*</span> tensor
z2 <span class="token operator">=</span> tensor<span class="token punctuation">.</span>mul<span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
z3 <span class="token operator">=</span> torch<span class="token punctuation">.</span>rand_like<span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
torch<span class="token punctuation">.</span>mul<span class="token punctuation">(</span>tensor<span class="token punctuation">,</span>tensor<span class="token punctuation">,</span>out<span class="token operator">=</span>z3<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>z1<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>z2<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>z3<span class="token punctuation">)</span>

<span class="token comment">##单元素张量（可转换为Python元素）</span>
agg <span class="token operator">=</span> tensor<span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>agg<span class="token punctuation">,</span><span class="token builtin">type</span><span class="token punctuation">(</span>agg<span class="token punctuation">)</span><span class="token punctuation">)</span>
agg_item <span class="token operator">=</span> agg<span class="token punctuation">.</span>item<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>agg_item<span class="token punctuation">,</span><span class="token builtin">type</span><span class="token punctuation">(</span>agg_item<span class="token punctuation">)</span><span class="token punctuation">)</span>

<span class="token comment">##原地张量运算（计算微分时会丢失历史信息，不建议使用）</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
tensor<span class="token punctuation">.</span>add_<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>tensor<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>1.12.0
True
tensor([[0.9527, 0.5502, 0.1051, 0.5587],
        [0.7882, 0.5084, 0.8422, 0.6339],
        [0.1735, 0.7458, 0.3827, 0.7142]], device=&#39;cuda:0&#39;)
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
tensor(12.) &lt;class &#39;torch.Tensor&#39;&gt;
12.0 &lt;class &#39;float&#39;&gt;
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
</code></pre><h2 id="与numpy互相转换" tabindex="-1"><a class="header-anchor" href="#与numpy互相转换" aria-hidden="true">#</a> 与Numpy互相转换</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">## tensor to numpy</span>
t <span class="token operator">=</span> torch<span class="token punctuation">.</span>ones<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>t<span class="token punctuation">)</span>
n <span class="token operator">=</span> t<span class="token punctuation">.</span>numpy<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>n<span class="token punctuation">)</span>

<span class="token comment">## numpy to tensor</span>
n <span class="token operator">=</span> np<span class="token punctuation">.</span>ones<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>n<span class="token punctuation">)</span>
t <span class="token operator">=</span> torch<span class="token punctuation">.</span>from_numpy<span class="token punctuation">(</span>n<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>t<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
[1. 1. 1. 1. 1.]
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
</code></pre>`,14),o=[e];function c(i,u){return s(),a("div",null,o)}const r=n(p,[["render",c],["__file","lesson1_tensor.html.vue"]]);export{r as default};
