import{_ as n,Y as a,Z as s,a0 as t}from"./framework-c2b1cf81.js";const p={},o=t(`<h1 id="lesson5-自动求导机制" tabindex="-1"><a class="header-anchor" href="#lesson5-自动求导机制" aria-hidden="true">#</a> lesson5. 自动求导机制</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> torch
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h2 id="定义最简单的单层神经网络" tabindex="-1"><a class="header-anchor" href="#定义最简单的单层神经网络" aria-hidden="true">#</a> 定义最简单的单层神经网络</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token comment">## input tensor</span>
x <span class="token operator">=</span> torch<span class="token punctuation">.</span>ones<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">)</span>
<span class="token comment">## expected output</span>
y <span class="token operator">=</span> torch<span class="token punctuation">.</span>zeros<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>
w <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">,</span><span class="token number">3</span><span class="token punctuation">,</span>requires_grad<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
b <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span>requires_grad<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
z <span class="token operator">=</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>x<span class="token punctuation">,</span>w<span class="token punctuation">)</span> <span class="token operator">+</span> b
loss <span class="token operator">=</span> torch<span class="token punctuation">.</span>nn<span class="token punctuation">.</span>functional<span class="token punctuation">.</span>binary_cross_entropy_with_logits<span class="token punctuation">(</span>z<span class="token punctuation">,</span>y<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="反向传播函数的引用存储在张量的grad-fn属性中" tabindex="-1"><a class="header-anchor" href="#反向传播函数的引用存储在张量的grad-fn属性中" aria-hidden="true">#</a> 反向传播函数的引用存储在张量的grad_fn属性中</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;z = &quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>z<span class="token punctuation">.</span>grad_fn<span class="token punctuation">)</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;loss = &quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>loss<span class="token punctuation">.</span>grad_fn<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>z = &lt;AddBackward0 object at 0x7f05c46fa040&gt;
loss = &lt;BinaryCrossEntropyWithLogitsBackward0 object at 0x7f05c4845f70&gt;
</code></pre><h2 id="计算梯度" tabindex="-1"><a class="header-anchor" href="#计算梯度" aria-hidden="true">#</a> 计算梯度</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>w<span class="token punctuation">.</span>grad<span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>b<span class="token punctuation">.</span>grad<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>tensor([[0.2915, 0.1360, 0.1072],
        [0.2915, 0.1360, 0.1072],
        [0.2915, 0.1360, 0.1072],
        [0.2915, 0.1360, 0.1072],
        [0.2915, 0.1360, 0.1072]])
tensor([0.2915, 0.1360, 0.1072])
</code></pre><h2 id="禁用梯度跟踪" tabindex="-1"><a class="header-anchor" href="#禁用梯度跟踪" aria-hidden="true">#</a> 禁用梯度跟踪</h2><ul><li>有时已有训练好的模型，我们只想推理，不想更新参数，可以使用torch.no_grad()或detach()来禁用梯度计算</li><li>以下是可能要禁用梯度跟踪的几点理由：1.做微调时可能需要冻结某些参数 2.只做推理时提高计算效率</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>z <span class="token operator">=</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>x<span class="token punctuation">,</span>w<span class="token punctuation">)</span> <span class="token operator">+</span> b
<span class="token keyword">print</span><span class="token punctuation">(</span>z<span class="token punctuation">.</span>requires_grad<span class="token punctuation">)</span>

<span class="token keyword">with</span> torch<span class="token punctuation">.</span>no_grad<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    z <span class="token operator">=</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>x<span class="token punctuation">,</span>w<span class="token punctuation">)</span> <span class="token operator">+</span> b
<span class="token keyword">print</span><span class="token punctuation">(</span>z<span class="token punctuation">.</span>requires_grad<span class="token punctuation">)</span>

z <span class="token operator">=</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>x<span class="token punctuation">,</span>w<span class="token punctuation">)</span> <span class="token operator">+</span> b
z_det <span class="token operator">=</span> z<span class="token punctuation">.</span>detach<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>z_det<span class="token punctuation">.</span>requires_grad<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>True
False
False
</code></pre><h2 id="张量梯度和jacobian-products" tabindex="-1"><a class="header-anchor" href="#张量梯度和jacobian-products" aria-hidden="true">#</a> 张量梯度和Jacobian Products</h2><ol><li>PyTorch backward grad实际计算保存的是Jacobian Products=vT*J，默认v=tensor(1.0)</li><li>PyTorch的机制会累积梯度，所以需要清零，实际训练时优化器会自动帮助我们清零</li></ol><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>inp <span class="token operator">=</span> torch<span class="token punctuation">.</span>eye<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">,</span>requires_grad<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
out <span class="token operator">=</span> <span class="token punctuation">(</span>inp<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">pow</span><span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">)</span>
out<span class="token punctuation">.</span>backward<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>ones_like<span class="token punctuation">(</span>inp<span class="token punctuation">)</span><span class="token punctuation">,</span>retain_graph<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;First Call\\n&quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>inp<span class="token punctuation">.</span>grad<span class="token punctuation">)</span><span class="token punctuation">)</span>
out<span class="token punctuation">.</span>backward<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>ones_like<span class="token punctuation">(</span>inp<span class="token punctuation">)</span><span class="token punctuation">,</span>retain_graph<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;\\nSecond Call\\n&quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>inp<span class="token punctuation">.</span>grad<span class="token punctuation">)</span><span class="token punctuation">)</span>
inp<span class="token punctuation">.</span>grad<span class="token punctuation">.</span>zero_<span class="token punctuation">(</span><span class="token punctuation">)</span>
out<span class="token punctuation">.</span>backward<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>ones_like<span class="token punctuation">(</span>inp<span class="token punctuation">)</span><span class="token punctuation">,</span>retain_graph<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;\\nCall after zeroing gradients\\n&quot;</span><span class="token operator">+</span><span class="token builtin">str</span><span class="token punctuation">(</span>inp<span class="token punctuation">.</span>grad<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><pre><code>First Call
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
</code></pre>`,18),e=[o];function c(i,u){return a(),s("div",null,e)}const r=n(p,[["render",c],["__file","lesson5_autograd.html.vue"]]);export{r as default};
