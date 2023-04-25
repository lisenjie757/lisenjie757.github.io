import{_ as n,Y as e,Z as i,$ as s,a0 as a,a1 as m,a2 as t,D as r}from"./framework-0d329256.js";const p={},c=s("h1",{id:"_3-【经典网络】transformer-论文精读",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#_3-【经典网络】transformer-论文精读","aria-hidden":"true"},"#"),a(" 3.【经典网络】Transformer 论文精读")],-1),h={href:"https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",target:"_blank",rel:"noopener noreferrer"},o=t('<h2 id="_0-核心总结" tabindex="-1"><a class="header-anchor" href="#_0-核心总结" aria-hidden="true">#</a> 0. 核心总结</h2><p>核心是提出了一种<strong>残差结构</strong>，极好地解决了网络精度随着深度的增加而下降的问题，使得可以通过堆叠层数的方式来提升精度。</p><h2 id="_1-摘要" tabindex="-1"><a class="header-anchor" href="#_1-摘要" aria-hidden="true">#</a> 1. 摘要</h2><figure><img src="https://i.imgur.com/VHXz0Mr.png" alt="屏幕截图 2023-04-13 023046" tabindex="0" loading="lazy"><figcaption>屏幕截图 2023-04-13 023046</figcaption></figure><h2 id="_2-引言" tabindex="-1"><a class="header-anchor" href="#_2-引言" aria-hidden="true">#</a> 2. 引言</h2><p>卷积神经网络的深度是非常重要的，之前在ImageNet数据集上取得领先结果的经典模型都从深度上受益，但是现在深度卷积神经网络正面临臭名昭著的梯度消失/爆炸问题。</p><p>虽然这个问题目前可以通过Normalization来解决，但却又出现了退化问题：随着网络深度的增加，精度达到饱和，然后迅速退化，即<strong>更深的网络精度反而下降</strong>。这种退化并不是由过拟合引起的，因为训练误差同样增高，实验结果如下图：</p><figure><img src="https://i.imgur.com/pnZMQpr.png" alt="不同层数卷积神经网络的训练和测试误差" tabindex="0" loading="lazy"><figcaption>不同层数卷积神经网络的训练和测试误差</figcaption></figure><p>根据朴素的直觉，即使更深的网络的深层部分“什么都不做”，即输出是输入的恒等映射，深层网络起码也不应该比浅层网络的误差高，但这样的实验结果表明目前的优化器找不到这样的解。</p><p>因此，本文引入了一个残差块来解决退化问题，如下图所示，残差块的核心是通过一个捷径连接（Shortcut Connections，或叫做残差连接）跳过一层或多层卷积将输出增加到层的输出中，添加一个输入的恒等映射。</p>',10),g=s("p",null,[a("用数学公式表达，即我们希望让卷积层拟合一个残差映射 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{stretchy:"false"},")"),s("mo",null,"+"),s("mi",null,"x")]),s("annotation",{encoding:"application/x-tex"},"F(x)+x")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4306em"}}),s("span",{class:"mord mathnormal"},"x")])])]),a(" ，而不是原映射 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"F(x)")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},")")])])]),a(" ，因为通过实验发现卷积层这样的非线性层很难拟合一个恒等映射，所以我们直接增加一个恒等映射。")],-1),u=t('<figure><img src="https://i.imgur.com/VFWqkqe.png" alt="残差块结构示意图" tabindex="0" loading="lazy"><figcaption>残差块结构示意图</figcaption></figure><p>通过在ImageNet数据集上的全面实验表明：</p><ol><li>本文的深度残差网络更易于优化，相比简单堆叠的“普通网络”有更低的训练误差；</li><li>本文的深度残差网络可以很容易得从大幅增加的深度中获得精度增益，产出的结果明显优于以前的网络。</li></ol><p>本文也在CIFAR-10数据集上做了同样的测试，得到了同样的优异的精度表现，并探索训练了超过1000层的模型。</p><p>本文训练的152层残差网络在ImageNet测试集上取得了3.57%的top-5误差，同时仍比VGG网络有更低的复杂度，赢得了ILSVRC 2015分类竞赛的第一名；极深的网络在其它识别任务上也具有出色的泛化性能，在ILSVRC和COCO 2015竞赛中进一步获得第1名（ImageNet检测、ImageNet定位、COCO检测和COCO分割）。这表明残差学习的原理具有普适性，它在其他视觉和非视觉问题中也是同样适用的。</p><h2 id="_3-resnet网络细节" tabindex="-1"><a class="header-anchor" href="#_3-resnet网络细节" aria-hidden="true">#</a> 3. ResNet网络细节</h2><h3 id="_3-1-如何增加残差连接" tabindex="-1"><a class="header-anchor" href="#_3-1-如何增加残差连接" aria-hidden="true">#</a> 3.1 如何增加残差连接</h3><p>在数学公式上，残差块的定义如下：</p>',8),d=s("p",{class:"katex-block"},[s("span",{class:"katex-display"},[s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("semantics",null,[s("mrow",null,[s("mi",null,"y"),s("mo",null,"="),s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{separator:"true"},","),s("mo",{stretchy:"false"},"{"),s("msub",null,[s("mi",null,"W"),s("mi",null,"i")]),s("mo",{stretchy:"false"},"}"),s("mo",{stretchy:"false"},")"),s("mo",null,"+"),s("mi",null,"x")]),s("annotation",{encoding:"application/x-tex"}," y = F(x,\\{W_i\\}) + x ")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"="),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mopen"},"{"),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3117em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"i")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mclose"},"})"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4306em"}}),s("span",{class:"mord mathnormal"},"x")])])])])],-1),x=s("p",null,[a("其中， "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"x"),s("mo",{separator:"true"},","),s("mi",null,"y")]),s("annotation",{encoding:"application/x-tex"},"x,y")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y")])])]),a(" 分别为层的输入和输出向量，函数 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{separator:"true"},","),s("mo",{stretchy:"false"},"{"),s("msub",null,[s("mi",null,"W"),s("mi",null,"i")]),s("mo",{stretchy:"false"},"}"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"F(x,\\{W_i\\})")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mopen"},"{"),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3117em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"i")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mclose"},"})")])])]),a(" 表示要学习的残差映射。")],-1),y=s("p",null,[a("例如，对于上一幅图所示的两层残差块， "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",null,"="),s("msub",null,[s("mi",null,"W"),s("mn",null,"2")]),s("mo",null,"⋅"),s("mi",null,"R"),s("mi",null,"e"),s("mi",null,"L"),s("mi",null,"U"),s("mo",{stretchy:"false"},"("),s("msub",null,[s("mi",null,"W"),s("mn",null,"1")]),s("mi",null,"x"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"F = W_2 \\cdot ReLU(W_1x)")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6833em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"="),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8333em","vertical-align":"-0.15em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3011em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},"2")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"⋅"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.00773em"}},"R"),s("span",{class:"mord mathnormal"},"e"),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10903em"}},"LU"),s("span",{class:"mopen"},"("),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3011em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},"1")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},")")])])]),a(" ，为了简化表示此处省略了偏差（Bias）。接着，残差连接通过 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",null,"+"),s("mi",null,"x")]),s("annotation",{encoding:"application/x-tex"},"F + x")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.7667em","vertical-align":"-0.0833em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4306em"}}),s("span",{class:"mord mathnormal"},"x")])])]),a(" 操作逐元素相加来实现，相加后再通过一层ReLU输入下一个残差块。")],-1),f=s("p",null,[a("此外需要注意，在等式中 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"x")]),s("annotation",{encoding:"application/x-tex"},"x")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4306em"}}),s("span",{class:"mord mathnormal"},"x")])])]),a(" 和 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F")]),s("annotation",{encoding:"application/x-tex"},"F")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6833em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F")])])]),a(" 的维度必须相等，若不相等则可以通过一个线性投影 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("msub",null,[s("mi",null,"W"),s("mi",null,"s")])]),s("annotation",{encoding:"application/x-tex"},"W_s")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8333em","vertical-align":"-0.15em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.1514em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"s")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])])])])]),a(" 来匹配维度，如下式：")],-1),v=s("p",{class:"katex-block"},[s("span",{class:"katex-display"},[s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("semantics",null,[s("mrow",null,[s("mi",null,"y"),s("mo",null,"="),s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{separator:"true"},","),s("mo",{stretchy:"false"},"{"),s("msub",null,[s("mi",null,"W"),s("mi",null,"i")]),s("mo",{stretchy:"false"},"}"),s("mo",{stretchy:"false"},")"),s("mo",null,"+"),s("msub",null,[s("mi",null,"W"),s("mi",null,"s")]),s("mi",null,"x")]),s("annotation",{encoding:"application/x-tex"}," y = F(x,\\{W_i\\}) + W_s x ")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"="),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mopen"},"{"),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3117em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"i")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mclose"},"})"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8333em","vertical-align":"-0.15em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.1514em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"s")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mord mathnormal"},"x")])])])])],-1),_=s("p",null,[a("此外，以上公式符号为了简单起见表示的是全连接层，但对于卷积层同样也是使用的，函数 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{separator:"true"},","),s("mo",{stretchy:"false"},"{"),s("msub",null,[s("mi",null,"W"),s("mi",null,"i")]),s("mo",{stretchy:"false"},"}"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"F(x,\\{W_i\\})")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mopen"},"{"),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"W"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.3117em"}},[s("span",{style:{top:"-2.55em","margin-left":"-0.1389em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"i")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.15em"}},[s("span")])])])])]),s("span",{class:"mclose"},"})")])])]),a(" 可以表示多个卷积层，逐元素相加在卷积层上即逐通道相加，线性映射在卷积层中一般通过一个1×1的卷积核来实现维度匹配。")],-1),b=t('<h3 id="_3-2-网络架构" tabindex="-1"><a class="header-anchor" href="#_3-2-网络架构" aria-hidden="true">#</a> 3.2 网络架构</h3><p>下图展示了34层的ResNet架构，并与34层的“普通”网络及VGG-19的结构对比：</p><figure><img src="https://i.imgur.com/hhl8lxx.png" alt="ResNet-34架构与其它架构的对比" tabindex="0" loading="lazy"><figcaption>ResNet-34架构与其它架构的对比</figcaption></figure><p>下表给出了ResNet架构的更多细节及更多变体：</p><figure><img src="https://i.imgur.com/Oz8YCZD.png" alt="ResNet不同架构的配置" tabindex="0" loading="lazy"><figcaption>ResNet不同架构的配置</figcaption></figure><p>其中，对于50层以上的深层网络，为了降低其计算复杂度，本文设计了一种瓶颈架构（Bottleneck），如下图所示。即先通过一个1×1的卷积先降低维度，再通过具有较小输入和输出维度的3×3卷积瓶颈，最后再通过一个1×1的卷积恢复维度。通过这样的瓶颈设计可以有效地减少模型的参数两和复杂度。</p><figure><img src="https://i.imgur.com/iOc24tZ.png" alt="屏幕截图 2023-03-26 103814" tabindex="0" loading="lazy"><figcaption>屏幕截图 2023-03-26 103814</figcaption></figure><h3 id="_3-3-网络训练细节" tabindex="-1"><a class="header-anchor" href="#_3-3-网络训练细节" aria-hidden="true">#</a> 3.3 网络训练细节</h3>',8),w=s("ol",null,[s("li",null,"数据增强：对图像的短边在[256,480]中随机采样进行尺度缩放；对图像或其随机翻转进行224×224的随机剪裁采样，并减去每个像素的均值；标准颜色增强。"),s("li",null,"在每次卷积之后、激活之前采用批归一化（Batch Normalization）。"),s("li",null,[a("初始化权重从头开始训练，使用SGD优化器，批大小（Batch Size）为256；学习率从0.1开始，当loss不再下降时除以10；最多训练 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mn",null,"60"),s("mo",null,"×"),s("mn",null,"1"),s("msup",null,[s("mn",null,"0"),s("mn",null,"4")])]),s("annotation",{encoding:"application/x-tex"},"60 \\times 10^4")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.7278em","vertical-align":"-0.0833em"}}),s("span",{class:"mord"},"60"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"×"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8141em"}}),s("span",{class:"mord"},"1"),s("span",{class:"mord"},[s("span",{class:"mord"},"0"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8141em"}},[s("span",{style:{top:"-3.063em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},"4")])])])])])])])])])]),a(" 个迭代（iteration）；权重衰减（Weight Decay）为0.0001，动量（Momentum）为0.9；不使用Dropout。")]),s("li",null,"（为了刷榜，没多大必要）在测试时采用标准的10-crop测试，为了获得最佳结果，采用全卷积网络形式，并在多尺度图像上取平均精度（对图像的短边进行缩放在{224,256,384,480,640}中取值）。")],-1),k=t('<h2 id="_4-实验与结果" tabindex="-1"><a class="header-anchor" href="#_4-实验与结果" aria-hidden="true">#</a> 4. 实验与结果</h2><p>本文在由1000个类组成的ImageNet-2012分类数据集上进行评估。模型在128万张训练集图像上进行训练，在5万张验证集图像上进行测试，得到top-1和top-5错误率。</p><p>下表对比了不同层数的ResNet和普通网络在ImageNet验证集上的Top-1误差：</p><figure><img src="https://i.imgur.com/LQorajE.png" alt="ResNet和普通网络的对比" tabindex="0" loading="lazy"><figcaption>ResNet和普通网络的对比</figcaption></figure><p>下图展示了它们在训练过程中的训练/验证误差：</p><figure><img src="https://i.imgur.com/OKkMSIU.png" alt="ResNet和普通网络的训练过程" tabindex="0" loading="lazy"><figcaption>ResNet和普通网络的训练过程</figcaption></figure><p>上述图表结果均表明，普通网络的精度随着深度的增加反而会退化，而ResNet则可以从增加的深度中获得精度增益，同时可以发现，ResNet的收敛速度会比普通网络更快。</p><p>本文还研究了三种恒等映射方式：</p><ol><li>A：使用补零的方式来实现维度匹配；</li><li>B：不同维度则使用线性投影来实现维度匹配，相同维度则直接相加；</li><li>C：不论维度是否匹配均使用线性投影。</li></ol><p>实验结果如下表所示，下表为10-crop测试结果，可以发现C略优于B，优于A，但使用C会导致参数量和复杂度过大，所以接下来的实验均采用B方案。</p><figure><img src="https://i.imgur.com/lfMBvg5.png" alt="不同网络架构的10-crop测试结果" tabindex="0" loading="lazy"><figcaption>不同网络架构的10-crop测试结果</figcaption></figure><p>此外，上表还展示了ResNet-50/101/152三种深层ResNet架构的测试结果，可以发现没有出现退化现象，均从深度中获得了精度增益。</p><p>下表展示了单模型的测试结果，得到的结论是相同的。</p><figure><img src="https://i.imgur.com/HBlhfBt.png" alt="不同网络架构的单模型测试结果" tabindex="0" loading="lazy"><figcaption>不同网络架构的单模型测试结果</figcaption></figure><p>此外，上述两个测试结果表均展示与之前SOTA单模型的对比结果，可以发现ResNet-50就由于之前所有的SOTA单模型。</p><p>本文还将6个不同深度的ResNet模型进行集成（两个152层，其余深度各一个），这个集成模型的精度最优，在ILSVRC 2015竞赛中取得了第一名，测试结果如下表所示，下表为与ImageNet测试集上排名前5的模型的对比结果：</p><figure><img src="https://i.imgur.com/RDubP6g.png" alt="与ImageNet测试集排名前5的模型的对比结果" tabindex="0" loading="lazy"><figcaption>与ImageNet测试集排名前5的模型的对比结果</figcaption></figure><h2 id="_5-个人思考" tabindex="-1"><a class="header-anchor" href="#_5-个人思考" aria-hidden="true">#</a> 5. 个人思考</h2><p>此前的卷积神经网络架构一直在追求更深的深度，因为能够从增加的深度中获得精度增益，但当深度到了某一界限，更深的层非但没有起作用，反而起了副作用，出现了梯度消失/爆炸的现象。</p><p>本文揭示了发生这种现象的本质原因是由于我们目前的优化器很难去优化到一个恒等映射，即输出对于输入总会发生一定程度的偏移，随着深度的增加，这种偏移会离最优解越偏越远，因此本文将前几层输入信息引入下一层，增加了下一层输入恒等映射的权重，让下一层有选择拟合恒等映射的空间。</p><p>个人的另一种理解是，神经网络的深层神经元会“遗忘”神经网络浅层的特征信息，所以需要通过不断加入浅层信息的方式去“提醒”深层的神经元这个输入的底层特征，来帮助深层神经元做出更优的推断。</p>',21);function M(z,N){const l=r("ExternalLinkIcon");return e(),i("div",null,[c,s("blockquote",null,[s("p",null,[a("原文链接："),s("a",h,[a("Attention Is All You Need"),m(l)])])]),o,g,u,d,x,y,f,v,_,b,w,k])}const R=n(p,[["render",M],["__file","3. Transformer.html.vue"]]);export{R as default};
