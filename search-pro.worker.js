const f=(e,s)=>{const n=e.toLowerCase(),o=s.toLowerCase(),a=[];let l=0,r=0;const u=(t,p=!1)=>{let c="";r===0?c=t.length>20?`… ${t.slice(-20)}`:t:p?c=t.length+r>100?`${t.slice(0,100-r)}… `:t:c=t.length>20?`${t.slice(0,20)} … ${t.slice(-20)}`:t,c&&a.push(c),r+=c.length,p||(a.push(["strong",s]),r+=s.length,r>=100&&a.push(" …"))};let i=n.indexOf(o,l);if(i===-1)return null;for(;i>=0;){const t=i+o.length;if(u(e.slice(l,i)),l=t,r>100)break;i=n.indexOf(o,l)}return r<100&&u(e.slice(l),!0),a};function $(e){return e}const h=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{},d="__vueuse_ssr_handlers__";h[d]=h[d]||{};var g;(function(e){e.UP="UP",e.RIGHT="RIGHT",e.DOWN="DOWN",e.LEFT="LEFT",e.NONE="NONE"})(g||(g={}));var m=Object.defineProperty,y=Object.getOwnPropertySymbols,C=Object.prototype.hasOwnProperty,w=Object.prototype.propertyIsEnumerable,I=(e,s,n)=>s in e?m(e,s,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[s]=n,N=(e,s)=>{for(var n in s||(s={}))C.call(s,n)&&I(e,n,s[n]);if(y)for(var n of y(s))w.call(s,n)&&I(e,n,s[n]);return e};const Q={easeInSine:[.12,0,.39,0],easeOutSine:[.61,1,.88,1],easeInOutSine:[.37,0,.63,1],easeInQuad:[.11,0,.5,0],easeOutQuad:[.5,1,.89,1],easeInOutQuad:[.45,0,.55,1],easeInCubic:[.32,0,.67,0],easeOutCubic:[.33,1,.68,1],easeInOutCubic:[.65,0,.35,1],easeInQuart:[.5,0,.75,0],easeOutQuart:[.25,1,.5,1],easeInOutQuart:[.76,0,.24,1],easeInQuint:[.64,0,.78,0],easeOutQuint:[.22,1,.36,1],easeInOutQuint:[.83,0,.17,1],easeInExpo:[.7,0,.84,0],easeOutExpo:[.16,1,.3,1],easeInOutExpo:[.87,0,.13,1],easeInCirc:[.55,0,1,.45],easeOutCirc:[0,.55,.45,1],easeInOutCirc:[.85,0,.15,1],easeInBack:[.36,0,.66,-.56],easeOutBack:[.34,1.56,.64,1],easeInOutBack:[.68,-.6,.32,1.6]};N({linear:$},Q);const b=Object.entries,v=Object.keys,E=e=>e.reduce((s,{type:n})=>s+(n==="title"?50:n==="heading"?20:n==="custom"?10:1),0),_=(e,s)=>{var n;const o={};for(const[a,l]of b(s)){const r=((n=s[a.replace(/\/[^\\]*$/,"")])==null?void 0:n.title)||"",u=`${r?`${r} > `:""}${l.title}`,i=f(l.title,e);i&&(o[u]=[...o[u]||[],{type:"title",path:a,display:i}]),l.customFields&&b(l.customFields).forEach(([t,p])=>{p.forEach(c=>{const O=f(c,e);O&&(o[u]=[...o[u]||[],{type:"custom",path:a,index:t,display:O}])})});for(const t of l.contents){const p=f(t.header,e);p&&(o[u]=[...o[u]||[],{type:"heading",path:a+(t.slug?`#${t.slug}`:""),display:p}]);for(const c of t.contents){const O=f(c,e);O&&(o[u]=[...o[u]||[],{type:"content",header:t.header,path:a+(t.slug?`#${t.slug}`:""),display:O}])}}}return v(o).sort((a,l)=>E(o[a])-E(o[l])).map(a=>({title:a,contents:o[a]}))},P=JSON.parse("{\"/\":{\"/\":{\"title\":\"\",\"contents\":[{\"header\":\"\",\"slug\":\"\",\"contents\":[\"This is my knowledge reposity.\"]}]},\"/intro.html\":{\"title\":\"个人介绍\",\"contents\":[{\"header\":\"\",\"slug\":\"\",\"contents\":[\"浙江大学硕士\"]}]},\"/knowledge/\":{\"title\":\"\",\"contents\":[{\"header\":\"\",\"slug\":\"\",\"contents\":[\"This is my knowledge reposity！！！！！.\"]}]},\"/knowledge/paper/1.%20AlexNet.html\":{\"title\":\"1.【经典网络】AlexNet 论文精读\",\"contents\":[{\"header\":\"\",\"slug\":\"\",\"contents\":[\"原文链接：ImageNet Classification with Deep Convolutional Neural Networks\"]},{\"header\":\"0. 核心总结\",\"slug\":\"_0-核心总结\",\"contents\":[\"核心是提出了一种深层卷积神经网络的架构，并引入了GPU训练、ReLU激活函数、Dropout正则化这些训练深层卷积神经网络的手段。\"]},{\"header\":\"1. 摘要\",\"slug\":\"_1-摘要\",\"contents\":[\"训练了一个大型的卷积神经网络，在ImageNet LSVRC-2010和ILSVRC-2012图像分类竞赛中取得了远低于第二名错误率的成绩。\"]},{\"header\":\"2. 引言\",\"slug\":\"_2-引言\",\"contents\":[\"事实证明，大图像数据集比小图像数据集能够训练更优的模型，但同时对于ImageNet这样的超大型数据集也需要与之对应的具有强大学习能力的模型，CNN对图像的性质（局部平稳和参数共享）使其比相同性能的标准前馈神经网络参数量要少得多。\",\"尽管CNN具有如此优秀的性质，但计算它还是很昂贵的，然而得益于GPU使得训练大型的CNN网络成为了可能，并且大型图像数据集ImageNet的出现使得模型不会出现严重的过拟合。\",\"本文的具体贡献如下：\",\"在ImageNet子集上训练了迄今为止最大的卷积神经网络，并取得了迄今为止最好的成绩；\",\"编写了一个高度优化的2D卷积的GPU实现，以及提供了训练卷积神经网络的所有操作细节，包括如何提高性能、减少训练时间等；\",\"即使有大量的数据，但由于网络的太大还是会有过拟合的问题，所以采用了几种有效的技术来防止过拟合。\"]},{\"header\":\"3. 数据集\",\"slug\":\"_3-数据集\",\"contents\":[\"使用了著名的ImageNet数据集，介绍直接从原文翻译。\",\"ImageNet是一个由超过1500万张标注的高分辨率图像组成的数据集，属于大约22，000个类别。这些图像是从网上收集的，并由人类贴标者使用亚马逊的研究工具众包工具进行标记。从2010年开始，作为Pascal视觉对象挑战赛的一部分，每年举办一次名为ImageNet大规模视觉识别挑战赛( ILSVRC )的竞赛。ILSVRC使用ImageNet的一个子集，在1000个类别中的每个类别中大约有1000个图像。总共有大约120万张训练图像、50,000张验证图像和150,000张测试图像。 ILSVRC-2010是唯一一个测试集标签可用的ILSVRC版本，因此这是我们进行大部分实验的版本。由于我们也在ILSVRC-2012竞赛中输入了我们的模型，在第6节中我们也报告了我们在这个版本的数据集上的结果，对于这个版本的数据集，测试集标签是不可用的。在ImageNet上，通常报告两个错误率：top-1和top-5，其中top-5错误率是模型认为最可能的5个标签中没有正确标签的测试图像的分数。 ImageNet由可变分辨率的图像组成，而我们的系统需要一个恒定的输入维度。因此，我们将图像降采样到256×256的固定分辨率。给定一个矩形图像，我们首先对图像进行缩放，使较短的边长为256，然后从结果图像中裁剪出中心的256×256块。我们没有以任何其他方式预处理图像，除了从每个像素中减去训练集的平均活动。因此，我们在像素的(中心)原始RGB值上训练我们的网络。\"]},{\"header\":\"4. 架构\",\"slug\":\"_4-架构\",\"contents\":[\"AlexNet的网络架构如下图所示，它包含5个卷积层和3个全连接层，下面会介绍AlexNet网络一些新颖的特征，以下特征根据重要性向下排序。\",\"AlexNet网络架构图\"]},{\"header\":\"4.1 ReLU激活函数\",\"slug\":\"_4-1-relu激活函数\",\"contents\":[\"ReLU激活函数的公式：\",\"$f(x) = max(0,x)$\",\"本文通过大量实验证明使用ReLU作为激活函数比使用tanh作激活函数训练速度要快好几倍，如下图所示：\",\"relu与tanh速度对比图\"]},{\"header\":\"4.2 多GPU训练\",\"slug\":\"_4-2-多gpu训练\",\"contents\":[\"本文使用两块GTX 580 GPU来训练AlexNet，采用并行化方案，每个GPU上放置一半的内核。\"]},{\"header\":\"4.3 局部响应归一化（LRN，Local Response Normalization）\",\"slug\":\"_4-3-局部响应归一化-lrn-local-response-normalization\",\"contents\":[\"一种提高泛化性的手段，现在已不常用，被批归一化Batch Normalization代替。\"]},{\"header\":\"4.4 重叠池化（Overlapping Pooling）\",\"slug\":\"_4-4-重叠池化-overlapping-pooling\",\"contents\":[\"即步长要小于池化单元的大小，使得每一个池化单元组成的网格有所重叠，这样能稍微降低点错误率。\"]},{\"header\":\"4.5 整体架构\",\"slug\":\"_4-5-整体架构\",\"contents\":[\"AlexNet包含8个带权重的层，前5个为卷积层，其余3个为全连接层，最后输出到一个1000类的Softmax层。\",\"第2层，第4层，第5层卷积层内核仅连接到位于同一GPU上的前一层内核。第3层内核连接到第二层的所有内核，全连接层也与上一层的所有神经元相连。局部相应归一化应用于第1层和第2层，最大池化层是重叠池化，使用在第1层，第2层和第5层。\"]},{\"header\":\"5. 减少过拟合\",\"slug\":\"_5-减少过拟合\",\"contents\":[\"第一种方式是使用数据增强，包括以下两种形式：\",\"通过图像平移和水平反射生成图像；\",\"改变训练图像中RGB通道的强度，即对整个ImageNet训练集的RGB像素值集合进行PCA主成分分析，将权重加到对应的RGB通道上。\",\"第二种方式是使用Dropout方法，即在每轮前向传播和反向传播中，每层的神经元将会有一定的比例“失活”，即在本轮置为0，不参与前向传播和反向传播的过程。\"]},{\"header\":\"6. 训练细节\",\"slug\":\"_6-训练细节\",\"contents\":[\"优化器为SGD，batch size为128，momentum为0.9，权重衰减（L2正则）为5e-4。\",\"从一个均值为0，标准差为0.01的高斯分布中初始化各层权重，用常数1初始化第2层，第4层，第5层和全连接层的偏置，用常数0初始化其余层的偏置。\",\"每层的学习率相同，并在训练过程中手动调整。学习率初始化为0.01，当验证集错误率停止下降时，将学习率减小10倍继续训练，在终止训练前减小3次。\",\"本文在120万张图像训练集上训练了90个epoch，在两个NVDIA GTX 580 3GB GPU上训练了5～6天的时间。\"]},{\"header\":\"7. 结果\",\"slug\":\"_7-结果\",\"contents\":[\"在ILSVRC-2010测试集上与其它方法的对比结果：\",\"AlexNet测试结果图1\",\"在ILSVRC-2012验证集和测试集上的对比结果，*号表示在完整版ImageNet数据集上进行过预训练再微调，1 CNN表示一个标准的AlexNet，5 CNN表示训练5个AlexNet将预测结果取平均，7 CNN表示在AlexNet最后一个池化层后再加上第6个卷积层：\",\"AlexNet测试结果图2\",\"下图展示了第1层卷积层的96个卷积核的可视化情况，表征了卷积核所学习到的特征：\",\"AlexNet卷积核可视化\",\"下图展示了AlexNet在8张测试图像上的top-5预测结果，可以发现即使是偏离中心的对象（如第1幅图）也可以被网络识别，并且大多数图像top-5预测结果尽管并非完全正确但似乎也是合理的，同时也可以注意到部分图像（如第7幅图）本身就具有模糊性：\",\"AlexNet图像预测\",\"下图展示了在AlexNet最后一个隐藏层的特征向量欧氏距离最小的几幅图像（每行），可以发现特征向量欧式距离相近的图片确实表征的是同一个对象，即使对象的位置、朝向、明暗不同，这表明卷积神经网络可以很好的提取出图像的语义特征\",\"AlexNet最后一个隐藏层所表征的特征向量\"]},{\"header\":\"8. 思考\",\"slug\":\"_8-思考\",\"contents\":[\"本文认为卷积神经网络的深度是很重要的，减小深度会导致性能的下降。\",\"AlexNet虽然在图像分类任务上取得了很大的提升，但对比人类视觉系统的识别还是有一定差距的。\"]}]},\"/knowledge/paper/%E6%96%87%E7%8C%AE%E9%98%85%E8%AF%BB%E6%96%B9%E6%B3%95%E8%AE%BA.html\":{\"title\":\"hellow\",\"contents\":[]},\"/knowledge/paper/\":{\"title\":\"Paper\",\"contents\":[]}}}");self.onmessage=({data:e})=>{self.postMessage(_(e.query,P[e.routeLocale]))};
//# sourceMappingURL=original.js.map
