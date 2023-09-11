const e=JSON.parse('{"key":"v-321b113e","path":"/%E7%9F%A5%E8%AF%86%E5%BA%93/%E6%96%87%E7%8C%AE%E7%B2%BE%E8%AF%BB/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84/4.%20GNN.html","title":"4.【新神经网络范式】GNN/GCN 论文精读","lang":"zh-CN","frontmatter":{"description":"4.【新神经网络范式】GNN/GCN 论文精读 原文链接：Attention Is All You Need 源码链接：https://github.com/tensorflow/tensor2tensor","head":[["meta",{"property":"og:url","content":"https://lisenjie757.github.io/%E7%9F%A5%E8%AF%86%E5%BA%93/%E6%96%87%E7%8C%AE%E7%B2%BE%E8%AF%BB/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84/4.%20GNN.html"}],["meta",{"property":"og:site_name","content":"Li Senjie"}],["meta",{"property":"og:title","content":"4.【新神经网络范式】GNN/GCN 论文精读"}],["meta",{"property":"og:description","content":"4.【新神经网络范式】GNN/GCN 论文精读 原文链接：Attention Is All You Need 源码链接：https://github.com/tensorflow/tensor2tensor"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2023-09-11T06:11:36.000Z"}],["meta",{"property":"article:author","content":"李森杰"}],["meta",{"property":"article:modified_time","content":"2023-09-11T06:11:36.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"4.【新神经网络范式】GNN/GCN 论文精读\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2023-09-11T06:11:36.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"李森杰\\",\\"url\\":\\"https://lisenjie757.github.io\\"}]}"]]},"headers":[{"level":2,"title":"0. 核心总结","slug":"_0-核心总结","link":"#_0-核心总结","children":[]},{"level":2,"title":"1. 摘要","slug":"_1-摘要","link":"#_1-摘要","children":[]},{"level":2,"title":"2. 引言","slug":"_2-引言","link":"#_2-引言","children":[]},{"level":2,"title":"3. Transformer模型结构","slug":"_3-transformer模型结构","link":"#_3-transformer模型结构","children":[{"level":3,"title":"3.1 编码器和解码器块细节","slug":"_3-1-编码器和解码器块细节","link":"#_3-1-编码器和解码器块细节","children":[]},{"level":3,"title":"3.2 注意力机制细节","slug":"_3-2-注意力机制细节","link":"#_3-2-注意力机制细节","children":[]},{"level":3,"title":"3.3 逐点前馈网络细节","slug":"_3-3-逐点前馈网络细节","link":"#_3-3-逐点前馈网络细节","children":[]},{"level":3,"title":"3.4 位置编码细节","slug":"_3-4-位置编码细节","link":"#_3-4-位置编码细节","children":[]}]},{"level":2,"title":"4. 为什么使用自注意力","slug":"_4-为什么使用自注意力","link":"#_4-为什么使用自注意力","children":[]},{"level":2,"title":"5. 训练细节","slug":"_5-训练细节","link":"#_5-训练细节","children":[{"level":3,"title":"5.1 训练数据集和批大小","slug":"_5-1-训练数据集和批大小","link":"#_5-1-训练数据集和批大小","children":[]},{"level":3,"title":"5.2 训练硬件和训练计划","slug":"_5-2-训练硬件和训练计划","link":"#_5-2-训练硬件和训练计划","children":[]},{"level":3,"title":"5.3 优化器","slug":"_5-3-优化器","link":"#_5-3-优化器","children":[]},{"level":3,"title":"5.4 正则化","slug":"_5-4-正则化","link":"#_5-4-正则化","children":[]}]},{"level":2,"title":"6. 实验结果","slug":"_6-实验结果","link":"#_6-实验结果","children":[{"level":3,"title":"6.1 机器翻译结果","slug":"_6-1-机器翻译结果","link":"#_6-1-机器翻译结果","children":[]},{"level":3,"title":"6.2 模型参数分析","slug":"_6-2-模型参数分析","link":"#_6-2-模型参数分析","children":[]}]},{"level":2,"title":"7. 个人思考","slug":"_7-个人思考","link":"#_7-个人思考","children":[]}],"git":{"createdTime":1694412696000,"updatedTime":1694412696000,"contributors":[{"name":"lisenjie757","email":"1215954303@qq.com","commits":1}]},"readingTime":{"minutes":10.3,"words":3090},"filePathRelative":"知识库/文献精读/神经网络架构/4. GNN.md","localizedDate":"2023年9月11日","excerpt":"<h1> 4.【新神经网络范式】GNN/GCN 论文精读</h1>\\n<blockquote>\\n<p>原文链接：<a href=\\"https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">Attention Is All You Need</a>\\n源码链接：<a href=\\"https://github.com/tensorflow/tensor2tensor\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">https://github.com/tensorflow/tensor2tensor</a></p>\\n</blockquote>","autoDesc":true}');export{e as data};
