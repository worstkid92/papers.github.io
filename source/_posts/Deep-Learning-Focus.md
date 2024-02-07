---
title: Deep Learning Focus
date: 2024-02-07 15:37:20
tags:
---
# quoted, url: 
https://cameronrwolfe.substack.com/p/language-model-training-and-inference
https://cameronrwolfe.substack.com/p/graph-based-prompting-and-reasoning#%C2%A7the-transformer-from-top-to-bottom
https://cameronrwolfe.substack.com/p/chain-of-thought-prompting-for-llms
https://cameronrwolfe.substack.com/p/tree-of-thoughts-prompting
# Language Model Scaling Laws
url:https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt#%C2%A7language-models-are-few-shot-learners
https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2#%C2%A7prerequisites-for-gpt
![pre-trainning](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F89687ad1-ab5d-4c72-840c-343d7fa26ab2_1854x1030.png)
GPT 模型是利用语言建模目标，通过未标注文本数据的语料库/数据集进行预训练的。简单地说，这意味着我们通过以下方式来训练模型：(i) 从数据集中抽取一些文本；(ii) 训练模型预测下一个单词；见上图。这种预训练过程是一种自我监督学习，因为只需查看数据集中的下一个单词，就能确定正确的 "下一个 "单词。
## 数学中的语言建模。
要理解语言建模，我们只需掌握上述基本概念。不过，为了使这一点更加严谨，我们可以注意到，我们的语料库只是一组标记。我们可以将标记视为数据集中的单个单词，但这并不完全正确。实际上，标记可能是子词，甚至是字符；
有一组tokens
![语料库](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F45501723-a132-40e7-8cb8-5050b2b265fb_1328x378.png)
![modeling loss](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.
com%2Fpublic%2Fimages%2F3430b67c-2d19-4840-9207-09e68a25d03a_1318x444.png)
### why use log
https://chrispiech.github.io/probabilityForComputerScientists/en/part1/log_probabilities/
https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/bayes-rules-conditional-probability-chain-rule/tutorial/
利用语言建模损失（它只是表征我们的模型准确预测序列中下一个标记的能力！），我们可以按照下面的步骤预训练模型参数 θ，从而使损失最小：
1.预训练语料库中的样本文本
2.用我们的模型预测下一个标记
3.使用随机梯度下降（SGD）或其他优化器，提高下一个标记的正确概率
通过多次重复这种（自我监督）训练过程，我们的模型最终会成为真正的语言建模高手（即预测序列中的下一个标记）。
### 什么是语言模型？
使用这种自监督语言建模目标预先训练的模型通常被称为语言模型（LM）。LM 随着规模的扩大（即层数和参数的增加等）而变得更加有效。因此，我们经常会看到这些模型的大型版本（如 GPT-3），它们被称为大型语言模型 (LLM)。
### 为什么 LMs 有用？
LM 可以通过迭代预测最有可能出现的下一个标记来生成连贯的文本，这使得从文本自动完成到聊天机器人等一系列应用成为可能。不过，除了生成能力之外，NLP 领域的前期工作已经表明，LM 预训练对各种任务都有极大的帮助；例如，预训练的词嵌入在下游任务中非常有用，LM预训练可以提高 LSTM 的性能。
在这些方法之外，GPT 模型探索了使用转换器进行语言模型预训练的方法。与顺序模型（如 LSTM）相比，变换器（i）具有令人难以置信的表现力（即高表示能力、多参数等），（ii）更适合现代 GPU 的并行计算能力，允许使用更大的模型和更多的数据进行 LM 预训练。这种可扩展性使 LLM 的探索成为可能，而 LLM 已经彻底改变了 NLP 的应用。
(tranformers:https://cameronrwolfe.substack.com/p/vision-transformers#%C2%A7background)
![transformer](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F57f3c98e-39d1-4eda-9a53-309210d42f49_662x968.png)
## 纯解码transformer
GPT 和 GPT-2 都使用纯解码器变压器架构。上面博客链接里有。
![decoder-only transformers](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)
decoder-only transformers移除了以下组件：
整个编码器模块
解码器中的所有编码器-解码器自注意模块
移除这些组件后，解码器的每一层都由一个屏蔽自注意层和一个前馈神经网络组成。将几个这样的层堆叠在一起，就形成了一个深度解码器专用变压器架构，例如用于 GPT 或 GPT-2 的架构；
### 为什么使用解码器？
选择使用解码器架构（而不是编码器）来处理 LM，因为解码器中的屏蔽的self-attention layers确保了模型在制作标记表示时不能在序列中向前看。与此相反，self-attention（在编码器中使用）允许根据序列中的所有其他标记来调整每个标记的表示。因为在预测下一个标记时，我们不应该向前看句子。使用屏蔽的self-attention layers会产生一种自回归结构（即模型在时间 t 的输出被用作时间 t+1 的输入），它可以持续预测序列中的下一个标记。

# chain of thought
![pic.1 chain of thought](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2be86a82-3d94-444c-90a2-9428ff629b2f_1994x1404.png)

# transformer
![pic.1 总览](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F553be3b4-3c80-435d-88c5-c7079bff9cbb_1940x1090.png)
