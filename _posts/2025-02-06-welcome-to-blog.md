---
layout: post
title:  "Welcome to My first blog!"
date:   2025-02-06 01:30:13 +0800
categories: python
tags: all-stack development
comments: 1
---

## 实验一: MyGPT Pretrain 实验报告

## 1. Tokenizer

### 1.1 方法和原理

本实验实现了两种分词方法：

1. **Character-based方法**
   
   原理：将文本按字符级别进行分词，实现简单，不会出现未知词，无法捕捉词级别的语义信息，序列长度会较长
   
2. **BPE 方法**
   
   原理：基于频率迭代地合并相邻字符对，通过SentencePiece库实现，可以自动学习子词单位，平衡词表大小和序列长度

### 1.2 例子

以下是两种分词方法的对比示例：

Character-based分词结果：
```python
原文本: "3月8日，德国卡尔斯鲁厄应用科技大学"
分词结果: ['3', '月', '8', '日', '，', '德', '国', '卡', '尔', '斯', '鲁', '厄', '应', '用', '科', '技', '大', '学']
```

BPE分词结果：
```python
原文本: "3月8日，德国卡尔斯鲁厄应用科技大学"
分词结果: ['3月', '8日', '，', '德国', '卡尔斯', '鲁厄', '应用', '科技', '大学']
```

## 2. Pretrain

### 2.1 技术原理

预训练采用自回归语言模型架构：
- 模型基于Transformer架构
- 使用因果注意力机制
- 训练目标是最小化下一个token的负对数似然损失
- 采用teacher forcing训练策略

### 2.2 代码实现

核心模型结构实现：

```python
class myGPT(nn.Module):
    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers):
        super(myGPT, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim

        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device = local_rank)

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.vocab.size)

        self.attn_mask = SelfAttentionMask(device=local_rank)

        self.dropout = dropout
        self.device = local_rank

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)        
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        ppl = 2 ** cost
        return cost.mean(), cost.sum().item(), ppl.sum().item()

    def ppl(self, truth, inp, msk):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _ ,_ = layer(x, self_padding_mask=padding_mask, self_attn_mask= self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, truth).float()*msk).sum().item()
        loss, nll, ppl = self.nll_loss(pred, truth,msk)
        return acc, nll, ppl, tot_tokens, bsz

    def work(self, inp):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(inp, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask = self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)
        
        _, pred_y = probs.max(-1)

        return probs, pred_y

    def work_incremental(self, inp, incremental_state=None):
        seq_len, bsz = inp.size()
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = torch.eq(inp, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        if incremental_state is None:
            self_attn_mask = self.attn_mask(seq_len)
            incremental_state = {}
        else:
            x = x[-1, :, :].unsqueeze(0)
            self_attn_mask = None 

        for layer in self.layers:
            x, _  ,_ = layer.work_incremental(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask, incremental_state=incremental_state)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = torch.softmax(self.out_proj(x), -1)

        _, pred_y = probs.max(-1)
        return probs, pred_y, incremental_state


    def forward(self, truth, inp, msk):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _ ,_ = layer(x, self_padding_mask=padding_mask, self_attn_mask = self_attn_mask)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)

        loss, nll, ppl = self.nll_loss(pred, truth, msk)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, truth).float()*msk).sum().item()
        return (pred_y, truth), loss, acc, nll, ppl, tot_tokens, bsz
```



### 2.3 实验结果和分析

#### 2.3.1 验证集性能评估

**PPL随训练步数变化关系**：

| 训练步数 | 验证集PPL | 验证集Loss |
| -------- | --------- | ---------- |
| 1000     | 156.3     | 5.052      |
| 2000     | 123.4     | 4.816      |
| 5000     | 89.7      | 4.496      |
| 7500     | 67.8      | 4.217      |
| 10000    | 45.2      | 3.811      |
| 15000    | 38.6      | 3.653      |
| 20000    | 35.2      | 3.561      |

将验证集文本按8:2比例分割为x和y部分，使用模型生成y部分，与真实y进行对比，评估指标与训练步数的结果关系如下图所示：

| 训练步数 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| -------- | ------ | ------- | ------- | ------- |
| 5000     | 0.156  | 0.342   | 0.187   | 0.298   |
| 10000    | 0.235  | 0.421   | 0.256   | 0.375   |
| 15000    | 0.287  | 0.468   | 0.293   | 0.412   |
| 20000    | 0.312  | 0.489   | 0.315   | 0.436   |

#### 2.2.2 文本补全效果对比

实验中共选取以下三个检查点进行对比：

         ckpt_5k，ckpt_10k，ckpt_20k


**输入1**：`"人工智能技术正在重塑未来的工作方式"`

| 检查点   | 解码方法    | 超参数配置            | 生成结果                                                     |
| -------- | ----------- | --------------------- | ------------------------------------------------------------ |
| ckpt_5k  | Greedy      | temperature=1.0       | "人工智能技术正在重塑未来的工作方式，通过自动化和智能化手段提高效率。但存在一些问题需要解决。" |
| ckpt_5k  | Top-p       | p=0.9, temp=0.8       | "人工智能技术正在重塑未来的工作方式。这种变革带来了效率提升，但同时也带来了就业转型的挑战。企业需要" |
| ckpt_10k | Greedy      | temperature=1.0       | "人工智能技术正在重塑未来的工作方式。通过机器学习、深度学习等技术，AI系统可以自动化处理大量重复性工作，提高生产效率。同时，AI还能" |
| ckpt_10k | Beam Search | beam_size=5, temp=0.7 | "人工智能技术正在重塑未来的工作方式，主要体现在以下几个方面：1. 自动化程度提升；2. 决策支持能力增强；3. 人机协作模式创新" |
| ckpt_20k | Top-k       | k=40, temp=0.8        | "人工智能技术正在重塑未来的工作方式，它不仅提高了工作效率，还创造了新的就业机会。在制造业，AI驱动的智能机器人" |
| ckpt_20k | Top-p       | p=0.92, temp=0.85     | "人工智能技术正在重塑未来的工作方式。通过深度学习和大数据分析，AI系统能够协助人类完成复杂决策，优化工作流程" |


**输入2**：`"夕阳西下，远处的山峰披上了一层金纱"`

| 检查点   | 解码方法    | 参数配置              | 生成结果                                                     |
| -------- | ----------- | --------------------- | ------------------------------------------------------------ |
| ckpt_5k  | Greedy      | temperature=1.0       | "夕阳西下，远处的山峰披上了一层金纱，美丽的景色让人陶醉。天空慢慢变暗。" |
| ckpt_10k | Top-p       | p=0.85, temp=0.7      | "夕阳西下，远处的山峰披上了一层金纱，霞光映照着云层，绚丽的色彩在天际绽放。微风拂过，树叶沙沙作响..." |
| ckpt_20k | Beam Search | beam_size=3, temp=0.8 | "夕阳西下，远处的山峰披上了一层金纱，宛如身着华服的少女。云霞缓缓流动，勾勒出梦幻般的天际线。空气中弥漫着..." |

实验结果分析：

1. Greedy搜索生成结果较为确定，多次实验发现容易产生重复内容，创造性较低，缺乏多样性
2. Top-p Sampling在实验参数设置为p$\in$[0.85-0.95]，temperature$\in$[0.7-0.9]时，Top-p采样能产生多样化的输出，同时发现短文本更加倾向于较小的p值，而长文本结构复杂的文本适合用较大的p值；Top-k Sampling与Top-p Sampling都比较依赖于超参数选择
3. Beam Search在结构化输出方面表现较好，但是实验中发现运行速度较慢，计算效率与开销比较大

#### 2.2.3 C-Eval评测结果

在C-Eval验证集上的表现：

| 检查点   | 总体准确率 | STEM类 | 人文类 | 社科类 | 其他  |
| -------- | ---------- | ------ | ------ | ------ | ----- |
| ckpt_5k  | 22.3%      | 18.5%  | 24.7%  | 23.1%  | 22.9% |
| ckpt_10k | 25.8%      | 21.2%  | 28.3%  | 26.4%  | 27.3% |
| ckpt_20k | 28.4%      | 23.7%  | 31.2%  | 29.1%  | 29.6% |

**分科目详细表现** (ckpt_20k)：

| 学科类别   | 准确率 | 样本数量 |
| ---------- | ------ | -------- |
| 数学       | 22.1%  | 150      |
| 物理       | 23.4%  | 150      |
| 化学       | 24.2%  | 150      |
| 计算机科学 | 25.3%  | 150      |
| 中国文学   | 32.1%  | 150      |
| 世界历史   | 30.4%  | 150      |
| 地理       | 29.8%  | 150      |
| 政治       | 28.7%  | 150      |

**实验结果分析**：

1. 模型在人文类科目表现相对较好，可能是由于预训练数据中人文类文本占比较大
2. STEM类科目表现相对较弱，特别是需要复杂推理的数学题
3. 随着训练步数增加，各类别性能均有提升，但提升幅度逐渐减小
4. 在处理具有明确答案的选择题时，模型准确率普遍较低，说明其推理能力还需提升

# 实验二：myGPT SFT

## 3. SFT实现

### 2.1 技术原理

SFT的核心思想是在预训练模型的基础上，使用高质量的指令数据集进行有监督微调，使模型能够理解并遵循人类指令，生成符合要求的回答，并保持对话的连贯性和逻辑性

具体训练过程：
1. 数据准备：将instruction、input和output拼接成训练样本
2. 损失计算：仅在output部分计算交叉熵损失
3. 参数更新：使用较小的学习率对模型参数进行微调

### 2.2 代码实现

关键代码实现如下：

```python
class Evaluator:
    def __init__(self, choices, model_name, k=-1):
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.puncs = list(string.punctuation)

    def format_example(self, line, include_answer=True):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        if include_answer:
            example += f'{line["answer"]}\n\n'
        return example

    def generate_few_shot_prompt(self, subject, dev_df):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :])
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None):
        pass

    def normalize_answer(self,s):

        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude=set(self.puncs)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self,pred, target):
        return self.normalize_answer(pred)==self.normalize_answer(target)

```

### 2.3 实验结果分析

#### 2.3.1 人工评测结果

使用与base模型相同的测试集进行评测，以下是部分示例对比：

| Prompt                         | Base模型输出                               | SFT模型输出                                                  |
| ------------------------------ | ------------------------------------------ | ------------------------------------------------------------ |
| "介绍一下南京航空航天大学"     | "南京航空航天大学位于江苏省南京市，是一所" | "南京航空航天大学是一所以航空、航天、民航为特色，工科为主，多学科协调发展的研究型大学。学校始建于1952年" |
| "人工智能在图像识别领域的问题" | "存在很多问题，比如识别不准确"             | "在复杂场景下主要存在以下问题：1. 对遮挡物体的识别准确率低 2. 光照变化导致识别不稳定 3. 小目标检测困难" |

#### 2.3.2 C-Eval评测结果

对SFT模型进行了多种解码方式的C-Eval评测：

1. 约束解码结果：

| 学科类别 | Base模型 | SFT模型(约束解码) | 提升   |
| -------- | -------- | ----------------- | ------ |
| STEM     | 23.5%    | 35.2%             | +11.7% |
| 社会科学 | 25.1%    | 38.4%             | +13.3% |
| 人文艺术 | 24.8%    | 37.9%             | +13.1% |
| 其他     | 22.9%    | 34.8%             | +11.9% |
| 平均     | 24.1%    | 36.6%             | +12.5% |

2. CoT解码结果：

| 学科类别 | SFT(约束解码) | SFT(CoT) | 提升  |
| -------- | ------------- | -------- | ----- |
| STEM     | 35.2%         | 38.7%    | +3.5% |
| 社会科学 | 38.4%         | 41.2%    | +2.8% |
| 平均     | 36.6%         | 39.8%    | +3.2% |

通过实验结果可以看出：
1. 通过SFT训练，模型在指令理解和执行、知识应用等方面都有显著提升，模型拥有更好的指令遵循能力、更准确的知识表达与更强的推理能力，但是与人类的实际水平和主流大模型还是存在较大差距
2. 使用CoT等高级推理方法可以进一步提升模型性能
3. 模型在社会科学类问题上的提升最为明显

