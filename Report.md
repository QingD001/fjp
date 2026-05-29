# Fake-Job-Posting 实验报告

## 摘要

本项目基于 Kaggle Fake Job Posting 的竞赛数据，完成虚假招聘信息的二分类识别。数据包含职位标题、地点、薪资范围、公司介绍、职位描述、岗位要求、福利、行业、职能、雇佣类型等字段，目标变量为 `fraudulent`，表示该招聘信息是否为虚假岗位。

我们先后试验 3 类模型：树模型、线性文本模型和文本深度学习模型。树模型包括 XGBoost、LightGBM、CatBoost 和 Random Forest；线性文本模型包括 Logistic Regression 和 Linear SVM；深度学习模型包括 TextCNN 和 BiLSTM。实验结果显示，文本深度学习模型在本任务上更有优势，TextCNN 在 Kaggle 返回的 Average Precision 指标上达到 0.941，是本次实验中表现最好的模型。

## 任务背景

网络招聘平台降低了求职成本，也让虚假招聘信息更容易传播。虚假岗位通常会利用看似正常的职位描述吸引求职者，再通过押金、培训费、个人信息收集等方式造成损失。因此，自动识别虚假招聘信息有实际应用价值。

本项目将问题建模为二分类任务。给定一条招聘信息，模型输出该信息为虚假岗位的概率。比赛采用 Average Precision（AP）作为评分指标，AP 更关注模型能否把真正的虚假招聘排在更靠前的位置。因此，提交文件中的 `fraudulent` 列使用概率分数，而不是 0/1 硬标签。

## 数据分析

训练集共有 12000 条样本，字段数为 18。目标变量分布不均衡，其中真实招聘 11411 条，虚假招聘 589 条，虚假样本比例约为 4.91%。因此指标不能只看准确率，如果模型学到全部预测为真实岗位也能得到很高的准确率，但没有实际识别价值。

数据中存在较多缺失值。`salary_range` 缺失率约 83.64%，`department` 缺失率约 63.97%，`required_education` 缺失率约 45.64%，`benefits` 缺失率约 40.12%。缺失本身也是信号，例如虚假招聘中 `company_profile` 缺失比例明显更高。因此，项目没有简单丢弃缺失字段，而是把缺失模式作为特征的一部分。

文本字段是本任务最重要的信息来源。`description`、`requirements`、`company_profile` 和 `benefits` 中包含大量岗位语义、公司背景、要求描述和福利描述。虚假招聘往往在这些文本中呈现模板化、承诺夸张、公司信息不足等特点。

## 特征工程

项目针对传统机器学习模型、线性文本模型和深度学习模型设计了不同特征。

### 传统机器学习特征

传统模型使用 `FeatureEngineering.py` 生成特征，主要包括：

- 文本统计特征：字符数、词数、平均词长、是否包含 URL、是否包含邮箱、是否包含 HTML 标记。
- 薪资特征：从 `salary_range` 中解析最低薪资、最高薪资、平均薪资，以及是否存在薪资信息。
- 二值字段：`telecommuting`、`has_company_logo`、`has_questions`。
- 类别字段独热编码：`employment_type`、`required_experience`、`required_education`。
- TF-IDF 文本特征：将 `company_profile`、`description`、`requirements`、`benefits` 拼接后提取 800 维 unigram 和 bigram 特征。

这套特征适合 XGBoost、LightGBM、CatBoost 和 Random Forest 等树模型。它能够保留一部分文本信息，也能利用结构化字段中的统计规律。

### 线性文本模型特征

线性模型和支持向量机更适合直接处理高维稀疏文本特征。本项目中，Logistic Regression 和 Linear SVM 复用了 `FeatureEngineering.py` 生成的统计特征、类别编码和 800 维 TF-IDF 特征，作为传统模型体系中的补充对照。

这类特征不强调复杂的非线性组合，而是依赖稀疏文本维度形成排序分数。对于样本量不大的文本分类任务，`TF-IDF + Logistic Regression` 和 `TF-IDF + Linear SVM` 往往是很有参考价值的基线。后续如果继续优化，可以加入字符级 n-gram，让模型捕捉特殊符号、邮箱、URL、大小写异常和拼写变体。

### 面向 CNN 的文本画像特征

由于 TextCNN 和 BiLSTM 直接处理 token 序列，项目新增 `feature4cnn.py`，将结构化字段转化为可被神经网络读取的文本 token。增强后的文本不仅包含原始职位描述，还包含字段标签和人工构造的风险信号。

构造后的文本包含：

- 职位标题 `title`。
- 地点拆分后的 `country`、`state`、`city`。
- 薪资 token，例如 `salary_missing`、`salary_present`、`salary_high`、`salary_zero_zero`。
- 公司 logo、是否有问题、是否远程办公等二值 token。
- 雇佣类型、经验要求、教育要求、行业、职能和部门。
- 字段缺失 token，例如 `company_profile_missing`、`benefits_missing`。
- 文本质量 token，例如文本过短、过长、大写比例高、数字比例高、包含邮箱或 URL。
- 风险短语 token，例如 `risk_work_from_home`、`risk_easy_money`、`risk_data_entry`、`risk_no_experience`。

这一步让 CNN 不再只依赖自然语言正文，而是同时接收结构化信息。对本任务来说，这比单纯拼接几段文本更合适，因为虚假招聘的风险往往同时体现在文本内容、字段缺失和结构化属性中。

## 算法与模型

### XGBoost

XGBoost 使用梯度提升树进行分类。模型设置 `max_depth=7`、`learning_rate=0.2`，评价指标为 logloss。它适合处理非线性特征组合，对 TF-IDF 和结构化特征都有较好的利用能力。

### LightGBM

LightGBM 同样是梯度提升树模型，训练速度较快。项目中设置 `max_depth=7`、`learning_rate=0.2`。LightGBM 在表格数据中通常表现稳定，但在本实验中分数低于 XGBoost 和 CatBoost。

### CatBoost

CatBoost 对类别特征友好，适合包含较多离散字段的任务。当前实现中类别字段已经被预处理成数值特征，因此 CatBoost 的类别建模优势没有完全发挥出来。模型设置 `depth=7`、`learning_rate=0.2`、`iterations=200`。

### Random Forest

Random Forest 使用 200 棵树，并设置 `class_weight='balanced'` 处理类别不平衡。它的优点是稳定、容易训练，但对高维稀疏文本特征的表达能力有限，因此最终效果弱于梯度提升树和神经网络模型。

### Logistic Regression

Logistic Regression 是文本分类中常用的线性基线。它可以直接接收 TF-IDF 稀疏矩阵，并输出概率分数，和 Kaggle 的 AP 指标比较匹配。由于本任务正负样本不均衡，实验中可以设置 `class_weight='balanced'`，让模型更重视虚假招聘样本。

这个模型的意义不只是追求最高分，也可以作为判断特征工程是否有效的参考。如果简单的 TF-IDF 线性模型已经取得较高 AP，说明数据中的文本关键词和短语模式非常强。

### Linear SVM

支持向量机适合高维稀疏文本特征。Linear SVM 的目标是找到区分正负样本的最大间隔超平面，通常在文本分类任务中表现稳定。由于普通 Linear SVM 默认输出类别而不是概率，若要用于 AP 评分，可以使用 `decision_function` 的连续分数，或者用 `CalibratedClassifierCV` 做概率校准。

Linear SVM 可以作为 Logistic Regression 的互补对照。两者都使用 TF-IDF，但优化目标不同，有助于判断当前任务更适合概率线性模型还是间隔分类模型。

### TextCNN

TextCNN 是本项目表现最好的模型。模型先将 token 映射到 embedding，再使用不同大小的卷积核提取局部文本模式，经过最大池化和全连接层，得出欺诈概率。

本任务中，虚假招聘文本常常存在局部短语模式，例如 “work from home”“data entry”“no experience required” 等。CNN 的卷积结构正好适合捕捉这种局部 n-gram 语义。加入 `feature4cnn.py` 后，CNN 还可以学习结构化 token 与文本 token 的组合关系。

### BiLSTM

BiLSTM 使用双向 LSTM 建模序列信息。相比 CNN，它更关注 token 的前后顺序和长距离依赖。理论上 BiLSTM 能处理更复杂的上下文，但本数据集规模不大，训练样本只有 12000 条，且正类样本较少。从实验结果看，BiLSTM 没有超过 TextCNN。

## 实验结果

Kaggle 返回的 Average Precision 结果如下：

| 模型 | 文件 | Kaggle AP | Rank |
| --- | --- | ---: | ---: |
| XGBoost | `model.py` | 0.869 | 2 |
| LightGBM | `model2.py` | 0.808 | 5 |
| CatBoost | `model3.py` | 0.817 | 4 |
| Random Forest | `model4.py` | 0.658 | 7 |
| TextCNN | `model5.py` | 0.941 | 1 |
| BiLSTM | `model6.py` | 0.851 | 3 |
| Logistic Regression | `model7.py` | 0.673 | 6 |
| Linear SVM | `model8.py` | 0.635 | 8 |

从结果看，TextCNN 明显领先，AP 达到 0.941。XGBoost 是传统机器学习模型中效果最好的，AP 为 0.869。BiLSTM 达到 0.851，说明深度学习模型确实能从文本中学习到有效模式，但 LSTM 在本任务上不如 CNN 稳定。

Logistic Regression 的 AP 为 0.673，略高于 Random Forest；Linear SVM 的 AP 为 0.635，是本次实验中最低的结果。这个结果说明，当前 800 维 TF-IDF 加统计特征不足以支撑线性模型取得高分。相比之下，XGBoost 能通过非线性树结构组合特征，因此在传统机器学习模型中表现更好。

Random Forest 的 AP 为 0.658，说明普通随机森林对高维文本特征和类别不平衡的适应能力不足。LightGBM 和 CatBoost 的表现接近，但都低于 XGBoost。

## 结果分析

TextCNN 表现最好，主要有两个原因。

第一，虚假招聘信息具有明显的局部文本模式。很多虚假岗位会反复出现短语、固定描述和夸张承诺。CNN 通过多个卷积核扫描 token 序列，能够直接捕捉这些局部组合。

第二，增强后的 CNN 特征把结构化字段转成了文本 token。模型不仅能看到职位描述，还能看到 `salary_missing`、`company_profile_missing`、`no_logo_no_questions`、`industry_oil_energy` 等信息。这些 token 和正文一起进入网络，使 CNN 具备了处理混合数据的能力。

XGBoost 的效果也较好，说明传统特征工程仍然有价值。它能够利用文本统计、薪资解析、类别编码和 TF-IDF 特征，在小样本场景下保持稳定。相比之下，LightGBM 和 CatBoost 可能需要更细致的参数调优，CatBoost 如果直接使用原始类别字段，可能还有提升空间。

线性模型和支持向量机虽然结构简单，但在本项目中仍然值得保留。它们回答了一个关键问题：模型分数到底来自复杂模型本身，还是来自 TF-IDF 特征已经足够强。本次实验中，Logistic Regression 和 Linear SVM 明显落后于 XGBoost、BiLSTM 和 TextCNN，说明简单线性边界不足以刻画虚假招聘信息的复杂模式。CNN 的优势因此更有说服力，它不仅利用正文，还利用了字段 token、缺失模式和局部短语组合。

BiLSTM 没有超过 CNN，原因可能是训练数据规模偏小。LSTM 对序列建模能力更强，但参数更多，训练时间更长。在没有预训练词向量或 Transformer 表示的情况下，它不一定比 CNN 更适合这个任务。

## 项目实现

项目入口为 `src/main.py`。通过修改 `MODEL` 变量，可以切换不同模型：

```python
MODEL = 'cnn'
```

非深度学习模型会调用 `feature_engineering` 生成结构化特征和 TF-IDF 特征。深度学习模型会调用 `add_cnn_text_col` 生成增强文本，再交给 CNN 或 LSTM 训练。

模型训练后会输出验证指标，并生成 `submission.csv`。由于 Kaggle 使用 AP 评分，提交文件中保存的是预测概率：

```python
sub['fraudulent'] = test_prob
```

这样可以保留样本之间的排序信息，比提交 0/1 硬标签更适合 AP 指标。

## 问题与改进方向

当前项目已经得到较好的结果，但还有几处可以继续改进。

1. 可以继续优化 Logistic Regression 和 Linear SVM 的文本表示。目前它们复用了传统特征工程中的 800 维 TF-IDF，信息量偏少。后续可以加入字符级 n-gram、扩大 TF-IDF 维度，并分别对标题、公司介绍、职位描述和岗位要求建模。

2. 可以对 TextCNN 做更系统的调参，例如调整 embedding 维度、卷积核大小、dropout、batch size 和训练轮数。当前模型结构较简单，仍有提升空间。

3. 可以尝试 CNN 与传统模型融合。XGBoost 和 TextCNN 的建模方式不同，前者更依赖表格特征和 TF-IDF，后者更依赖局部文本模式。将多个模型的预测概率做加权平均，可能进一步提高 AP。

4. 可以加入预训练语言模型，例如 DistilBERT 或 DeBERTa-small。它们能提供更强的文本语义表示，但训练成本更高，也需要更好的显卡环境。

5. 可以优化类别特征处理。对 `industry`、`function`、`location` 等高基数字段使用目标编码或频率编码，可能提高树模型表现。

## 总结

本项目完成了虚假招聘信息识别的完整机器学习流程，包括数据分析、特征工程、模型训练、交叉验证、Kaggle 提交和结果对比。实验表明，TextCNN 最适合当前任务，在 Kaggle 上取得 0.941 的 AP 分数。
