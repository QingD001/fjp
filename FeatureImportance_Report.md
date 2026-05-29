# 特征重要性分析报告

## 分析目的

本报告用于解释虚假招聘识别模型中哪些特征对预测结果贡献更大。主实验已经比较了 8 个模型的 Kaggle AP 分数，其中 TextCNN 表现最好，AP 为 0.941。为了进一步理解模型为什么有效，本报告以 XGBoost 为代表，对传统特征工程生成的特征进行重要性分析。

XGBoost 适合做这类分析，因为它可以输出每个特征在树分裂中的 gain。gain 越高，说明该特征在降低分类损失时贡献越大。需要注意的是，gain 反映的是树模型中的使用价值，不等同于因果关系，也不一定完全代表 CNN 的内部判断方式。

## 分析方法

本次分析使用 `FeatureEngineering.py` 生成特征，再训练 `model.py` 中的 XGBoost 模型。特征总数为 859 维，包括：

- 文本统计特征：字符数、词数、平均词长、邮箱、URL、HTML 标记等。
- 薪资特征：最低薪资、最高薪资、平均薪资、是否填写薪资。
- 二值属性：是否远程、是否有公司 logo、是否有筛选问题。
- 类别特征：雇佣类型、经验要求、教育要求。
- TF-IDF 文本特征：从 `company_profile`、`description`、`requirements`、`benefits` 拼接文本中提取 800 维 unigram 和 bigram。

为了提高可读性，报告中将 `tfidf_编号` 反查为真实词项，例如 `tfidf_512` 对应词项 `positions`。

## 按特征组汇总

从 gain 总量看，文本特征占据绝对主导地位。

| 特征组 | Gain 占比 |
| --- | ---: |
| TF-IDF 文本词项 | 85.59% |
| 文本统计特征 | 8.60% |
| 类别特征 | 3.30% |
| 二值属性 | 1.65% |
| 薪资特征 | 0.86% |

这个结果说明，虚假招聘识别首先是一个文本识别问题。职位描述、公司介绍、岗位要求和福利文本中的词项，对模型排序贡献最大。结构化字段也有价值，但更多起辅助作用。

文本统计特征排在第二位，说明“文本有没有、写得长不长、是否包含邮箱”等非语义信息也很关键。虚假招聘往往会出现公司介绍缺失、描述模板化、联系方式异常等特征。

## Top 特征分析

XGBoost gain 排名前 20 的特征如下：

| 排名 | 特征 | 含义 | Gain |
| ---: | --- | --- | ---: |
| 1 | `tfidf:positions` | 文本词项 positions | 158.405 |
| 2 | `tfidf:values` | 文本词项 values | 137.574 |
| 3 | `cp_char_count` | 公司介绍字符数 | 136.576 |
| 4 | `tfidf:compensation` | 文本词项 compensation | 96.406 |
| 5 | `desc_email_placeholder` | 职位描述中出现邮箱占位 | 33.323 |
| 6 | `tfidf:safety` | 文本词项 safety | 33.007 |
| 7 | `tfidf:000` | 文本词项 000 | 30.648 |
| 8 | `required_education_Bachelor's Degree` | 学历要求为本科 | 28.976 |
| 9 | `tfidf:professional` | 文本词项 professional | 28.630 |
| 10 | `tfidf:access` | 文本词项 access | 27.706 |
| 11 | `tfidf:with our` | 文本短语 with our | 26.953 |
| 12 | `tfidf:time` | 文本词项 time | 26.686 |
| 13 | `tfidf:must have` | 文本短语 must have | 25.673 |
| 14 | `tfidf:member` | 文本词项 member | 24.841 |
| 15 | `required_experience_Mid-Senior level` | 经验要求为中高级 | 22.612 |
| 16 | `has_questions` | 是否包含筛选问题 | 21.790 |
| 17 | `tfidf:word` | 文本词项 word | 20.989 |
| 18 | `tfidf:industry` | 文本词项 industry | 20.126 |
| 19 | `tfidf:real` | 文本词项 real | 19.098 |
| 20 | `tfidf:web` | 文本词项 web | 18.927 |

排名靠前的特征可以分成几类。

第一类是文本词项，例如 `positions`、`values`、`compensation`、`professional`、`must have`。这些词项不一定单独代表虚假岗位，但它们和其他特征组合后能帮助模型区分真实招聘与虚假招聘。

第二类是公司介绍长度。`cp_char_count` 排名第 3，说明公司介绍是否充分很重要。结合数据分析，虚假招聘中 `company_profile` 缺失比例明显更高，因此公司介绍长度能成为强信号。

第三类是联系方式异常。`desc_email_placeholder` 排名第 5，说明职位描述中出现邮箱相关信息会显著影响模型判断。正常招聘通常会通过平台流程收集简历，而虚假招聘更可能直接引导求职者联系邮箱或外部渠道。

第四类是职位属性。`required_education_Bachelor's Degree`、`required_experience_Mid-Senior level` 和 `has_questions` 都进入了前 20。它们说明岗位要求和招聘流程完整度也有区分作用。

## 对不同特征的理解

### 文本词项是主力

TF-IDF 词项贡献了 85.59% 的 gain。这和最终实验结果一致：TextCNN 能取得最高 AP，说明文本局部模式非常强。虚假招聘并不只体现在某个单独字段，而是体现在多个词项、短语和描述方式的组合中。

这也解释了为什么单纯结构化模型难以超过 TextCNN。树模型可以利用 TF-IDF，但它对文本局部顺序的理解有限；CNN 通过卷积核扫描 token 序列，可以更自然地学习短语级模式。

### 公司介绍质量很关键

`cp_char_count` 和 `cp_word_count` 都进入重要特征列表。公司介绍越完整，通常越接近真实招聘；公司介绍缺失或过短，则风险更高。

这类特征不是语义特征，但非常实用。它反映的是招聘信息的完整度。虚假招聘经常缺少可信的公司背景，而真实企业通常会填写较完整的公司介绍。

### 邮箱和外部联系方式有风险信号

`desc_email_placeholder` 排名较高，说明联系方式类信息值得重点关注。招聘平台中的真实岗位通常不需要在描述正文里反复放邮箱。若岗位描述直接引导求职者通过邮箱、URL 或外部方式联系，可能存在更高风险。

因此，后续特征工程可以继续强化联系方式特征，例如统计邮箱数量、URL 数量、是否包含非平台联系方式、是否出现即时通信账号等。

### 学历、经验和筛选问题是辅助信号

学历要求、经验要求和 `has_questions` 的重要性不如文本词项高，但仍然有贡献。真实招聘通常对经验、学历、技能和筛选问题有较清晰要求。虚假招聘可能更倾向于“门槛低、回报高”的描述。

这些特征单独使用时不一定强，但和文本特征组合后能提高排序能力。

## 对 CNN 特征工程的启发

本次重要性分析虽然基于 XGBoost，但对 TextCNN 的特征工程有直接启发。

第一，CNN 应该继续强化文本输入。TF-IDF 词项贡献最高，说明文本内容是主战场。当前 `feature4cnn.py` 已经将标题、公司介绍、职位描述、岗位要求和福利全部纳入输入，这是合理的。

第二，公司介绍缺失和长度应被显式 token 化。当前 CNN 特征中已经加入 `company_profile_missing`、`company_profile_present`、`company_profile_very_short`、`company_profile_very_long` 等 token，和重要性结果一致。

第三，邮箱、URL、HTML 等格式特征应该保留。当前 CNN 特征中已经加入 `email_token`、`url_token`、`html_token`，并生成字段级 token，例如 `description_has_email`。这些特征对虚假招聘识别有实际价值。

第四，结构化字段可以继续文本化。`has_questions`、学历和经验要求在 XGBoost 中有贡献，因此把它们变成 `has_questions_yes`、`required_education_bachelor_s_degree`、`required_experience_mid_senior_level` 这类 token 是有效方向。

## 当前特征工程的不足

当前传统特征工程仍有几处不足。

第一，TF-IDF 特征维度只有 800 维，信息压缩较强。对于 Logistic Regression 和 Linear SVM，这可能不够，因此两个线性模型的 AP 分别只有 0.673 和 0.635。后续可以扩大到 3000 到 10000 维，并加入字符级 n-gram。

第二，传统模型没有充分使用 `title`、`location`、`industry`、`function` 和 `department` 等字段。CNN 版本已经通过 `feature4cnn.py` 使用了这些字段，但树模型和线性模型还有补强空间。

第三，当前 TF-IDF 是把多个文本字段拼接后统一建模。不同字段的信息密度不同，职位标题、公司介绍和职位描述不应完全等价。后续可以分别为不同字段构造 TF-IDF，再拼接到模型中。

第四，类别特征只对少数字段做了独热编码。`industry`、`function` 和 `location` 这些字段可能包含较强信号，可以尝试频率编码、目标编码或 CatBoost 原生类别特征处理。

## 结论

特征重要性分析表明，虚假招聘识别最依赖文本词项，TF-IDF 文本特征贡献了 85.59% 的 XGBoost gain。公司介绍长度、邮箱占位、学历要求、经验要求和筛选问题也有明显价值。

这个结果和模型实验相互印证。TextCNN 之所以取得 0.941 的最高 AP，主要是因为它更适合捕捉文本中的局部短语模式，同时 `feature4cnn.py` 又把结构化字段、缺失模式和风险信号转成了可学习的 token。相比之下，线性模型和 Random Forest 表现较弱，说明简单特征组合不足以充分刻画虚假招聘信息。

后续优化应围绕两条线展开：一是继续增强 CNN 的字段 token 和风险短语；二是扩展传统文本特征，加入更高维 TF-IDF、字符级 n-gram 和字段级 TF-IDF。这样可以让模型既保留深度学习的表达能力，也保留传统模型的可解释性。
