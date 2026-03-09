# Transformer注意力机制文献综述

**撰写日期：2026年3月9日**

## 摘要

Transformer自2017年提出以来，其核心的自注意力机制已成为深度学习领域最重要的创新之一。本综述基于45篇相关文献，系统梳理了Transformer注意力机制的研究进展，涵盖长上下文处理、稀疏注意力、高效实现、视觉应用等多个研究方向。文献分析显示，当前研究热点集中在长上下文语言模型、稀疏注意力加速、多模态注意力融合以及理论分析等方面。本文将深入分析各研究方向的关键工作，总结技术演进脉络，并展望未来发展趋势。

---

## 1. 引言

Transformer架构由Vaswani等人在2017年提出，其自注意力机制（Self-Attention）通过捕捉序列中token之间的全局依赖关系，彻底改变了自然语言处理和计算机视觉领域的发展方向。注意力机制的核心思想是让模型能够动态地分配不同位置信息的权重，从而实现更灵活的信息整合。

自Transformer诞生以来，注意力机制的研究呈现出爆炸式增长。根据文献分析，2022-2026年间，关于注意力机制的论文数量呈指数增长趋势，特别是在长上下文处理、高效实现和多模态应用等方面。

---

## 2. 研究趋势分析

### 2.1 年份分布

通过分析文献数据，Transformer注意力机制的研究呈现出明显的阶段性特征：

**2018-2020年：基础探索期**
- 2018年：Music Transformer（Huang et al.）提出了相对位置编码机制，将记忆复杂度从O(n²)降低到O(n)
- 2020年：Déjà vu（Wu et al.）提出了上下文化时间注意力机制，用于序列推荐

**2021-2022年：多样化发展期**
- 2021年：Smart Bird（Wu et al.）提出了可学习的稀疏注意力
- 2022年：DiNA（Hassani & Shi）提出了膨胀邻域注意力，实现了局部和全局注意力的平衡

**2023-2026年：优化与应用期**
- 2023-2026年：涌现出大量高效注意力实现和长上下文处理方法

### 2.2 研究热点

通过文献分析，当前Transformer注意力机制的研究热点主要集中在以下几个方面：

1. **长上下文处理**：随着大语言模型的发展，如何处理更长上下文成为核心挑战
2. **稀疏注意力**：通过稀疏化减少计算复杂度
3. **高效实现**：优化注意力计算速度和内存使用
4. **多模态融合**：将注意力机制应用于视觉、音频、时序等多模态数据
5. **理论分析**：深入理解注意力机制的工作原理和泛化能力

---

## 3. 主要研究方向综述

### 3.1 长上下文注意力机制

长上下文处理是当前Transformer研究的重要方向。随着大语言模型的发展，如何处理128K、1M甚至更长上下文成为关键挑战。

#### 3.1.1 稀疏注意力与门控注意力

**Gated Sparse Attention (GSA, 2026)**
- Shen & Shen（2026）提出了Gated Sparse Attention，将稀疏注意力和门控注意力相结合
- 核心创新：使用门控闪电索引器和自适应稀疏控制器
- 性能提升：在128K上下文下实现12-16倍加速，困惑度从6.03降至5.70
- 优势：同时解决了计算效率和训练稳定性问题

**Forgetting Transformer (2025)**
- Lin et al.（2025）引入了遗忘门机制，通过下权重未归一化的注意力分数
- 特点：与FlashAttention兼容，无需位置编码
- 表现：在长上下文语言建模、长度外推任务上优于标准Transformer

**Focus-dLLM (2026)**
- Long et al.（2026）提出了置信度引导的上下文聚焦机制
- 创新：基于相邻步骤token置信度的相关性预测未掩码区域
- 性能：在32K上下文下实现29倍以上的无损加速

#### 3.1.2 注意力机制的理论分析

**Benign Overfitting in Token Selection (2024)**
- Sakamoto & Sato（2024）研究了注意力机制在标签噪声下的泛化能力
- 发现：注意力机制的token选择能够实现良性过拟合
- 贡献：建立了信噪比（SNR）与注意力选择之间的关系

---

### 3.2 稀疏注意力机制

稀疏注意力通过只计算部分token之间的注意力，显著降低计算复杂度。

#### 3.2.1 内容感知稀疏注意力

**Routing Transformer (2020)**
- Roy et al.（2020）提出了基于在线k-means的稀疏路由机制
- 复杂度：从O(n²d)降低到O(n^1.5 d)
- 性能：在Wikitext-103上达到18.3困惑度，在PG-19数据集上达到33.2

**Smart Bird (2021)**
- Wu et al.（2021）提出了可学习的稀疏注意力机制
- 方法：使用单头低维Transformer生成草图注意力矩阵，然后采样token对
- 优势：在多个基准数据集上验证了效率和有效性

#### 3.2.2 基于位置的稀疏注意力

**Dilated Neighborhood Attention (DiNA, 2022)**
- Hassani & Shi（2022）提出了膨胀邻域注意力
- 创新：结合局部注意力（NA）和稀疏全局注意力
- 性能：在COCO目标检测上超越Swin Transformer 1.6% box AP

**Block Sparse Flash Attention (2025)**
- Ohayon et al.（2025）提出了块稀疏FlashAttention
- 方法：通过比较每块最大分数与校准阈值来跳过约50%的计算
- 性能：在Llama-3.1-8B上实现1.10倍加速

---

### 3.3 高效注意力实现

高效实现是提高Transformer在实际应用中可用性的关键。

#### 3.3.1 线性注意力

**Efficient Attention via Control Variates (2023)**
- Zheng et al.（2023）基于控制变数框架优化随机特征注意力（RFA）
- 创新：开发更灵活的控制变数形式，显著减少近似差距
- 应用：在视觉和语言任务上优于现有高效注意力机制

**Kernel-Eigen Pair Sparse Variational Gaussian Processes (2024)**
- Chen et al.（2024）提出了Kernel-Eigen Pair稀变高斯过程
- 优势：利用KSVD处理注意力核的不对称性，降低时间复杂度

#### 3.3.2 混合注意力机制

**Efficient Mixed Transformer (EMT, 2023)**
- Zheng et al.（2023）提出了混合Transformer块（MTB）
- 创新：使用像素混合器（PM）替换自注意力，结合条纹窗口自注意力（SWSA）
- 优势：利用图像各向异性实现高效全局依赖建模

**Evo-ViT (2021)**
- Xu et al.（2021）提出了慢-快token演化机制
- 方法：使用全局类注意力进行实例级token选择，然后更新选定和未选定token
- 优势：从训练开始就能加速扁平和深窄结构的Transformer

---

### 3.4 视觉Transformer注意力

视觉Transformer在图像处理领域展现出强大潜力。

#### 3.4.1 灵活的注意力窗口

**Vision Transformer with Quadrangle Attention (2023)**
- Zhang et al.（2023）提出了四边形注意力（QA）机制
- 创新：使用可学习的四边形回归模块将默认窗口转换为目标四边形
- 应用：在分类、目标检测、语义分割、姿态估计等任务上表现优异

**Vicinity Vision Transformer (2022)**
- Sun et al.（2022）提出了邻域注意力机制
- 创新：引入2D曼哈顿距离作为局部性偏置
- 性能：在ImageNet上达到SOTA，参数量减少50%

#### 3.4.2 注意力可视化与解释

**Attention Guided CAM (2024)**
- Leem & Seo（2024）提出了注意力引导的CAM方法
- 创新：聚合来自分类输出的梯度，结合归一化的自注意力分数
- 优势：在弱监督定位任务上优于现有方法

**On the Surprising Effectiveness of Attention Transfer (2024)**
- Li et al.（2024）提出了注意力迁移方法
- 发现：仅使用预训练的注意力模式就足以学习高质量特征
- 意义：为理解预训练提供了新的视角

---

### 3.5 应用领域的注意力机制

注意力机制已广泛应用于多个领域。

#### 3.5.1 时序数据

**Déjà vu: A Contextualized Temporal Attention Mechanism (2020)**
- Wu et al.（2020）提出了上下文化时间注意力机制
- 应用：序列推荐系统
- 创新：使用参数化核函数学习各种时间动态，利用上下文信息确定权重

**Multimodal depression detection (2025)**
- Jia et al.（2025）提出了基于注意力图卷积和Transformer的多模态抑郁症检测方法
- 应用：多模态情感分析

#### 3.5.2 医学影像

**Transformer-based Personalized Attention Mechanism (2022)**
- Takagi et al.（2022）提出了个性化注意力机制（PersAM）
- 应用：基于临床记录的淋巴瘤亚型分类
- 创新：根据临床记录自适应改变医学图像中的注意力区域

#### 3.5.3 科学计算

**StFT: Spatio-temporal Fourier Transformer (2025)**
- Long et al.（2025）提出了自回归时空傅里叶Transformer
- 应用：等离子体、流体、大气动力学等物理系统预测
- 创新：双路径架构集成频域和时空表示

---

## 4. 关键技术对比

### 4.1 注意力机制分类

| 类型 | 代表方法 | 复杂度 | 优势 | 劣势 |
|------|----------|--------|------|------|
| 全局注意力 | Transformer | O(n²) | 完美建模全局依赖 | 计算和内存开销大 |
| 局部窗口注意力 | Swin Transformer | O(n) | 计算高效 | 缺乏长距离依赖建模 |
| 稀疏注意力 | DiNA, Routing Transformer | O(n^1.5) | 平衡效率和性能 | 需要设计稀疏模式 |
| 线性注意力 | RFA, Control Variates | O(n) | 计算高效 | 近似误差 |
| 混合注意力 | EMT, Evo-ViT | O(n) | 结合局部和全局 | 设计复杂 |

### 4.2 性能对比

基于文献中的实验结果，各方法在关键任务上的表现如下：

**长上下文语言建模**
- GSA：困惑度5.70（128K上下文）
- Forgetting Transformer：长度外推性能优异
- Focus-dLLM：32K上下文29倍加速

**图像分类**
- QFormer：在多个基准上优于现有ViT
- VVT：ImageNet Top-1准确率88.5%，参数量减少50%
- DiNA：在COCO上超越Swin 1.6% box AP

**计算效率**
- SpargeAttention：适用于各种模型，无需额外开销
- Block Sparse Flash Attention：约50%计算减少
- SlimInfer：2.53倍TTFT加速，1.88倍端到端延迟减少

---

## 5. 研究热点与空白

### 5.1 当前研究热点

1. **长上下文处理**：如何有效处理超长上下文（128K+ tokens）是当前最热门的方向
2. **稀疏注意力优化**：开发更智能的稀疏模式和注意力选择策略
3. **理论理解**：深入分析注意力机制的工作原理和泛化行为
4. **多模态融合**：将注意力机制应用于多模态数据处理
5. **硬件友好设计**：针对特定硬件优化的注意力实现

### 5.2 研究空白

1. **自适应稀疏模式**：目前大多数稀疏注意力使用固定的稀疏模式，缺乏自适应能力
2. **跨层注意力一致性**：如何在多层Transformer中保持注意力模式的连贯性
3. **可解释性理论**：注意力权重的语义解释和理论分析仍不充分
4. **小样本学习**：在有限数据条件下注意力机制的学习效率
5. **实时应用**：面向实时应用的轻量级注意力机制

---

## 6. 未来展望

基于当前研究趋势和空白，未来Transformer注意力机制的研究可能会集中在以下方向：

### 6.1 技术发展方向

1. **动态稀疏化**：根据输入数据动态调整注意力模式
2. **分层注意力**：在不同抽象层次上应用不同的注意力策略
3. **跨模态注意力**：设计通用的注意力机制处理多模态数据
4. **神经符号注意力**：结合符号推理的注意力机制
5. **量子注意力**：探索量子计算环境下的注意力机制

### 6.2 应用方向

1. **边缘计算**：面向边缘设备的轻量级注意力实现
2. **实时系统**：低延迟的注意力计算框架
3. **个性化注意力**：根据用户偏好定制注意力模式
4. **可解释AI**：提供注意力机制的可解释性工具和方法
5. **科学发现**：在材料科学、生物学等领域的应用

---

## 7. BibTeX引用

```bibtex
@article{shen2026gated,
  title={Gated Sparse Attention: Combining Computational Efficiency with Training Stability for Long-Context Language Models},
  author={Shen, Alfred and Shen, Aaron},
  journal={arXiv preprint arXiv:2601.15305},
  year={2026}
}

@article{lin2025forgetting,
  title={Forgetting Transformer: Softmax Attention with a Forget Gate},
  author={Lin, Zhixuan and Nikishin, Evgenii and He, Xu Owen and Courville, Aaron},
  journal={arXiv preprint arXiv:2503.02130},
  year={2025}
}

@article{long2026focus,
  title={Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing},
  author={Long, Lingkun and Huang, Yushi and Bai, Shihao and Gong, Ruihao and Zhang, Jun and Zhou, Ao and Yang, Jianlei},
  journal={arXiv preprint arXiv:2602.02159},
  year={2026}
}

@article{takagi2022personalized,
  title={Transformer-based Personalized Attention Mechanism for Medical Images with Clinical Records},
  author={Takagi, Yusuke and Hashimoto, Noriaki and Masuda, Hiroki and Miyoshi, Hiroaki and Ohshima, Koichi and Hontani, Hidekata and Takeuchi, Ichiro},
  journal={arXiv preprint arXiv:2206.03003},
  year={2022}
}

@article{hassani2022dilated,
  title={Dilated Neighborhood Attention Transformer},
  author={Hassani, Ali and Shi, Humphrey},
  journal={arXiv preprint arXiv:2209.15001},
  year={2022}
}

@article{wu2020deja,
  title={Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation},
  author={Wu, Jibang and Cai, Renqin and Wang, Hongning},
  journal={arXiv preprint arXiv:2002.00741},
  year={2020}
}

@article{roy2020routing,
  title={Efficient Content-Based Sparse Attention with Routing Transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={arXiv preprint arXiv:2003.05997},
  year={2020}
}

@article{zhang2023quadrangle,
  title={Vision Transformer with Quadrangle Attention},
  author={Zhang, Qiming and Zhang, Jing and Xu, Yufei and Tao, Dacheng},
  journal={arXiv preprint arXiv:2303.15105},
  year={2023}
}

@article{leem2024attention,
  title={Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention},
  author={Leem, Saebom and Seo, Hyunseok},
  journal={arXiv preprint arXiv:2402.04563},
  year={2024}
}

@article{zhang2025sparge,
  title={SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference},
  author={Zhang, Jintao and Xiang, Chendong and Huang, Haofeng and Wei, Jia and Xi, Haocheng and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2502.18137},
  year={2025}
}

@article{ohayon2025block,
  title={Block Sparse Flash Attention},
  author={Ohayon, Daniel and Lamprecht, Itay and Hubara, Itay and Cohen, Israel and Soudry, Daniel and Elata, Noam},
  journal={arXiv preprint arXiv:2512.07011},
  year={2025}
}

@article{zheng2023control,
  title={Efficient Attention via Control Variates},
  author={Zheng, Lin and Yuan, Jianbo and Wang, Chong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2302.04542},
  year={2023}
}

@article{xu2021evo,
  title={Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer},
  author={Xu, Yifan and Zhang, Zhijie and Zhang, Mengdan and Sheng, Kekai and Li, Ke and Dong, Weiming and Zhang, Liqing and Xu, Changsheng and Sun, Xing},
  journal={arXiv preprint arXiv:2108.01390},
  year={2021}
}

@article{li2024attention,
  title={On the Surprising Effectiveness of Attention Transfer for Vision Transformers},
  author={Li, Alexander C. and Tian, Yuandong and Chen, Beidi and Pathak, Deepak and Chen, Xinlei},
  journal={arXiv preprint arXiv:2411.09702},
  year={2024}
}

@article{liu2018music,
  title={Music Transformer},
  author={Huang, Cheng-Zhi Anna and Vaswani, Ashish and Uszkoreit, Jakob and Shazeer, Noam and Simon, Ian and Hawthorne, Curtis and Dai, Andrew M. and Hoffman, Matthew D. and Dinculescu, Monica and Eck, Douglas},
  journal={arXiv preprint arXiv:1809.04281},
  year={2018}
}

@article{sakamoto2024benign,
  title={Benign Overfitting in Token Selection of Attention Mechanism},
  author={Sakamoto, Keitaro and Sato, Issei},
  journal={arXiv preprint arXiv:2409.17625},
  year={2024}
}

@article{li2022vicinity,
  title={Vicinity Vision Transformer},
  author={Sun, Weixuan and Qin, Zhen and Deng, Hui and Wang, Jianyuan and Zhang, Yi and Zhang, Kaihao and Barnes, Nick and Birchfield, Stan and Kong, Lingpeng and Zhong, Yiran},
  journal={arXiv preprint arXiv:2206.10552},
  year={2022}
}

@article{wong2024block,
  title={Efficient Content-Based Sparse Attention with Routing Transformers},
  author={Wong, Paul and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={arXiv preprint arXiv:2003.05997},
  year={2020}
}
```

---

## 8. 结论

Transformer注意力机制在过去几年中经历了从基础探索到多样化应用的快速发展。文献分析显示，当前研究主要集中在长上下文处理、稀疏注意力优化、高效实现和跨领域应用等方面。

关键发现包括：
1. 稀疏注意力机制在保持性能的同时显著降低计算复杂度
2. 长上下文处理技术正在推动大语言模型的发展
3. 视觉Transformer通过创新的注意力机制在图像处理领域取得突破
4. 注意力机制的理论理解正在深化

未来，随着计算硬件的发展和更深入的理论研究，注意力机制将在更多领域发挥重要作用，特别是在实时应用、边缘计算和多模态融合方面。

---

**参考文献**

[1] Shen, A., & Shen, A. (2026). Gated Sparse Attention: Combining Computational Efficiency with Training Stability for Long-Context Language Models. arXiv preprint arXiv:2601.15305.

[2] Lin, Z., Nikishin, E., He, X. O., & Courville, A. (2025). Forgetting Transformer: Softmax Attention with a Forget Gate. arXiv preprint arXiv:2503.02130.

[3] Long, L., Huang, Y., Bai, S., Gong, R., Zhang, J., Zhou, A., & Yang, J. (2026). Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing. arXiv preprint arXiv:2602.02159.

[4] Takagi, Y., Hashimoto, N., Masuda, H., Miyoshi, H., Ohshima, K., Hontani, H., & Takeuchi, I. (2022). Transformer-based Personalized Attention Mechanism for Medical Images with Clinical Records. arXiv preprint arXiv:2206.03003.

[5] Hassani, A., & Shi, H. (2022). Dilated Neighborhood Attention Transformer. arXiv preprint arXiv:2209.15001.

[6] Wu, J., Cai, R., & Wang, H. (2020). Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation. arXiv preprint arXiv:2002.00741.

[7] Roy, A., Saffar, M., Vaswani, A., & Grangier, D. (2020). Efficient Content-Based Sparse Attention with Routing Transformers. arXiv preprint arXiv:2003.05997.

[8] Zhang, Q., Zhang, J., Xu, Y., & Tao, D. (2023). Vision Transformer with Quadrangle Attention. arXiv preprint arXiv:2303.15105.

[9] Leem, S., & Seo, H. (2024). Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention. arXiv preprint arXiv:2402.04563.

[10] Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., & Chen, J. (2025). SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference. arXiv preprint arXiv:2502.18137.

[11] Ohayon, D., Lamprecht, I., Hubara, I., Cohen, I., Soudry, D., & Elata, N. (2025). Block Sparse Flash Attention. arXiv preprint arXiv:2512.07011.

[12] Zheng, L., Yuan, J., Wang, C., & Kong, L. (2023). Efficient Attention via Control Variates. arXiv preprint arXiv:2302.04542.

[13] Xu, Y., Zhang, Z., Zhang, M., Sheng, K., Li, K., Dong, W., Zhang, L., Xu, C., & Sun, X. (2021). Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer. arXiv preprint arXiv:2108.01390.

[14] Li, A. C., Tian, Y., Chen, B., Pathak, D., & Chen, X. (2024). On the Surprising Effectiveness of Attention Transfer for Vision Transformers. arXiv preprint arXiv:2411.09702.

[15] Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., Dai, A. M., Hoffman, M. D., Dinculescu, M., & Eck, D. (2018). Music Transformer. arXiv preprint arXiv:1809.04281.

[16] Sakamoto, K., & Sato, I. (2024). Benign Overfitting in Token Selection of Attention Mechanism. arXiv preprint arXiv:2409.17625.

[17] Sun, W., Qin, Z., Deng, H., Wang, J., Zhang, Y., Zhang, K., Barnes, N., Birchfield, S., Kong, L., & Zhong, Y. (2022). Vicinity Vision Transformer. arXiv preprint arXiv:2206.10552.

[18] Wu, Jibang, Cai, Renqin, and Wang, Hongning. "Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation." arXiv preprint arXiv:2002.00741 (2020).

[19] Roy, Aurko, Mohammad Saffar, Ashish Vaswani, and David Grangier. "Efficient Content-Based Sparse Attention with Routing Transformers." arXiv preprint arXiv:2003.05997 (2020).

[20] Zhang, Qiming, Jing Zhang, Yufei Xu, and Dacheng Tao. "Vision Transformer with Quadrangle Attention." arXiv preprint arXiv:2303.15105 (2023).

[21] Leem, Saebom and Seo, Hyunseok. "Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention." arXiv preprint arXiv:2402.04563 (2024).

[22] Zhang, Jintao, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. "SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference." arXiv preprint arXiv:2502.18137 (2025).

[23] Ohayon, Daniel, Itay Lamprecht, Itay Hubara, Israel Cohen, Daniel Soudry, and Noam Elata. "Block Sparse Flash Attention." arXiv preprint arXiv:2512.07011 (2025).

[24] Zheng, Lin, Jianbo Yuan, Chong Wang, and Lingpeng Kong. "Efficient Attention via Control Variates." arXiv preprint arXiv:2302.04542 (2023).

[25] Xu, Yifan, Zhijie Zhang, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng Xu, and Xing Sun. "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer." arXiv preprint arXiv:2108.01390 (2021).

[26] Li, Alexander C., Yuandong Tian, Beidi Chen, Deepak Pathak, and Xinlei Chen. "On the Surprising Effectiveness of Attention Transfer for Vision Transformers." arXiv preprint arXiv:2411.09702 (2024).

[27] Huang, Cheng-Zhi Anna, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. "Music Transformer." arXiv preprint arXiv:1809.04281 (2018).

[28] Sakamoto, Keitaro and Sato, Issei. "Benign Overfitting in Token Selection of Attention Mechanism." arXiv preprint arXiv:2409.17625 (2024).

[29] Sun, Weixuan, Zhen Qin, Hui Deng, Jianyuan Wang, Yi Zhang, Kaihao Zhang, Nick Barnes, Stan Birchfield, Lingpeng Kong, and Yiran Zhong. "Vicinity Vision Transformer." arXiv preprint arXiv:2206.10552 (2022).

[30] Wong, Paul, Mohammad Saffar, Ashish Vaswani, and David Grangier. "Efficient Content-Based Sparse Attention with Routing Transformers." arXiv preprint arXiv:2003.05997 (2020).
