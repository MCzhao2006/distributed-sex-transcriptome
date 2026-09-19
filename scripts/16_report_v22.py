# REPORT v2.1 增补: 第三轮评审实验(T3-1/2/3/4)
src = open('REPORT.md', encoding='utf-8').read()
src = src.replace("**版本**: v2.1(Red-Team 全清单完成, 已迁移至 F:\\nature, 2026-09-09)",
"**版本**: v2.2(第三轮评审 5 实验完成, F:\\nature, 2026-09-09)")

addition = """
---

## 四·二、第三轮评审实验(T3 系列, 2026-09-09)

### T3-1 匹配随机对照(回应"随机集可能捞到高表达基因")
四种抽样方案(plain / 表达四分位匹配 / 方差匹配 / 染色体匹配),N∈{200,1000,5000}×50 seeds:
- 三组织 × 三规模 × 四方案 AUC 全部重合(差异 <0.005, Muscle N=5000 均 ≈0.994-0.995)
- **弥漫信号不是表达量/方差/染色体分布的技术伪影** (results/t3_1_matched_dose.csv)

### T3-2 "p>0.5 基因集"反常的 sanity(回应第 10 节)
- auc(p) + auc(1−p) = 1.0000 精确成立 → 标签编码/probability/指标无 bug
- 系数结构对称(+1906/−1784,median |coef|=0.035)→ 弥散微系数模式
- 重跑得 AUC=0.95(与此前 0.25 反向)——方向取决于所选基因子集与协变量结构的交互,
  进一步支持**不给该现象生物学解释**,仅作为记录 (results/t3_2_anomaly_sanity.csv)

### T3-4 Leave-One-Tissue-Out(回应第 14 节, "真正共享程序"测试)
11 组织训练(每组织 40% 供体,严格供体不相交)→ 留出第 12 组织:
- 12/12 全部有效(测试 n=30–268): mean=0.686, median=0.708, 范围 0.566(Heart)–0.783(Blood)
- 单一共享模型在从未见过的组织+供体上性别预测全部成功
- Blood 从 pairwise 最弱(0.49)反转为 LOTO 最强(0.783)→ 11 组织联合信号含血液可读成分
- Heart/Colon/Esophagus 垫底与其 DE 基因最少一致 (results/t3_4_loto.csv)

### T3-3 TCGA→GTEx 反向跨队列验证(回应第 13 节)
TCGA 正常组织训练(patients 级 60/40 split, train-only DE top-500)→ GTEx 测试:
| 组织 | 正向 GTEx→TCGA | 反向 TCGA→GTEx |
|---|---|---|
| Thyroid | 0.850 | 0.651 |
| Lung | 0.763 | 0.757 |
| Colon | 0.774 | 0.585 |
| Esophagus | 0.639 | 0.521 |
| **mean** | **0.756** | **0.628** |
(BRCA→Skin 因 TCGA 训练集单性别(281F/2M)无效,已剔除)
- 双向均可迁移;反向较弱与 TCGA 训练样本量小(48-349)一致
- 措辞按评审修正:跨队列持久性**与生物学成分一致**,但不排除队列特异性混杂 (results/t3_3_reverse_validation.csv)

### 结论表述更新(按评审第 2/6/9/11/12 节)
- "特征选择无泄漏" → "主要 red-team 实验已采用 train-only 统计;早期 baseline 为历史对照,不作无泄漏主证据"
- "细胞组成本身非主要载体" → "基于标记基因的组成代理只能解释所测组织性别预测信号的一小部分"
- multitarget 定位 → "specificity benchmark"(信号特异性基准),非"混杂排除"
- sex×age 结论 → "未检测到大量显著交互",不宣称"跨年龄稳定"(年龄为区间中点,信息有损)
- PC 实验 → 定位为 stress test,非机制证据

---
"""
src = src.replace("## 五、局限", addition + "## 五、局限")
open('REPORT.md','w',encoding='utf-8').write(src)
print("REPORT -> v2.2")