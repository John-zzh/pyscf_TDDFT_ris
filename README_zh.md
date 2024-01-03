**其他语言版本: [English](README.md), [中文](README_zh.md).**


# pyscf-TDDFT-ris (v1.0)
这个基于PySCF的Python包提供了半经验的TDDFT-ris方法，可以执行快速准确的TDDFT紫外-可见吸收光谱计算。TDDFT-ris方法基于精确的DFT计算，从完成了的SCF计算开始续算，比如高斯的`.fch`文件, ORCA的gbw文件，或者PySCF的`mf`对象。

目前，它支持UKS/RKS、TDA/TDDFT、纯/杂化/范围分离杂化泛函。

注意:

(1) 软件包TURBOMOLE7.7dev已经内置了TDDFT-ris，请参见[the TDDFT-ris+p plugin for Turbomole](https://github.com/John-zzh/TDDFT-ris)

(2) 软件包Amespv1.1dev已经内置了TDDFT-ris，请参见[Amesp](https://amesp.xyz/)


## 理论
在从头算线性响应TDDFT的框架下，我们构造了半经验方法TDDFT-ris[1,2]。ris方法是通过两个步骤实现的：

- 使用密度拟合（**RI-JK**）来近似双电子积分，但是每个原子只用一个$s$型高斯轨道作为辅助基
- 忽略交换关联泛函项的贡献

至于辅助基的参数，我们把它和原子半径联系起来。位于原子$A$上的$s$型轨道的指数$\alpha_A$与半经验原子半径$R_A$的关系为，
$\alpha_A = \frac{\theta}{R_A^2}$，
全局参数$\theta$用于微调以降低能量和光谱误差。

与传统的从头算TDDFT相比，在有机分子的激发能计算上，TDDFT-ris的能量误差几乎可以忽略不计，仅为0.06 eV。此外，它提供了显著的计算优势，速度约快300倍。这比sTDDFT方法有显著的改进，后者的能量误差为0.24 eV。

由于其构造与传统的TDDFT相似，TDDFT-ris可以轻松地整合到大多数量子化学软件包中，几乎不需要额外的实施工作。软件包如TURBOMOLE7.7和Amespv1.1dev已经内置了TDDFT-ris方法。

[ORCA5.2](https://orcaforum.kofo.mpg.de/app.php/portal)将在下一个版本中支持TDDFT-ris计算。