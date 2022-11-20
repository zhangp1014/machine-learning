# machine-learning
CNN &amp; SVM

import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self, C=1.0):
        self.support_vectors = None  # 支持向量
        self.C = C  # 惩罚系数，惩罚松弛变量， C值越大，拟合越好，泛化能力越弱
        self.W = None  # 参数W  shape (d,)
        self.b = None  # 参数b
        self.X = None  # 特征  shape (n,d)
        self.y = None  # 标签  shape (n,)
        self.n = 0  # 样本数量
        self.d = 0  # 特征维度

