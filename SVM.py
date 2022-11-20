
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

    def __desicion_function(self, X):  # 决策函数 模型
        return X.dot(self.W) + self.b

    def __margin(self, X, y):  # 函数间隔
        return y * (X.dot(self.W) + self.b)

    def __cost(self, margin):
        return (1 / 2) * self.W.dot(self.W) + self.C * np.sum(np.maximum(0, 1 - margin))

    def fit(self, X, y, lr=0.001, epoch=500):  # 训练
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.W = np.random.rand(self.d)
        self.b = np.random.rand()
        self.X = X
        self.y = y
        losses = []
        for i in range(epoch):
            margin = self.__margin(X, y)
            loss = self.__cost(margin)
            losses.append(loss)

            miss_idx = np.where(margin < 1)
            df_W = self.W - self.C * y[miss_idx].dot(X[miss_idx])  # 计算W偏导
            self.W = self.W - lr * df_W  # 更新W

            df_b = -self.C * np.sum(y[miss_idx])  # 计算b编导
            self.b = self.b - lr * df_b  # 更新b

            self.support_vectors = X[miss_idx]

    def predict(self, X):
        return np.sign(self.__desicion_function(X))

    def score(self, X, y):  # 得分，1为分类全对，0为分类全错
        P = self.predict(X)
        return np.mean(P == y)


if __name__ == '__main__':
    # 生成随机的两类数据点，每类中各30个样本点
    np.random.seed(5)
    X = np.r_[np.random.randn(30, 2) - [2, 2], np.random.randn(30, 2) + [2, 2]]
    y = np.array([1] * 30 + [-1] * 30)

    clf = SVM(C=0.2)
    clf.fit(X, y)

    plt.figure(figsize=(8, 4))
    xx = np.linspace(-5, 5)  # xx为待绘制的分类线横坐标区间，(-5,5)之间x的值
    yy = -(clf.W[0] * xx + clf.b) / clf.W[1]  # yy为分类线区间上对应x的y值
    yy_up = -(clf.W[0] * xx + clf.b - 1) / clf.W[1]  # yy_up为分类线正方向的间隔线
    yy_down = -(clf.W[0] * xx + clf.b + 1) / clf.W[1]  # 分类线负方向的间隔线

    plt.plot(xx, yy, color='r')
    plt.plot(xx, yy_down, '--', color='y')
    plt.plot(xx, yy_up, '--', color='g')
    plt.scatter(clf.support_vectors[:, 0], clf.support_vectors[:, 1], s=80)  # X[:,0]表示第0维取全部元素（:），第1维取第0个元素
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # plt.cm中cm全称表示colormap，paired表示两个两个相近色彩输出

    plt.show()
