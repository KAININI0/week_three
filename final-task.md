```
import numpy as np
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt

# ======================
# 基础工具函数（无依赖）
# ======================
def sigmoid(z):
    """sigmoid激活函数，防止数值溢出"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def standardize(X):
    """特征标准化（Z-score）"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # 防止标准差为0导致除0错误
    return (X - mean) / std

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """训练集测试集随机划分"""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split_point = int(len(X) * (1 - test_ratio))
    return X[indices[:split_point]], X[indices[split_point:]], y[indices[:split_point]], y[indices[split_point:]]

# ======================
# 1. 线性回归模型（纯手动实现）
# ======================
class LinearRegression:
    def __init__(self, lr=0.01, epochs=10000, seed=42):
        self.lr = lr          # 学习率
        self.epochs = epochs  # 迭代次数
        self.seed = seed
        self.w = None         # 权重参数
        self.b = 0            # 偏置项

    def fit(self, X, y):
        """梯度下降训练模型"""
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化权重

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        """预测连续值"""
        return np.dot(X, self.w) + self.b

# 线性回归评估指标
def calculate_rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_r2(y_true, y_pred):
    """决定系数R²"""
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total) if ss_total != 0 else 0

# ======================
# 2. 逻辑回归模型（纯手动实现）
# ======================
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=10000, seed=42):
        self.lr = lr          # 学习率
        self.epochs = epochs  # 迭代次数
        self.seed = seed
        self.w = None         # 权重参数
        self.b = 0            # 偏置项

    def fit(self, X, y):
        """梯度下降训练模型"""
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化权重

        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_pred = sigmoid(z)
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        """预测正类概率"""
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测分类结果"""
        return (self.predict_proba(X) >= threshold).astype(int)

# 逻辑回归评估指标
def calculate_accuracy(y_true, y_pred):
    """准确率"""
    return np.sum(y_true == y_pred) / len(y_true)

def calculate_confusion_matrix(y_true, y_pred):
    """混淆矩阵"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def calculate_precision_recall_f1(y_true, y_pred):
    """精确率、召回率、F1分数"""
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def calculate_roc_auc(y_true, y_score):
    """手动计算ROC-AUC"""
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    tpr, fpr = [0.0], [0.0]
    tp, fp = 0, 0
    prev_score = -np.inf

    for i in range(len(y_score)):
        if y_score[sorted_idx[i]] != prev_score:
            tpr.append(tp / n_pos if n_pos != 0 else 0)
            fpr.append(fp / n_neg if n_neg != 0 else 0)
            prev_score = y_score[sorted_idx[i]]
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(1.0)
    fpr.append(1.0)

    # 梯形法计算AUC
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc, fpr, tpr

# ======================
# 3. K-Means聚类模型（纯手动实现）
# ======================
class KMeans:
    def __init__(self, k=3, max_iter=100, seed=42):
        self.k = k                # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.seed = seed
        self.centers = None       # 聚类中心

    def fit(self, X):
        """训练聚类模型"""
        np.random.seed(self.seed)
        n_samples = X.shape[0]
        # 从样本中随机初始化聚类中心
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[random_idx]

        for _ in range(self.max_iter):
            # 分配样本到最近的簇
            clusters = [[] for _ in range(self.k)]
            for sample in X:
                dists = [np.sum((sample - center) ** 2) for center in self.centers]
                clusters[np.argmin(dists)].append(sample)

            # 更新聚类中心
            new_centers = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centers.append(self.centers[np.random.randint(self.k)])
                else:
                    new_centers.append(np.mean(cluster, axis=0))
            new_centers = np.array(new_centers)

            # 收敛判断：中心不再变化
            if np.allclose(new_centers, self.centers):
                break
            self.centers = new_centers

    def predict(self, X):
        """预测样本所属簇"""
        labels = []
        for sample in X:
            dists = [np.sum((sample - center) ** 2) for center in self.centers]
            labels.append(np.argmin(dists))
        return np.array(labels)

# K-Means评估指标
def calculate_sse(X, labels, centers):
    """簇内平方和SSE"""
    sse = 0.0
    for i in range(len(X)):
        sse += np.sum((X[i] - centers[labels[i]]) ** 2)
    return sse

def calculate_silhouette_score(X, labels):
    """轮廓系数"""
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters == 1:
        return -1.0

    silhouette_scores = np.zeros(n_samples)
    for i in range(n_samples):
        current_label = labels[i]
        same_cluster = X[labels == current_label]
        # 计算a：同簇平均距离
        a = np.mean([np.sum((X[i] - s) ** 2) for s in same_cluster if not np.array_equal(X[i], s)]) if len(same_cluster) > 1 else 0.0
        # 计算b：到其他簇的最小平均距离
        min_b = np.inf
        for label in unique_labels:
            if label == current_label:
                continue
            other_cluster = X[labels == label]
            avg_dist = np.mean([np.sum((X[i] - s) ** 2) for s in other_cluster])
            if avg_dist < min_b:
                min_b = avg_dist
        b = min_b
        silhouette_scores[i] = (b - a) / max(a, b)
    return np.mean(silhouette_scores)

# ======================
# 自动下载数据集（无需手动操作）
# ======================
def download_wine_quality():
    """自动下载红酒质量数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    file_path = "winequality-red.csv"
    if not os.path.exists(file_path):
        print("正在下载红酒质量数据集...")
        with open(file_path, "wb") as f:
            f.write(requests.get(url).content)
        print("下载完成！")
    return pd.read_csv(file_path, sep=";")

def download_iris():
    """自动下载鸢尾花数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    file_path = "iris.csv"
    if not os.path.exists(file_path):
        print("正在下载鸢尾花数据集...")
        with open(file_path, "wb") as f:
            f.write(requests.get(url).content)
        # 添加表头
        cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        pd.read_csv(file_path, header=None, names=cols).to_csv(file_path, index=False)
        print("下载完成！")
    return pd.read_csv(file_path)

# ======================
# 主程序：执行所有任务
# ======================
if __name__ == "__main__":
    # 解决matplotlib中文乱码
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # ======================================
    # 任务1：红酒数据集 线性回归+逻辑回归
    # ======================================
    print("=" * 70)
    print("📊 任务1：Wine Quality 红酒质量数据集")
    print("=" * 70)

    # 数据预处理
    wine_df = download_wine_quality()
    X_wine = wine_df.drop("quality", axis=1).values
    y_reg = wine_df["quality"].values                  # 回归标签：质量分数
    y_clf = (wine_df["quality"] > 6).astype(int).values # 分类标签：>6为好酒(1)
    X_wine_std = standardize(X_wine)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_wine_std, y_reg)
    _, _, y_train_clf, y_test_clf = train_test_split(X_wine_std, y_clf)

    # 线性回归训练与评估
    print("\n🔹 线性回归（预测红酒质量分数）结果：")
    lr_model = LinearRegression(lr=0.05, epochs=20000)
    lr_model.fit(X_train, y_train_reg)
    y_pred_reg = lr_model.predict(X_test)
    print(f"RMSE：{calculate_rmse(y_test_reg, y_pred_reg):.4f}")
    print(f"R² ：{calculate_r2(y_test_reg, y_pred_reg):.4f}")

    # 逻辑回归训练与评估
    print("\n🔹 逻辑回归（区分好/坏酒）结果：")
    lgr_model = LogisticRegression(lr=0.1, epochs=20000)
    lgr_model.fit(X_train, y_train_clf)
    y_pred_clf = lgr_model.predict(X_test)
    y_score_clf = lgr_model.predict_proba(X_test)

    acc = calculate_accuracy(y_test_clf, y_pred_clf)
    precision, recall, f1 = calculate_precision_recall_f1(y_test_clf, y_pred_clf)
    auc, fpr, tpr = calculate_roc_auc(y_test_clf, y_score_clf)
    print(f"准确率：{acc:.4f}")
    print(f"精确率：{precision:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    print(f"ROC-AUC：{auc:.4f}")
    # 保存ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("假正例率(FPR)")
    plt.ylabel("真正例率(TPR)")
    plt.title("逻辑回归ROC曲线")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    print("\nROC曲线已保存为 roc_curve.png")
    # 模型异同点总结（可直接写进报告）
    print("\n" + "=" * 70)
    print("📋 线性回归 vs 逻辑回归 异同点总结")
    print("=" * 70)
    print("【相同点】")
    print("1. 均为线性模型，核心结构为 z = X·w + b 的线性组合")
    print("2. 均使用梯度下降算法优化参数")
    print("3. 对特征量纲敏感，必须做标准化预处理")
    print("4. 同属广义线性模型范畴")
    print("\n【不同点】")
    print("1. 任务类型：线性回归→回归任务（预测连续值）；逻辑回归→分类任务（输出概率）")
    print("2. 输出范围：线性回归(-∞,+∞)；逻辑回归经sigmoid映射到(0,1)")
    print("3. 损失函数：线性回归用均方误差MSE；逻辑回归用交叉熵损失")
    print("4. 数据假设：线性回归假设数据服从高斯分布；逻辑回归假设服从伯努利分布")
    print("5. 评估指标：线性回归用RMSE、R²；逻辑回归用准确率、F1、ROC-AUC")
    # ======================================
    # 任务2：鸢尾花数据集 K-Means聚类
    # ======================================
    print("\n" + "=" * 70)
    print("📊 任务2：Iris 鸢尾花数据集 K-Means聚类")
    print("=" * 70)
    # 数据预处理
    iris_df = download_iris()
    X_iris = iris_df.drop("species", axis=1).values
    X_iris_std = standardize(X_iris)
    # 聚类训练与评估
    kmeans_model = KMeans(k=3, max_iter=100)
    kmeans_model.fit(X_iris_std)
    cluster_labels = kmeans_model.predict(X_iris_std)
    sse = calculate_sse(X_iris_std, cluster_labels, kmeans_model.centers)
    silhouette = calculate_silhouette_score(X_iris_std, cluster_labels)
    print("\n🔹 K-Means聚类结果：")
    print(f"簇内平方和(SSE)：{sse:.4f}")
    print(f"轮廓系数：{silhouette:.4f}")
    print(f"聚类标签分布：{np.bincount(cluster_labels)}")
    print(f"真实标签分布：{np.bincount(pd.factorize(iris_df['species'])[0])}")
    print("\n" + "=" * 70)
    print("✅ 所有任务执行完成！")
    print("=" * 70)
```