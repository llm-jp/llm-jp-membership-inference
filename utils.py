import numpy as np
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def js_divergence(value_list_1, value_list_2):
    hist1, bin_edges = np.histogram(value_list_1, bins=300, density=True)
    hist2, _ = np.histogram(value_list_2, bins=300, density=True)
    eps = 1e-10
    hist1 += eps
    hist2 += eps
    # 确保向量总和为1（归一化），表示概率分布
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    m = 0.5 * (hist1 + hist2)
    js_distance = 0.5 * entropy(hist1, m) + 0.5 * entropy(hist2, m)
    return js_distance

def waterstein_distance(value_list_1, value_list_2):
    return wasserstein_distance(value_list_1, value_list_2)

def ks_hypothesis_test(value_list_1, value_list_2):
    return ks_2samp(value_list_1, value_list_2)

def roc_auc_score_caculation(member_list, non_member_list):
    label = [1] * len(member_list) + [0] * len(non_member_list)
    score = member_list + non_member_list
    return roc_auc_score(label, score)

def gmean_method(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmeans)
    return thresholds[idx]

def decide_threshold_direction(y_true, scores, threshold):
    positive_scores = scores[y_true == 1]
    negative_scores = scores[y_true == 0]
    pos_mean = np.mean(positive_scores)
    neg_mean = np.mean(negative_scores)
    return '<=' if pos_mean < neg_mean else '>='


def threshold_selection(member_list, non_member_list):
    X_train, X_test, y_train, y_test = train_test_split(member_list + non_member_list, [1] * len(member_list) + [0] * len(non_member_list), test_size=0.2)
    # 使用G-mean方法计算最佳阈值
    best_threshold = gmean_method(y_train, X_train)
    threshold_direction = decide_threshold_direction(y_train, X_train, best_threshold)
    return best_threshold