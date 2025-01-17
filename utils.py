import numpy as np
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
from sklearn.metrics import roc_auc_score

def distribution_similarity(values_list_1, value_list_2):
    pass

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