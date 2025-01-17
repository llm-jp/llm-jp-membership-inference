from mia_dataset import WikiMIA
from mia_model import MIAModel
from mia_method import LossMIA, MinKMIA, MinKPlusMIA, SaMIA, CDDMIA, ReferenceMIA, EDAPACMIA, RecallMIA
from utils import js_divergence, waterstein_distance, ks_hypothesis_test, roc_auc_score_caculation


test_data = WikiMIA("WikiMIA", length=64)
model = MIAModel("EleutherAI/pythia-70m-deduped")
recall_method = RecallMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, recall_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, recall_method)
print(member_feature_value_dict)
print(non_member_feature_value_dict)
js_distance = js_divergence(member_feature_value_dict[recall_method.name], non_member_feature_value_dict[recall_method.name])
waterstein_distance = waterstein_distance(member_feature_value_dict[recall_method.name], non_member_feature_value_dict[recall_method.name])
ks_statistic, ks_p_value = ks_hypothesis_test(member_feature_value_dict[recall_method.name], non_member_feature_value_dict[recall_method.name])
auc = roc_auc_score_caculation(member_feature_value_dict[recall_method.name], non_member_feature_value_dict[recall_method.name])
print(f"JS Divergence: {js_distance}")
print(f"Waterstein Distance: {waterstein_distance}")
print(f"KS Statistic: {ks_statistic}, KS P Value: {ks_p_value}")
print(f"AUC: {auc}")










