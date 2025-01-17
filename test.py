from mia_dataset import WikiMIA
from mia_model import MIAModel
from mia_method import LossMIA, MinKMIA, MinKPlusMIA, SaMIA, CDDMIA, ReferenceMIA, EDAPACMIA, RecallMIA
from utils import js_divergence, waterstein_distance, ks_hypothesis_test, roc_auc_score_caculation


test_data = WikiMIA("WikiMIA", length=64)
model = MIAModel("EleutherAI/pythia-70m-deduped")
samia_method = SaMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, samia_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, samia_method)
print(member_feature_value_dict)
print(non_member_feature_value_dict)
for key in list(member_feature_value_dict[samia_method.name].keys()):
    js_distance = js_divergence(member_feature_value_dict[samia_method.name][key], non_member_feature_value_dict[samia_method.name][key])
    waterstein_distance = waterstein_distance(member_feature_value_dict[samia_method.name][key], non_member_feature_value_dict[samia_method.name][key])
    ks_statistic, ks_p_value = ks_hypothesis_test(member_feature_value_dict[samia_method.name][key], non_member_feature_value_dict[samia_method.name][key])
    auc = roc_auc_score_caculation(member_feature_value_dict[samia_method.name][key], non_member_feature_value_dict[samia_method.name][key])
    print(f"JS Divergence: {js_distance}")
    print(f"Waterstein Distance: {waterstein_distance}")
    print(f"KS Statistic: {ks_statistic}, KS P Value: {ks_p_value}")
    print(f"AUC: {auc}")










