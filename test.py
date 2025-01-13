from mia_dataset import WikiMIA
from mia_model import MIAModel
from mia_method import LossMIA, MinKMIA, MinKPlusMIA, SaMIA, CDDMIA, ReferenceMIA, EDAPACMIA

test_data = WikiMIA("WikiMIA", length=64)
model = MIAModel("EleutherAI/pythia-70m-deduped")
eda_pac_method = EDAPACMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, eda_pac_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, eda_pac_method)
print(member_feature_value_dict)
print(non_member_feature_value_dict)









