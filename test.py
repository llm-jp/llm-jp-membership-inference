from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA


test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
mia_method = LossMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, mia_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, mia_method)
print(member_feature_value_dict[mia_method.name])
print(non_member_feature_value_dict[mia_method.name])






