from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA
from mia_method import MinKMIA
from mia_method import MinKPlusMIA
from mia_method import SaMIA
from mia_method import CDDMIA
from mia_method import ReferenceMIA
from mia_method import EDAPACMIA

test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
eda_pac_method = EDAPACMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, eda_pac_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, eda_pac_method)
print(member_feature_value_dict)
print(non_member_feature_value_dict)









