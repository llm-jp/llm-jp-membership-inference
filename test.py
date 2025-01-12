from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA
from mia_method import MinKMIA
from mia_method import MinKPlusMIA
from mia_method import SaMIA

test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
mia_method = LossMIA()
mink_method = MinKMIA()
mink_plus_method = MinKPlusMIA()
samia_method = SaMIA()

member_feature_value_dict = model.collect_outputs(test_data.member, samia_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, samia_method)









