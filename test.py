from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA
from mia_method import MinKMIA
from mia_method import MinKPlusMIA
from mia_method import SaMIA
from mia_method import CDDMIA

test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
mia_method = LossMIA()
mink_method = MinKMIA()
mink_plus_method = MinKPlusMIA()
samia_method = SaMIA()
cdd_method  = CDDMIA()

member_feature_value_dict = model.collect_outputs(test_data.member, cdd_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, cdd_method)









