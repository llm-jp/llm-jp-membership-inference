from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA
from mia_method import MinKMIA
from mia_method import MinKPlusMIA
from mia_method import SaMIA
from mia_method import CDDMIA
from mia_method import ReferenceMIA

test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
refer_method = ReferenceMIA()
member_feature_value_dict = model.collect_outputs(test_data.member, refer_method)
non_member_feature_value_dict = model.collect_outputs(test_data.non_member, refer_method)









