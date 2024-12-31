from mia_dataset import WikiMIA
from mia_model import GPTNeoX
from mia_method import LossMIA

test_data = WikiMIA("WikiMIA", length=64)
model = GPTNeoX("70m")
mia_method = LossMIA()
member_llm_outputs, member_tokenized_intputs, member_tokenized_targets = model.collect_outputs(test_data.member)
non_member_llm_outputs, non_member_tokenized_intput, non_member_tokenized_target = model.collect_outputs(test_data.non_member)

member_feature_value = mia_method.feature_compute(member_llm_outputs, member_tokenized_intputs, member_tokenized_targets)
non_member_feature_value = mia_method.feature_compute(non_member_llm_outputs, non_member_tokenized_intput, non_member_tokenized_target)







