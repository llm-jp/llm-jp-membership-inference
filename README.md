# llm-jp-membership-inference
This is the repository for llm jp membership inference attack.

## Description
Membership inference attack is a type of privacy attack that aims to determine whether a specific data sample was used to train a machine learning model. In this project, we propose a membership inference attack on a Japanese language model (LLM) trained on a large-scale Japanese text corpus. We demonstrate that the attack can successfully infer membership with high accuracy, even when the model is fine-tuned on a small dataset. We also show that the attack can be used to identify the presence of specific keywords in the training data, which can be used to infer the training data's source.  
Usually, it caculates a feature value for every input sample. Then each MIA method has its own hypothesis in how to seperate trained or un-trained samples, for exmaple, the loss method would assume that the loss of trained samples is smaller than un-trained samples. Then it would use the feature value to predict the label of the sample.

### Attack Methods
We implement the following MIA methods:  
- Loss-based MIA