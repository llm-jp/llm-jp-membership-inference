# llm-jp-membership-inference
This is the repository for llm jp membership inference attack.

## Description
Membership inference attack is a type of privacy attack that aims to determine whether a specific data sample was used to train a machine learning model.   
In this project, we implement current representative membership inference attacks.  
For a specific method, usually, it caculates a feature value for every input sample. Then each MIA method has its own hypothesis in how to seperate trained or un-trained samples, for exmaple, the loss method would assume that the loss of trained samples is smaller than un-trained samples. Then it would use the feature value to predict the label of the sample.

### Attack Methods
We implement the following MIA methods:  
- Loss-based MIA
- Gradient-based MIA
- Perplexity-based MIA
- RECALL-based MIA


### How to run the code
Please refer to the test.py file for how to run the code.
