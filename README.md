# llm-jp-membership-inference
This is the repository for llm jp membership inference attack.

## Description
Membership inference attack is a type of privacy attack that aims to determine whether a specific data sample was used to train a machine learning model.   
In this project, we implement current representative membership inference attacks.  
For a specific method, usually, it caculates a feature value for every input sample. Then each MIA method has its own hypothesis in how to seperate trained or un-trained samples, for exmaple, the loss method would assume that the loss of trained samples is smaller than un-trained samples. Then it would use the feature value to predict the label of the sample.

### Attack Methods
We implemented the following MIA methods:  

Gray-box Method:
- Loss-based MIA
- Gradient-based MIA
- Perplexity-based MIA
- Reference-based MIA
- Min-k\% 
- Max-k\% ++
- EDA-PAC  

Black-box Method:
- SaMIA
- CDD


### How to run the code
Please refer to the test.py file for how to run the code.
#### Code　Structure
The code is structured as follows:  
- `mia_dataset.py` is the dataset class for the MIA attack. 
- `mia_model.py` is the attacked model class for the MIA attack.
- `mia_methods.py` is the MIA attack methods class.

#### How to use
You may refer to the test.py for a simple example usage.  
In general, the codes are used as follows:  
1. Load the dataset for the MIA attack. This could be a dataset that is already processed in current codes (WikiMIA), but also can be a dataset that you have prepared by your own. A MIA dataset has two list of samples, one is for the member samples and another one is for the non-member samples. You can access those two samples by using `dataset.member` and `datast.non_member`.
2. Create the target model for the MIA attack. We have already prepared GPTNeox as the initial model. You can use this model as the target model, or you can use your own model. You can refer to mia_model.py for how to create the target model.
3. Load MIA method from `mia_methods.py`. Some methods have hyparameters, you should refer to this py file to check related hyperparameters as it may have certain influences on the performance of MIA attack.
4. Run the MIA by using `mia_model.collect_outputs(dataset.member, mia_method)` and `mia_model.collect_outputs(dataset.non_member, mia_method)`. This will return the feature value caculated for every sample in both member and non-member set. Then you can use this feature value to predict the label of the sample.

### To-do
1. Add more MIA methods.
2. Add more datasets for the MIA attack.
3. Add evaluation metrics for the MIA attack.

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.  
For any wanted MIA methods that are not implemented, please feel free to open an issue.

### Caution
The analysis results of a MIA method does not mean the text is absolutely trained by the model.  
The result should be only used as a reference rather than an evidence to support conclusion like "my novel is trained by this LLM".  
Current SotA LLMs usually use a close-source training data, so it is impossible to get the ground truth of the training data. 
Thus, the MIA is only doing inference rather than a proof.



