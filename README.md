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
- RECALL-based MIA
- Min-k\% 
- Max-k\% ++

Black-box Method:
- SaMIA
- CDD


### How to run the code
Please refer to the test.py file for how to run the code.

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.  
For any wanted MIA methods that are not implemented, please feel free to open an issue.

### Citation
If you use this code, please cite the following paper:
```
@misc{chen2024statisticalmultiperspectiverevisitingmembership,
      title={A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models}, 
      author={Bowen Chen and Namgi Han and Yusuke Miyao},
      year={2024},
      eprint={2412.13475},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13475}, 
}
```

