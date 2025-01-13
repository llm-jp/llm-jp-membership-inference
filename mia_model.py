from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MIAModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logging.log(logging.INFO, "Model and Tokenizer loaded")
        logging.basicConfig(level=logging.ERROR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model = self.model.to(self.device)

    def collect_outputs(self, text, mia_method, batch_size=4):
        logging.log(logging.INFO, "Running LLM on Inputted Member/Non-Member Text")
        data_loader = DataLoader(text, batch_size=batch_size, shuffle=False)
        all_texts = [text for batch_texts in data_loader for text in batch_texts]
        avg_length = int(np.mean([len(self.tokenizer.encode(ex)) for ex in all_texts]))
        # Tokenize all texts at once
        tokenized_inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,  # This will pad all sequences to the same length
            max_length=self.model.config.max_position_embeddings
        )
        # Move the tokenized data to the correct device
        input_ids = tokenized_inputs['input_ids'].to(self.device)
        attention_mask = tokenized_inputs['attention_mask'].to(self.device)
        # Prepare target labels
        target_labels = input_ids.clone().to(self.device)
        target_labels[attention_mask == 0] = -100
        # Create a TensorDataset
        dataset = TensorDataset(input_ids, attention_mask, target_labels)
        # Create a DataLoader to yield batches
        tensor_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feature_value_dict = {mia_method.name: []}
        for ((input_ids_batch, attention_mask_batch, target_labels_batch), text) in tqdm(
                zip(tensor_data_loader, data_loader)):
            # Forward pass through the model
            if mia_method.type == "gray":
                if mia_method.name in ["Loss", "MinK", "MinKPlus", "Perpelexity", "Gradient"]:
                    outputs = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                                         labels=target_labels_batch)
                    feature_value_dict[mia_method.name].extend(
                        mia_method.feature_compute(outputs[1], input_ids_batch, attention_mask_batch,
                                                   target_labels_batch, self.tokenizer))
                elif mia_method.name == "Refer":
                    outputs = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                                         labels=target_labels_batch)
                    refer_tokenized = mia_method.refer_tokenizer(text, return_tensors="pt", truncation=True,
                                                                 padding=True,
                                                                 max_length=mia_method.refer_model.config.max_position_embeddings)
                    refer_input_ids = refer_tokenized['input_ids'].to(self.device)
                    refer_attention_mask = refer_tokenized['attention_mask'].to(self.device)
                    refer_target_labels = refer_input_ids.clone().to(self.device)
                    refer_outputs = mia_method.refer_model(input_ids=refer_input_ids,
                                                           attention_mask=refer_attention_mask,
                                                           labels=refer_target_labels)
                    feature_value_dict[mia_method.name].extend(
                        mia_method.feature_compute(outputs[1], input_ids_batch, attention_mask_batch,
                                                   target_labels_batch, self.tokenizer,
                                                   refer_outputs[1], refer_input_ids, refer_attention_mask,
                                                   refer_target_labels))
                elif mia_method.name == "EDAPAC":
                    feature_value_dict[mia_method.name].extend(
                        mia_method.feature_compute(text, self.model, self.tokenizer))
                elif mia_method.name == "Recall":
                    feature_value_dict[mia_method.name].extend(
                        mia_method.feature_compute(text, self.model, self.tokenizer))
            else:
                feature_value_dict[mia_method.name].extend(
                    mia_method.feature_compute(self.model, input_ids_batch, attention_mask_batch, target_labels_batch,
                                               self.tokenizer))
        return feature_value_dict







