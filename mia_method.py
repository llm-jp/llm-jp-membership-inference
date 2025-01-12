from torch.nn import CrossEntropyLoss
import zlib
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm
import numpy as np
import pdb

class MIA:
    def __init__(self, name, type="gray"):
        self.name = name
        self.type = type

class LossMIA(MIA):
    """
    This class computes the loss of the model on the input.
    """
    def __init__(self):
        super().__init__("Loss")
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer):
        shift_logits = batch_logits[:, :-1, :].contiguous()
        labels = target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        loss_value_list = []
        for idx, i in enumerate(instance_losses):
            loss = i.sum() / sum(i != 0)
            loss_value_list.append(loss.item())
        return loss_value_list

class ZlibMIA(MIA):
    """
    This class computes the ratio of the loss to the compressed size of the input.
    """
    def __init__(self):
        super().__init__("Zlib")
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        shift_logits = batch_logits[:, :-1, :].contiguous()
        labels = target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        zlib_value_list = []
        for idx, i in enumerate(instance_losses):
            loss = i.sum() / sum(i != 0)
            zlib_value = loss.float().cpu() / (len(zlib.compress(
                bytes(tokenizer.decode(tokenized_inputs[idx], skip_special_tokens=True), "utf-8"))))
            zlib_value_list.append(zlib_value.item())
        return zlib_value_list

class ReferenceMIA(MIA):
    def __init__(self, reference_model):
        super().__init__("Refer")
        self.refer_model = AutoModelForCausalLM.from_pretrained(reference_model,
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                ).eval()
        self.refer_tokenizer = AutoTokenizer.from_pretrained(reference_model)
        self.refer_tokenizer.pad_token = self.refer_tokenizer.eos_token
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        pass

class GradientMIA(MIA):
    """
    This class computes the gradient of the loss with respect to the model parameters.
    """
    def __init__(self):
        super().__init__("Gradient")
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        shift_logits = batch_logits[:, :-1, :].contiguous()
        labels = target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        grad_value_list = []
        for idx, i in enumerate(instance_losses):
            torch.cuda.empty_cache()
            loss = i.sum() / sum(i != 0)
            loss.backward(retain_graph=True)
            grad_norms = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.detach().norm(2))
            grad_norm = torch.stack(grad_norms).mean()
            self.model.zero_grad()
            grad_value_list.append(grad_norm.item())
        return grad_value_list

class PerplexityMIA(MIA):
    def __init__(self):
        super().__init__("Perplexity")
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        shift_logits = batch_logits[:, :-1, :].contiguous()
        labels = target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        perp_value_list = []
        for idx, i in enumerate(instance_losses):
            loss = i.sum() / sum(i != 0)
            perp_value_list.append(torch.exp(loss.float()).item())
        return perp_value_list

class MinKMIA(MIA):
    def __init__(self, k=0.2):
        super().__init__("MinK")
        self.k = k
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer, k=0.2):
        batch_input_ids = tokenized_inputs[:, 1:].unsqueeze(-1)
        target_labels = tokenized_inputs.clone()
        target_labels[attention_mask == 0] = -100
        batch_probs = F.softmax(batch_logits[:, :-1].float(), dim=-1)
        batch_log_probs = F.log_softmax(batch_logits[:, :-1].float(), dim=-1)
        mask = target_labels[:, 1:] != -100
        mask = mask.unsqueeze(-1)
        batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids).squeeze(-1)
        batch_probs_masked = batch_probs.where(mask, 0)
        batch_log_probs_masked = batch_log_probs.where(mask, 0)
        batch_mu = (batch_probs_masked.float() * batch_log_probs_masked.float()).float().sum(-1)
        batch_sigma = ((batch_probs_masked.float() * torch.square(
            torch.where(batch_probs_masked > 0, batch_log_probs_masked.float(),
                        torch.tensor(0.0, device=batch_log_probs_masked.device, dtype=torch.float32)))).sum(
            dim=-1) - torch.square(batch_mu.float()).squeeze())
        mask = mask.squeeze(-1)
        batch_mink_plus = (batch_token_log_probs - batch_mu).float() * mask / batch_sigma.float().sqrt()
        token_length = mask.sum(dim=1)
        batch_mink_plus[mask == False] = torch.inf
        batch_token_log_probs[mask == False] = torch.inf
        sorted_mink_plus, _ = torch.sort(batch_mink_plus)
        sorted_mink, _ = torch.sort(batch_token_log_probs)
        batch_mink_plus_avg = []
        batch_mink_avg = []
        for i, length in enumerate(token_length):
            caculate_length = int(length * self.k) if length > 5 else length
            front_values = sorted_mink_plus[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_plus_avg.append(avg)
            front_values = sorted_mink[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_avg.append(avg)
        return batch_mink_avg

class MinKPlusMIA(MIA):
    def __init__(self, k=0.2):
        super().__init__("MinKPlus")
        self.k = k
        self.type = "gray"
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer):
        batch_input_ids = tokenized_inputs[:, 1:].unsqueeze(-1)
        target_labels = tokenized_inputs.clone()
        target_labels[attention_mask == 0] = -100
        batch_probs = F.softmax(batch_logits[:, :-1].float(), dim=-1)
        batch_log_probs = F.log_softmax(batch_logits[:, :-1].float(), dim=-1)
        mask = target_labels[:, 1:] != -100
        mask = mask.unsqueeze(-1)
        batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids).squeeze(-1)
        batch_probs_masked = batch_probs.where(mask, 0)
        batch_log_probs_masked = batch_log_probs.where(mask, 0)
        batch_mu = (batch_probs_masked.float() * batch_log_probs_masked.float()).float().sum(-1)
        batch_sigma = ((batch_probs_masked.float() * torch.square(
            torch.where(batch_probs_masked > 0, batch_log_probs_masked.float(),
                        torch.tensor(0.0, device=batch_log_probs_masked.device, dtype=torch.float32)))).sum(
            dim=-1) - torch.square(batch_mu.float()).squeeze())
        mask = mask.squeeze(-1)
        batch_mink_plus = (batch_token_log_probs - batch_mu).float() * mask / batch_sigma.float().sqrt()
        token_length = mask.sum(dim=1)
        batch_mink_plus[mask == False] = torch.inf
        batch_token_log_probs[mask == False] = torch.inf
        sorted_mink_plus, _ = torch.sort(batch_mink_plus)
        sorted_mink, _ = torch.sort(batch_token_log_probs)
        batch_mink_plus_avg = []
        batch_mink_avg = []
        for i, length in enumerate(token_length):
            caculate_length = int(length * self.k) if length > 5 else length
            front_values = sorted_mink_plus[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_plus_avg.append(avg)
            front_values = sorted_mink[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_avg.append(avg)
        return batch_mink_plus_avg

class RecallMIA(MIA):
    def __init__(self):
        super().__init__("Recall")
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        pass

class DCPDDMIA(MIA):
    def __init__(self):
        super().__init__("DCPDD")
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        pass

class SaMIA(MIA):
    def __init__(self, generation_samples=10, input_length=128, temperature=0.8, generation_batch_size=11, max_mew_tokens=128):
        super().__init__("SAMIA")
        self.config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained(
            'lucadiliello/BLEURT-20')
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
        self.bleurt_model.eval()
        self.gen_samples = generation_samples
        self.max_input_tokens = input_length
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size
        self.max_new_tokens = max_mew_tokens
        self.type = "black"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bleurt_model.to(self.device)
    def bleurt_score(self, reference, generations):
        self.bleurt_model.eval()
        with torch.no_grad():
            inputs = self.bleurt_tokenizer([reference for i in range(len(generations))], generations, max_length=512,
                               truncation=True, padding="max_length", return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            res = self.bleurt_model(**inputs).logits.flatten().tolist()
        return res
    def feature_compute(self, model, tokenized_inputs, attention_mask, target_labels, tokenizer):
        #decide the input length, if the input length is less than the max_input_tokens, then use the input length, otherwise use the max_input_tokens
        input_length = int(min(attention_mask.sum(dim=1)) / 2) if (attention_mask[0].sum() < self.max_input_tokens) else self.max_input_tokens
        full_decoded = [[] for _ in range(self.generation_batch_size)]
        for generation_idx in tqdm(range(self.generation_batch_size)):
            if generation_idx == 0:
                zero_temp_generation = model.generate(input_ids=tokenized_inputs[:, :input_length],
                                                      attention_mask=attention_mask[:,:input_length],
                                                      temperature=0,
                                                      max_new_tokens=self.max_new_tokens,
                                                      )
                decoded_sentences = tokenizer.batch_decode(zero_temp_generation, skip_special_tokens=True)
                for i in range(zero_temp_generation.shape[0]):
                    full_decoded[generation_idx].append(decoded_sentences[i])
            else:
                generations = model.generate(input_ids=tokenized_inputs[:, :input_length],
                                             attention_mask=attention_mask[:, :input_length],
                                             do_sample=True,
                                             temperature=self.temperature,
                                             max_new_tokens=self.max_new_tokens,
                                             top_k=50,
                                             )
                decoded_sentences = tokenizer.batch_decode(generations, skip_special_tokens=True)
                for i in range(zero_temp_generation.shape[0]):
                    full_decoded[generation_idx].append(decoded_sentences[i])
        samia_value_list = []
        for batch_idx in range(zero_temp_generation.shape[0]):
            refer_sentence = full_decoded[0][batch_idx]
            other_sentences = [full_decoded[i][batch_idx] for i in range(1, len(full_decoded))]
            bleurt_value = np.array(self.bleurt_score([refer_sentence], other_sentences)).mean().item()
            samia_value_list.append(bleurt_value)
        return samia_value_list

class CDDMIA(MIA):
    def __init__(self, generation_samples=10, input_length=128, temperature=0.8, generation_batch_size=11,
                 max_mew_tokens=128):
        super().__init__("CDDMIA")
        self.genertion_samples = generation_samples
        self.gen_samples = generation_samples
        self.max_input_tokens = input_length
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size
        self.max_new_tokens = max_mew_tokens

    def levenshtein_distance(self, str1, str2):
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        distances = range(len(str1) + 1)
        for index2, char2 in enumerate(str2):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(str1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
            distances = new_distances
        return distances[-1]

    def strip_code(self, sample):
        return sample.strip().split('\n\n\n')[0] if '\n\n\n' in sample else sample.strip().split('```')[0]

    def tokenize_code(self, sample, tokenizer, length):
        return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)

    def get_edit_distance_distribution_star(self, samples, gready_sample, tokenizer, length=100):
        gready_sample = self.strip_code(gready_sample)
        gs = self.tokenize_code(gready_sample, tokenizer, length)
        num = []
        max_length = len(gs)
        for sample in samples:
            sample = self.strip_code(sample)
            s = self.tokenize_code(sample, tokenizer, length)
            num.append(self.levenshtein_distance(gs, s))
            max_length = max(max_length, len(s))
        return num, max_length

    def feature_compute(self, model, tokenized_inputs, attention_mask, target_labels, tokenizer):
        input_length = int(min(attention_mask.sum(dim=1)) / 2) if (
                attention_mask[0].sum() < self.max_input_tokens) else self.max_input_tokens
        full_decoded = [[] for _ in range(self.generation_batch_size)]
        for generation_idx in tqdm(range(self.generation_batch_size)):
            if generation_idx == 0:
                zero_temp_generation = model.generate(input_ids=tokenized_inputs[:, :input_length],
                                                      attention_mask=attention_mask[:,:input_length],
                                                      temperature=0,
                                                      max_new_tokens=self.max_new_tokens,
                                                      )
                decoded_sentences = tokenizer.batch_decode(zero_temp_generation,
                                                           skip_special_tokens=True)
                for i in range(zero_temp_generation.shape[0]):
                    full_decoded[generation_idx].append(decoded_sentences[i])
            else:
                generations = model.generate(input_ids=tokenized_inputs[:, :input_length],
                                             attention_mask=attention_mask[:, :input_length],
                                             do_sample=True,
                                             temperature=self.temperature,
                                             max_new_tokens=self.max_new_tokens,
                                             top_k=50,
                                             )
                decoded_sentences = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
                for i in range(zero_temp_generation.shape[0]):
                    full_decoded[generation_idx].append(decoded_sentences[i])
        cdd_value_list = []
        for batch_idx in range(zero_temp_generation.shape[0]):
            refer_sentence = full_decoded[0][batch_idx]
            other_sentences = [full_decoded[i][batch_idx] for i in range(1, len(full_decoded))]
            dist, ml = self.get_edit_distance_distribution_star([refer_sentence], other_sentences,
                                                           tokenizer, length=1000)
            cdd_value_list.append(sum(dist)/len(dist))
        return cdd_value_list

class PACMIA(MIA):
    def __init__(self):
        super().__init__("PAC")
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,):
        pass

