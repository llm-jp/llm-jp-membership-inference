from torch.nn import CrossEntropyLoss
import zlib
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm
import numpy as np
import pdb
import random
from copy import deepcopy


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
    """
    This method calculates the loss difference between the attacked model and a reference model.
    """
    def __init__(self, reference_model="StabilityAI/stablelm-base-alpha-3b"):
        super().__init__("Refer")
        self.refer_model = AutoModelForCausalLM.from_pretrained(reference_model,
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                ).eval()
        self.refer_tokenizer = AutoTokenizer.from_pretrained(reference_model)
        self.refer_tokenizer.pad_token = self.refer_tokenizer.eos_token
        self.type = "gray"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refer_model.to(self.device)
    def feature_compute(self, batch_logits, tokenized_inputs, attention_mask, target_labels, tokenizer,
                        refer_logits, refer_tokenized_inputs, refer_attention_mask, refer_target_labels):
        shift_logits = batch_logits[:, :-1, :].contiguous()
        labels = target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        loss_value_list = []
        for idx, i in enumerate(instance_losses):
            loss = i.sum() / sum(i != 0)
            loss_value_list.append(loss.item())
        shift_logits = refer_logits[:, :-1, :].contiguous()
        labels = refer_target_labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        instance_losses = lm_loss.view(-1, shift_logits.size(1))
        refer_loss_value_list = []
        for idx, i in enumerate(instance_losses):
            loss = i.sum() / sum(i != 0)
            refer_loss_value_list.append(loss.item())
        gap_value_list = []
        for i, j in zip(loss_value_list, refer_loss_value_list):
            gap_value_list.append(i-j)
        return gap_value_list


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
    """
    This method calcualtes the perplexity of the model on the input.
    """
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
    """
    This method calculates the bottom k% low probability tokens' average log probability of a given input.
    Please refer to https://arxiv.org/abs/2310.16789 for more details.
    """
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
    """
    This method calculates the standarized bottom k% low probability tokens' average log probability of a given input.
    Please refer to https://arxiv.org/abs/2404.02936
    """
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
    """
    This method caculates the similarity between the example generated at zero temperature and the examples generated at non-zero temperature.
    The basic hypothesis is that a trained text should have a higher such similarity.
    """
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
    """
    This method calculates the edit distance between the example generated at zero temperature and the examples generated at non-zero temperature.
    The difference with SaMIA is that this method uses the edit distance as the similarity metric rather than a neural similarity metric.
    """
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

class EDAPACMIA(MIA):
    def __init__(self, alpha=0.3, num_aug=5):
        super().__init__("EDAPAC")
        self.type = "gray"
        self.alpha = alpha
        self.num_aug = num_aug

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def eda(self, sentence):
        words = sentence.split(' ')
        num_words = len(words)
        augmented_sentences = []
        if (self.alpha > 0):
            n_rs = max(1, int(self.alpha * num_words))
            for _ in range(self.num_aug):
                a_words = self.random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))
        augmented_sentences = [sentence for sentence in augmented_sentences]
        random.shuffle(augmented_sentences)
        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        return augmented_sentences

    def create_pertubation_text(self, batched_text):
        new_prompt_list = []
        for prompt in batched_text:
            newprompts = self.eda(prompt, alpha=self.aplha, num_aug=self.num_aug)
            new_prompt_list.extend(deepcopy(newprompts))
        return new_prompt_list

    def prob_collection(self, prompt, tokenizer, model):
        all_probs = []
        tokenized_inputs = tokenizer(prompt,
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
        target_labels = tokenized_inputs["input_ids"].clone().to(device)
        target_labels[tokenized_inputs["attention_mask"] == 0] = -100
        outputs = model(**tokenized_inputs, labels=target_labels)
        logits = outputs[1]
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        for example_idx in range(len(prompt)):
            example_probability = probabilities[example_idx][target_labels[example_idx] != -100]
            temp_probs = []
            for token_idx, token_id in enumerate(tokenized_inputs["input_ids"][example_idx]):
                if token_id != 0:
                    temp_probs.append(example_probability[token_idx, token_id].item())
            all_probs.append(temp_probs)
        return all_probs

    def calculate_Polarized_Distance(self, prob_list: list, ratio_local=0.3, ratio_far=0.05):
        local_region_length = max(int(len(prob_list) * ratio_local), 1)
        far_region_length = max(int(len(prob_list) * ratio_far), 1)
        local_region = np.sort(prob_list)[:local_region_length]
        far_region = np.sort(prob_list)[::-1][:far_region_length]
        return np.mean(far_region) - np.mean(local_region)


    def feature_compute(self, batched_text, model, tokenizer):
        eda_pac_collect = []
        pertubation_text = self.create_pertubation_text(batched_text)
        all_probs = self.prob_collection(batched_text, model, tokenizer)
        new_all_probs = self.prob_collection(pertubation_text, model, tokenizer)
        pds = [self.calculate_Polarized_Distance(prob_list) for prob_list in all_probs]
        new_pds = [self.calculate_Polarized_Distance(prob_list) for prob_list in new_all_probs]
        calibrated_pds = [np.mean(new_pds[i:i + self.num_aug]) for i in range(0, len(new_pds), self.num_aug)]
        eda_pac_value = np.array(pds) - np.array(calibrated_pds)
        eda_pac_collect.extend(eda_pac_value)
        return eda_pac_collect


