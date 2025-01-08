from torch.nn import CrossEntropyLoss
import zlib
import torch
from torch.nn import functional as F
class MIA:
    def __init__(self, name):
        self.name = name

class LossMIA(MIA):
    """
    This class computes the loss of the model on the input.
    """
    def __init__(self):
        super().__init__("Loss")

    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
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

    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
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

class GradientMIA(MIA):
    """
    This class computes the gradient of the loss with respect to the model parameters.
    """
    def __init__(self):
        super().__init__("Gradient")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
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
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
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
    def __init__(self):
        super().__init__("MinK")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer, k=0.2):
        batch_input_ids = tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
        target_labels = tokenized_inputs["input_ids"].clone()
        target_labels[tokenized_inputs["attention_mask"] == 0] = -100
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
            caculate_length = int(length * 0.2) if length > 5 else length
            front_values = sorted_mink_plus[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_plus_avg.append(avg)
            front_values = sorted_mink[i, :caculate_length]
            avg = torch.mean(front_values.float()).item()
            batch_mink_avg.append(avg)
        return batch_mink_avg


class MinKPlusMIA(MIA):
    def __init__(self):
        super().__init__("MinKPlus")

    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer, k=0.2):
        batch_input_ids = tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
        target_labels = tokenized_inputs["input_ids"].clone()
        target_labels[tokenized_inputs["attention_mask"] == 0] = -100
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
            caculate_length = int(length * 0.2) if length > 5 else length
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
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

class DCPDDMIA(MIA):
    def __init__(self):
        super().__init__("DCPDD")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

class SaMIA(MIA):
    def __init__(self):
        super().__init__("SA")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

class PACMIA(MIA):
    def __init__(self):
        super().__init__("PAC")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

