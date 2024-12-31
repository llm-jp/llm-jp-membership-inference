from torch.nn import CrossEntropyLoss
import zlib
import torch

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
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

class MinKPlusMIA(MIA):
    def __init__(self):
        super().__init__("MinKPlus")
    def feature_compute(self, batch_logits, tokenized_inputs, target_labels, tokenizer):
        pass

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

