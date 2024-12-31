from torch.nn import CrossEntropyLoss


class MIA:
    def __init__(self):
        pass

class LossMIA(MIA):
    def __init__(self):
        super().__init__()
        self.name = "Loss"

    def feature_compute(self, batch_logits, tokenized_inputs, target_labels):
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

