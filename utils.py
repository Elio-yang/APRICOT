from config import *


br_label_indices = [0, 1, 2]


def calc_accuracy(output: torch.Tensor, labels: torch.Tensor):
    """
    A simple function to calculate the accuracy
    """
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)


def calc_br_accuracy(criterion, output: torch.Tensor, labels: torch.Tensor):
    br_epoch_correct = 0
    br_epoch_loss = 0
    br_epoch_total = 0

    for label in br_label_indices:
        mask = (labels == label)
        preds = torch.argmax(output, dim=1)
        correct = torch.sum(preds[mask] == labels[mask])
        total = torch.sum(mask)

        br_epoch_correct += correct.item()
        br_epoch_total += total.item()
        br_epoch_loss += torch.mean(criterion(output[mask], labels[mask])) * br_epoch_total

    return br_epoch_loss, br_epoch_correct, br_epoch_total
