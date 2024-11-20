import torch


def calculate_metric(model, val_loader, device, metric = None):
    """
    Calculate metric score for DataLoader instance
    (e.g.: macro/micro precision/recall/f1 score)

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score

    metric:
        Examples of possible metrics:
        precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        f1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    Returns
    -------
    Calculated metric: float
    """

    true_labels, predicted_labels = get_true_and_predicted_labels(model, val_loader, device)
    return metric(true_labels, predicted_labels)


def get_true_and_predicted_labels(model, val_loader, device):
    """
    Get true and predicted labels

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score


    Returns
    -------
    true_labels, predicted_labels: torch.Tensor, torch.Tensor
    """

    model.to(device)
    model.eval()
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for input, labels in val_loader:
            validation_output = model(input.to(device))
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels.to(device)))

    return true_labels.cpu(), predicted_labels.cpu()


def get_predicted_labels(model, val_loader, device):
    """
    Returns predicted labels for a validation dataset.

    Args:
        model (torch.nn.Module): Trained model
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to perform computations ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Predicted labels as a tensor.
    """
    model.to(device)
    model.eval()
    predicted_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for input, labels in val_loader:
            validation_output = model(input.to(device))
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))

    return predicted_labels.cpu()