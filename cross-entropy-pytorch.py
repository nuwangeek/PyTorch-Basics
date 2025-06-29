# %load_ext watermark
# %watermark -a 'nuwangeek' -v -p torch

# Understanding One-Hot Encoding and Cross-Entropy Loss in PyTorch
import torch


# One-hot encoding
def one_hot(y, num_classes):
    """
    Converts a tensor of class indices to a one-hot encoded tensor.
    Args:
        y (torch.Tensor): Tensor of class indices.
        num_classes (int): Total number of classes.
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    y_one_hot = torch.zeros(y.size(0), num_classes)
    y_one_hot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_one_hot


y = torch.tensor([0, 1, 2, 2])
y_encoded = one_hot(y, 3)

print("One-hot encoded tensor:\n", y_encoded)

# Softmax
# Suppose we have some net inputs Z, where each rowis one training sample.

Z = torch.tensor(
    [[-0.3, -0.5, -0.5], [-0.4, -0.1, -0.5], [-0.3, -0.94, -0.5], [0.99, -0.88, -0.5]]
)

# Next, we convert them to "probabilities" via softmax:


def softmax(z):
    """
    Applies the softmax function to the input tensor.
    Args:
        z (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Softmax probabilities.
    """
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()


smax = softmax(Z)
print("Softmax probabilities:\n", smax)
