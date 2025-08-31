''' Function to set dropout
'''

import torch.nn as nn

def set_dropout(model, dropout_value):
    """
    Recursively sets the dropout rate of all Dropout layers in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        dropout_value (float): The new dropout probability.

    Returns:
        None (modifies the model in place).
    """
    for name, module in model.named_children():
        # If the module is a Dropout layer, update its probability
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = dropout_value
        else:
            # Recursively apply the function to child modules (e.g., inside nn.Sequential, submodules)
            set_dropout(module, dropout_value)
        