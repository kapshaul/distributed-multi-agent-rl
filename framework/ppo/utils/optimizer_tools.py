def layer_lr(optimizer, model, target_layer_name, lr):
    """
    Finds the optimizer param_group containing the layer with name `target_layer_name`
    and updates its learning rate to `new_lr`.
    """
    found = False

    for group in optimizer.param_groups:
        for param in group['params']:
            for name, model_param in model.named_parameters():
                if param is model_param and target_layer_name in name:
                    group['lr'] = lr
                    found = True
                    print(f"Updated learning rate of '{target_layer_name}' layer to {lr}")
                    return  # Exit after first match (assumes unique layer name)

    if not found:
        print(f"Layer '{target_layer_name}' not found in any optimizer param_group.")
