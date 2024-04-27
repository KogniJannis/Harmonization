import torch
import numpy as np
from xplique.attributions import Saliency

def torch_saliency_explainer(batch, 
                            labels,
                            model,
                            model_preprocess,
                            device = 'cpu'):
    #TODO: check that device is cpu or contains cuda

    # convert to tensor and preprocess
    batch = torch.stack([model_preprocess(x) for x in
                            batch.numpy().astype(np.uint8)])
    labels = torch.Tensor(labels.numpy())
    batch = batch.to(device)
    labels = labels.to(device)
    batch.requires_grad_()

    logits = model(batch)

    output = torch.sum(logits * labels)
    output.backward()

    saliency, _ = torch.max(batch.grad.data.abs(), dim=1)
    # explainer need to return numpy array
    saliency = saliency.to('cpu')
    logits = logits.to('cpu').detach()
    saliency_maps = np.array(saliency)

    return saliency_maps, logits


def tensorflow_explainer(batch, 
                        labels,
                        model,
                        model_preprocess,
                        device = 'cpu'):
    batch =  model_preprocess(batch)
    try:
        explainer = Saliency(model)
    except Exception as explainer_error:
        raise ValueError(f"Backend tensorflow indicated but Xplique (tensorflow) failed to load") from explainer_error

    saliency_maps = explainer(batch, labels)
    logits = model(batch)

    return saliency_maps, logits