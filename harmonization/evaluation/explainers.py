import torch
import numpy as np
from xplique.attributions import Saliency
from torchvision import transforms

def torch_saliency_explainer(batch, 
                            labels,
                            model,
                            model_preprocess,
                            device = 'cpu'):
    #TODO: check that device is cpu or contains cuda
    original_shape = xbatch.shape
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
    if saliency.shape != original_shape:
        saliency, _ = torch.max(batch.grad.data.abs(), dim=1)
        restore_shape_PIL = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                                    transforms.ToTensor()
                                        ])
        saliency_PIL = torch.stack([restore_shape_PIL(image) for image in saliency]).squeeze(1)
        restore_shape_classic = transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        saliency_classic = torch.stack([restore_shape_classic(image.unsqueeze(0)).squeeze(0) for image in saliency])


        saliency_classic = saliency_classic.to('cpu')
        saliency_PIL = saliency_PIL.to('cpu')
        saliency_maps_PIL = np.array(saliency_PIL)
    else:
        saliency_classic = saliency.to('cpu')
        saliency_maps_PIL = None
    
    saliency_maps_classic = np.array(saliency_classic)
    logits = logits.to('cpu').detach()
    
    return saliency_maps_classic, saliency_maps_PIL, logits


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

    return saliency_maps, None, logits