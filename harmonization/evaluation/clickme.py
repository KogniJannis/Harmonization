"""
Module related to the click-me dataset
"""

import tensorflow as tf
import numpy as np
import torch

from ..common import load_clickme_val
from .metrics import spearman_correlation, dice, intersection_over_union
from .explainers import tensorflow_explainer, torch_saliency_explainer

from torchvision import transforms
from torchvision.transforms import InterpolationMode

HUMAN_SPEARMAN_CEILING = 0.65753
AUTO = tf.data.AUTOTUNE


# output a compose object that only crops and resizes to adjust heatmaps (either model gradient maps or the human heatmaps)
def filter_resize_crop(compose_obj):
    filtered_transforms = []
    for transform in compose_obj.transforms:
        if isinstance(transform, transforms.Resize) \
            or isinstance(transform, transforms.CenterCrop) \
            or isinstance(transform, transforms.ToPILImage) \
            or isinstance(transform, transforms.ToTensor):
            
            if isinstance(transform, transforms.CenterCrop):
                print(f"Detected crop to size {transform.size}x{transform.size}")
            if isinstance(transform, transforms.Resize):
                print(f"Detected resize to size {transform.size}x{transform.size}")
            
            filtered_transforms.append(transform)
        elif isinstance(transform, transforms.Compose):
          filtered_transforms.append(filter_resize_crop(transform))
    
    return transforms.Compose(filtered_transforms)


def evaluate_clickme(model, model_backend, clickme_val_dataset = None,
                     model_transform = None,
                     device = 'cpu', tf_input_size=224):
    """
    Evaluates a model on the Click-me validation set.

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate.
    explainer : callable, optional
        The explainer to use, by default use Xplique (tensorflow) Saliency maps.
        To define your own explainer, the function must take a batch of images and labels
        and return a saliency maps for each inputs, e.g. `f(images, labels) -> saliency_maps`.
    preprocess_inputs : function, optional
        The preprocessing function to apply to the inputs.
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    scores : dict
        The Human Alignements metrics (Spearman, Dice, IoU) on the Click-me validation set.
    """
    if clickme_val_dataset is None:
        print("WARNING: NO DATASET PROVIDED \n LOAD WITH DEFAULT BATCH SIZE")
        clickme_val_dataset = load_clickme_val()

    if model_transform is None:
        # default to identity
        print("\n WARNING: SET PREPROCESS TO IDENTITY \n")
        preprocess_inputs = lambda x : x
    elif model_backend == 'pytorch':
        print("Using pytorch backend and non-identity transforms, checking for resizes and crops...")
        preprocess_inputs = transforms.Compose([transforms.ToPILImage(), model_transform])
        preprocess_heatmaps = transforms.Compose([transforms.ToPILImage(mode="F"), model_transform])
        preprocess_heatmaps = filter_resize_crop(preprocess_heatmaps)
    elif model_backend == 'tensorflow':
        print(f"Backend is tensorflow and transform size is set to {tf_input_size}")
        preprocess_inputs = model_transform #pass-through of the tensorflow transform object
        preprocess_heatmaps = transforms.Resize(tf_input_size, interpolation=InterpolationMode.BILINEAR, antialias=True)

    #NOTE this is how preprocessing was done for tensorflow models:
    #for pytorch models there was no dataset level preprocessing and each batch was preprocessed individually in the explainer
    #right now I attempt moving this into the batch
    #clickme_val_dataset = clickme_val_dataset.map(lambda x, y, z: (preprocess_inputs(x), y, z),
    #                                               num_parallel_calls=AUTO)

    if model_backend == 'tensorflow':
        print("using tensorflow explainer")
        explainer = tensorflow_explainer
    elif model_backend == 'pytorch':
        print('using pytorch explainer')
        explainer = torch_saliency_explainer
    else:
        raise Exception(f"Backend {model_backend} not implemented")
    

    metrics = {
        'spearman': [],
        'dice': [],
        'iou': [],
        'correctness': [],
        'confidence': [],
    }
    iteration = 0
    for images_batch, heatmaps_batch, label_batch in clickme_val_dataset:
        if iteration % 50 == 0:
            print(f"processed {iteration} batches")
        iteration += 1
        
        saliency_maps, logits = explainer(images_batch, label_batch, model, preprocess_inputs, device)

        if model_transform is not None:
            heatmaps_batch = torch.stack([torch.tensor(preprocess_heatmaps(x)) for x in
                                heatmaps_batch.numpy()])
            heatmaps_batch = heatmaps_batch.permute(0, 2, 3, 1)
            heatmaps_batch = np.array(heatmaps_batch)

        #from this point onwards its assumed that heatmaps and model gradient maps have the same size, though it may not be 224x224
        if len(saliency_maps.shape) == 4:
            saliency_maps = tf.reduce_mean(saliency_maps, -1)
        if len(heatmaps_batch.shape) == 4:
            heatmaps_batch = tf.reduce_mean(heatmaps_batch, -1)

        spearman_batch = spearman_correlation(saliency_maps, heatmaps_batch)
        dice_batch = dice(saliency_maps, heatmaps_batch)
        iou_batch = intersection_over_union(saliency_maps, heatmaps_batch)

        probas = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probas, axis=1)
        label_indices = tf.argmax(label_batch, axis=1)
        is_choice_correct = tf.cast(tf.equal(predicted_indices, label_indices), dtype=tf.float32)
        correctness_batch = is_choice_correct.numpy()
        confidence_batch = tf.gather_nd(probas, tf.stack((tf.range(tf.shape(label_batch)[0]), tf.cast(label_indices, tf.int32)), axis=1)).numpy()
        metrics['spearman'] += spearman_batch.tolist()
        metrics['dice']     += dice_batch.tolist()
        metrics['iou']      += iou_batch.tolist()
        metrics['correctness'] += correctness_batch.tolist()
        metrics['confidence'] += confidence_batch.tolist()
    print("calculating aggregate metrics...")
    # add the score used in the paper: normalized spearman correlation
    metrics['alignment_score'] = str(np.mean(metrics['spearman']) / HUMAN_SPEARMAN_CEILING)
    metrics['accuracy'] = str(np.mean(metrics['correctness']))

    return metrics
