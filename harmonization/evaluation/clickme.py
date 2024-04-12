"""
Module related to the click-me dataset
"""

import tensorflow as tf
import numpy as np

from ..common import load_clickme_val
from .metrics import spearman_correlation, dice, intersection_over_union
from .explainers import tensorflow_explainer, torch_saliency_explainer

HUMAN_SPEARMAN_CEILING = 0.65753
AUTO = tf.data.AUTOTUNE


def evaluate_clickme(model, model_backend, clickme_val_dataset = None,
                     preprocess_inputs = None,
                     device = 'cpu'):
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
        clickme_val_dataset = load_clickme_val()

    if preprocess_inputs is None:
        # default to identity
        print("\n WARNING: SET PREPROCESS TO IDENTITY \n")
        preprocess_inputs = lambda x : x

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

    for images_batch, heatmaps_batch, label_batch in clickme_val_dataset:

        saliency_maps, logits = explainer(images_batch, label_batch, model, preprocess_inputs, device)

        if len(saliency_maps.shape) == 4:
            saliency_maps = tf.reduce_mean(saliency_maps, -1)
        if len(heatmaps_batch.shape) == 4:
            heatmaps_batch = tf.reduce_mean(heatmaps_batch, -1)

        spearman_batch = spearman_correlation(saliency_maps, heatmaps_batch)
        dice_batch = dice(saliency_maps, heatmaps_batch)
        iou_batch = intersection_over_union(saliency_maps, heatmaps_batch)

        probas = torch.softmax(logits, dim=1)
        _, predicted = torch.max(probas, 1)
        _, labels = torch.max(label_batch, 1)
        choices = (predicted == labels).float()
        correctness_batch = choices.detach().to('cpu').numpy()
        confidence_batch = probas[:, labels].detach().to('cpu').numpy()

        metrics['spearman'] += list(spearman_batch)
        metrics['dice']     += list(dice_batch)
        metrics['iou']      += list(iou_batch)
        metrics['correct'] += list(correctness_batch)
        metrics['confidence'] += list(confidence_batch)
    
    # add the score used in the paper: normalized spearman correlation
    metrics['alignment_score'] = np.mean(metrics['spearman']) / HUMAN_SPEARMAN_CEILING
    metrics['accuracy'] = np.mean(metrics['correct'])

    return metrics
