"""
Module related to the click-me dataset
"""

import tensorflow as tf
import numpy as np
import torch

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
        print("WARNING: NO DATASET PROVIDED \n LOAD WITH DEFAULT BATCH SIZE")
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
        'spearman_PIL': [],
        'dice_PIL': [],
        'iou_PIL': [],
    }
    iteration = 0
    for images_batch, heatmaps_batch, label_batch in clickme_val_dataset:
        if iteration % 50 == 0:
            print(f"processed {iteration} batches")
        iteration += 1
        saliency_maps, saliency_maps_PIL, logits = explainer(images_batch, label_batch, model, preprocess_inputs, device)

        if len(saliency_maps.shape) == 4:
            saliency_maps = tf.reduce_mean(saliency_maps, -1)
        if len(heatmaps_batch.shape) == 4:
            heatmaps_batch = tf.reduce_mean(heatmaps_batch, -1)

        spearman_batch = spearman_correlation(saliency_maps, heatmaps_batch)
        dice_batch = dice(saliency_maps, heatmaps_batch)
        iou_batch = intersection_over_union(saliency_maps, heatmaps_batch)

        if saliency_maps_PIL is not None:
            spearman_batch_PIL = spearman_correlation(saliency_maps_PIL, heatmaps_batch)
            dice_batch_PIL = dice(saliency_maps_PIL, heatmaps_batch)
            iou_batch_PIL = intersection_over_union(saliency_maps_PIL, heatmaps_batch)
            metrics['spearman_PIL'] += spearman_batch_PIL.tolist()
            metrics['dice_PIL']     += dice_batch_PIL.tolist()
            metrics['iou_PIL']      += iou_batch_PIL.tolist()
            metrics['alignment_score_PIL'] = str(np.mean(metrics['spearman_PIL']) / HUMAN_SPEARMAN_CEILING)
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
