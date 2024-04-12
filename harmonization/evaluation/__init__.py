"""
Module related to the Human Alignment benchmark
"""

from .clickme import load_clickme_val, evaluate_clickme
from .metrics import dice, spearman_correlation, intersection_over_union
from .main import evaluate_model
from .explainers import torch_saliency_explainer, tensorflow_explainer