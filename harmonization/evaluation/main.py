import os
import json
import csv
import torch
from .clickme import evaluate_clickme
from ..common import load_clickme_val

def collect_transform_as_dict(compose_obj):
    transform_list = []
    for transform in compose_obj.transforms:
        if isinstance(transform, transforms.Resize):
            transform_list.append({'type': 'Resize', 'size': transform.size})
        elif isinstance(transform, transforms.CenterCrop):
            transform_list.append({'type': 'CenterCrop', 'size': transform.size})
        elif isinstance(transform, transforms.ToTensor):
            transform_list.append({'type': 'ToTensor'})
        elif isinstance(transform, transforms.Normalize):
            transform_list.append({'type': 'Normalize', 'mean': f'{transform.mean}', 'std': f'{transform.std}'})
        elif isinstance(transform, transforms.Compose):
            transform_list.append({'type': 'Compose', 'composition': collect_transform_as_dict(transform)})
        else:
           transform_list.append({'type': f'{type(transform)}'})
    return transform_list


def evaluate_model(model,               # the model itself
                    model_name,         # e.g. "alexnet"     -> for results documentation
                    model_source,       # e.g. torchvision   -> for results documentation
                    model_backend,      # e.g. pytorch       -> important to select correct evaluator
                    model_transform,    # the raw transformation without conversion to PIL, takes an image
                    ROOT_RESULTS_DIR,   # path to results folder
                    document_transforms=True):  
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if model_backend == 'pytorch':
        model = model.to(device)
        model.eval() 
        print(f"moved pytorch model to device {device}")
        
    scores = evaluate_clickme(model, 
                                model_backend = model_backend,
                                clickme_val_dataset = load_clickme_val(batch_size=32),
                                model_transform = model_transform,
                                device = device)

    '''
    write detailed scores into a json file
    '''
    scores_file_path = os.path.join(ROOT_RESULTS_DIR, model_source + '_' + model_name + '.json')
    if os.path.exists(scores_file_path):
        # append a number if file exists, but that really should not happen! 
        print(f"\n WARNING: FILE {scores_file_path} ALREADY EXISTS \n")
        duplicate_marker = 1
        while os.path.exists(scores_file_path):
            duplicate_marker += 1
            if duplicate_marker > 3: #tolerate max. 3 duplicates (name, name2 and name3)
                raise Exception(f"error saving scores for model {model_name}")
            scores_file_path = f"{scores_file_path.split('.json')[0]}_{duplicate_marker}.json"
    
    #add documentation of preprocessing function to the json
    if document_transforms:
        scores['transforms'] = collect_transform_as_dict(model_transform)

    with open(scores_file_path, 'w') as f:
          json.dump(scores, f)

    '''
    write overal results into the results table
    '''
    results_summary = [model_name, model_source, scores['accuracy'], scores['alignment_score']]
    
    PERFORMANCE_TABLE_COLUMN_HEADERS = ['model_name', 'source', 'accuracy', 'spearman_feature_alignment']
    performance_table_path = os.path.join(ROOT_RESULTS_DIR, 'performances.csv')
    if not os.path.exists(performance_table_path):
        print("Warning: Performance table not found. New table started.")
        with open(performance_table_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(PERFORMANCE_TABLE_COLUMN_HEADERS)

    with open(performance_table_path, 'a+', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(results_summary)
    
    print(results_summary)
    del model #not sure if necessary


#TODO: a main function that automatically handles torchvision/TIMM from the command line