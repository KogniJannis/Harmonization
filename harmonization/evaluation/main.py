import os
import json
import csv
import torch
from .clickme import evaluate_clickme
from ..common import load_clickme_val
from torchvision import transforms
from torchvision.transforms import InterpolationMode

#resizes the image so that is has the shape of the final resize/crop without ever cropping
def adjust_transform(compose_obj, size_found=False):
    filtered_transforms = []

    for transform in reversed(compose_obj.transforms):
        if isinstance(transform, transforms.Resize) and not size_found:
            # Add the first found Resize (which is the last one applied) to the new list
            filtered_transforms.append(transform)
            size_found = True
        elif isinstance(transform, transforms.CenterCrop) and not size_found:
            # if a crop is found, resize instead
            #NOTE: its important that Resize gets the tuple (H,W) from transform.size here, so that resulting ratio matches
            filtered_transforms.append(transforms.Resize(transform.size, interpolation=InterpolationMode.BILINEAR, antialias=True))
            size_found = True
        elif isinstance(transform, transforms.Compose):
            #recursively handle nested compose
            adjusted_compose, size_found = adjust_transform(transform, size_found)
            filtered_transforms.append(adjusted_compose)
        elif not isinstance(transform, transforms.Resize) and not isinstance(transform, transforms.CenterCrop):
            # Add all non-Resize/non-CenterCrop transforms
            filtered_transforms.append(transform)

    # Reverse the list again to get the original order
    filtered_transforms.reverse()
    
    return transforms.Compose(filtered_transforms), size_found


def collect_transform_as_dict(compose_obj):
    transform_list = []
    for transform in compose_obj.transforms:
        if isinstance(transform, transforms.Resize):
            transform_list.append({'type': 'Resize', 'size': transform.size})
        elif isinstance(transform, transforms.CenterCrop):
            raise Exception("should not contain a crop")
        elif isinstance(transform, transforms.ToTensor):
            transform_list.append({'type': 'ToTensor'})
        elif isinstance(transform, transforms.Normalize):
            transform_list.append({'type': 'Normalize', 'mean': f'{transform.mean}', 'std': f'{transform.std}'})
        elif isinstance(transform, transforms.Compose):
            transform_list.append({'type': 'Compose', 'composition': collect_transform_as_dict(transform)})
        else:
           transform_list.append({'type': f'{type(transform)}'})
    return transform_list


def evaluate_model(model,                       # the model itself
                    model_name,                 # e.g. "alexnet"     -> for results documentation
                    model_source,               # e.g. torchvision   -> for results documentation
                    model_backend,              # e.g. pytorch       -> important to select correct evaluator
                    model_transform,            # the raw transformation without conversion to PIL, takes an image
                    ROOT_RESULTS_DIR,           # path to results folder
                    document_transforms=True,   # whether to put the transforms in the results json
                    batch_size=32,
                    tf_input_size=None):           #necessary for tensorflow models  
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if model_backend == 'pytorch':
        model = model.to(device)
        model.eval() 
        print(f"moved pytorch model to device {device}")
        print('Original Transform:')
        print(model_transform)
        model_transform, _ = adjust_transform(model_transform)
        print('Adjusted Transform to avoid cropping:')
        print(model_transform)
    
    elif model_backend == 'tensorflow':
        assert tf_input_size is not None, "when evaluating with tensorflow an input size to resize heatmaps must be provided"
        print("WARNING: make sure your model has no cropping in its preprocessing!!")
        print(f"Evaluating with tensorflow and setting input size to {tf_input_size}")
    
    else:
        raise Exception(f"resizing analysis currentlynot supported for {model_backend}")
    
    

    scores = evaluate_clickme(model, 
                                model_backend = model_backend,
                                clickme_val_dataset = load_clickme_val(batch_size=batch_size),
                                model_transform = model_transform,
                                device = device,
                                tf_input_size=tf_input_size)
    
    '''
    write detailed scores into a json file
    '''
    scores_file_path = os.path.join(ROOT_RESULTS_DIR, model_source + '_' + model_name + '_resized_map.json')
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
        if model_backend == 'pytorch':
            scores['transforms'] = collect_transform_as_dict(model_transform)
        else:
            print(f"WARNING: cannot document transforms for backend {model_backend}")
            scores['transforms'] = f"{model_backend} transform"
    
    with open(scores_file_path, 'w') as f:
          json.dump(scores, f)

    '''
    write overal results into the results table
    '''
    results_summary = [model_name, model_source, scores['accuracy'], scores['alignment_score']]
    
    PERFORMANCE_TABLE_COLUMN_HEADERS = ['model_name', 'source', 'accuracy', 'spearman_feature_alignment']
    performance_table_path = os.path.join(ROOT_RESULTS_DIR, 'serre_resized_maps_performances.csv')
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