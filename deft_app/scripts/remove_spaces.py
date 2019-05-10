import os
import sys
import json


from deft import available_shortforms
from deft.modeling.classify import load_model

from deft_app.locations import DATA_PATH
from deft_app.scripts.model_to_s3 import model_to_s3


def strip_dictionary(d):
    return {a.strip(): b.strip() for a, b in d.items()}


if __name__ == '__main__':
    models_path = os.path.join(DATA_PATH, 'models')
    groundings_path = os.path.join(DATA_PATH, 'groundings')
    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        if os.path.isdir(model_path) and \
           model_name in set(available_shortforms.values()):
            names_path = os.path.join(model_path, f'{model_name}_names.json')
            with open(names_path, 'r') as f:
                names = strip_dictionary(json.load(f))
            with open(names_path, 'w') as f:
                json.dump(names, f)
            gdict_path = os.path.join(model_path,
                                      f'{model_name}_grounding_dict.json')
            with open(gdict_path, 'r') as f:
                grounding_dict = json.load(f)
            grounding_dict = {shortform: strip_dictionary(grounding_map)
                              for shortform, grounding_map in
                              grounding_dict.items()}
            model_file = os.path.join(model_path, f'{model_name}_model.gz')
            model = load_model(model_file)
            model.pos_labels = [label.strip() for label in model.pos_labels]

            for i, label in (
                    enumerate(model.estimator.named_steps['logit'].classes_)):
                model.estimator.named_steps['logit'].classes_[i] = \
                    label.strip()

            model.dump_model(model_file)

            with open(gdict_path, 'w') as f:
                json.dump(grounding_dict, f)
            for shortform in grounding_dict:
                grounding_path = os.path.join(groundings_path, shortform)
                names_path = os.path.join(grounding_path,
                                          f'{shortform}_names.json')
                with open(names_path, 'r') as f:
                    names = strip_dictionary(json.load(f))
                with open(names_path, 'w') as f:
                    json.dump(names, f)
                gmap_path = os.path.join(grounding_path,
                                         f'{shortform}_grounding_map.json')
                with open(gmap_path, 'r') as f:
                    grounding_map = strip_dictionary(json.load(f))
                with open(gmap_path, 'w') as f:
                    json.dump(grounding_map, f)
                pos_labels_path = os.path.join(grounding_path,
                                               f'{shortform}_pos_labels.json')
                with open(pos_labels_path, 'r') as f:
                    pos_labels = json.load(f)
                pos_labels = [label.strip() for label in pos_labels]
                with open(pos_labels_path, 'w') as f:
                    json.dump(pos_labels_path, f)
            model_to_s3(model_name)
