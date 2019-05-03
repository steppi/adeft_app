import os
import json
import logging
from collections import defaultdict

from deft.modeling.classify import load_model

from deft_app.locations import DATA_PATH

logger = logging.getLogger(__file__)


def check_dictionaries(dicts):
    """Check if a list of dictionaries are pairwise consistent

    Two dictionaries are consistent if there does not exist a key k that
    has a different associated value in each dictionary.
    """
    big_dict = defaultdict(list)
    for dictionary in dicts:
        for key, value in dictionary.items():
            big_dict[key].append(value)
    lengths = [len(value) for value in big_dict.values]
    return max(lengths) == 1


def check_consistency(model_name):
    """This checks if all model files are consistent with each other.

    This is horribly written and will be refactored soon. Just trying to
    get it working for now.
    """
    model_path = os.path.join(DATA_PATH, 'models', model_name)
    with open(os.path.join(model_path,
                           f'{model_name}_grounding_dict.json'), 'r') as f:
        grounding_dict = json.load(f)
    with open(os.path.join(model_path,
                           f'{model_name}_names.json'), 'r') as f:
        model_names = json.load(f)
        groundings = {grounding for grounding_map in grounding_dict.values()
                      for grounding in grounding_map.values()
                      if grounding != 'ungrounded'}
        if groundings != set(model_names.keys()):
            logger.warning('inconsistent groundings in names map and'
                           ' grounding maps for model.')
            return False
    with open(os.path.join(model_path,
                           f'{model_name}_pos_labels.json'), 'r') as f:
        pos_labels = json.load(f)
    if not (set(pos_labels) <= groundings):
        logger.warning('positive labels exist that are not in list of'
                       ' groundings')
        return False
    model = load_model(os.path.join(model_path, f'{model_name}_model.gz'))
    classes = model.estimator.named_steps['logit'].classes_
    if sorted(pos_labels) != sorted(classes):
        logger.warning('labels for deft classifier out of sync with labels'
                       ' in file.')
    names_list = []
    local_pos_labels = set()
    for shortform, gmap1 in grounding_dict:
        with open(os.path.join(DATA_PATH, 'groundings', shortform,
                               f'{shortform}_grounding_map.json'), 'r') as f:
            gmap2 = json.load(f)
        if gmap1 != gmap2:
            logger.warning('inconsistent grounding maps for model.')
            return False
        with open(os.path.join(DATA_PATH, 'groundings', shortform,
                               f'{shortform}_names.json'), 'r') as f:
            names_list.append(json.load(f))
        with open(os.path.join(DATA_PATH, 'groundings', shortform,
                               f'{shortform}_pos_labels.json'), 'r') as f:
            local_pos_labels.update(json.load(f))
    if set(pos_labels) != local_pos_labels:
        logger.warning('positive labels have got out of sync between model'
                       ' and pre-training grounding files.')
        return False
    if not check_dictionaries(names_list):
        logger.warning('inconsistent names dictionaries for different'
                       ' shortforms in model with multiple shortforms')
        return False
    return True
