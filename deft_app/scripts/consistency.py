from collections import defaultdict


def check_grounding_dict(grounding_dict):
    """Check that a grounding_dict doesn't have the same longform mapping
    to different groundings in different grounding maps.
    """
    return check_dictionaries(grounding_dict.values())


def check_consistency_names_grounding_dict(grounding_dict, names_map):
    """Check that a grounding dict and names map have consistent names
    """
    groundings = {grounding for grounding_map in grounding_dict.values()
                  for grounding in grounding_map.values()
                  if grounding != 'ungrounded'}
    return groundings == set(names_map.keys())


def check_consistency_grounding_dict_pos_labels(grounding_dict, pos_labels):
    """Check that there are no pos labels not in the grounding dict
    """
    groundings = {grounding for grounding_map in grounding_dict.values()
                  for grounding in grounding_map.values()
                  if grounding != 'ungrounded'}
    return set(pos_labels) <= groundings


def check_model_consistency(model, grounding_dict, pos_labels):
    """Check that serialized model is consistent with associated json files.
    """
    groundings = {grounding for grounding_map in grounding_dict.values()
                  for grounding in grounding_map.values()}
    model_labels = set(model.estimator.named_steps['logit'].classes_)
    consistent_labels = groundings == model_labels

    shortforms = set(grounding_dict.keys())
    model_shortforms = set(model.shortforms)
    consistent_shortforms = shortforms == model_shortforms

    pos_labels = set(pos_labels)
    model_pos_labels = set(model.pos_labels)
    consistent_pos_labels = pos_labels == model_pos_labels

    model_labels = set(model.estimator.named_steps['logit'].classes_)

    return consistent_labels and consistent_shortforms and \
        consistent_pos_labels


def check_names_consistency(names_list):
    """Ensure names maps are consistent for model with multiple shortforms
    """
    return check_dictionaries(names_list)


def check_dictionaries(dicts):
    """Check if a list of dictionaries are pairwise consistent

    Two dictionaries are consistent with eachother if there does not exist a
    key k that has a different associated value in each dictionary
    """
    big_dict = defaultdict(list)
    for dictionary in dicts:
        for key, value in dictionary.items():
            big_dict[key].append(value)
    lengths = [len(value) for value in big_dict.values]
    return max(lengths) == 1

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
