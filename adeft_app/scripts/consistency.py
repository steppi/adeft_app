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
    consistent_labels = groundings <= model_labels

    shortforms = set(grounding_dict.keys())
    model_shortforms = set(model.shortforms)
    consistent_shortforms = shortforms == model_shortforms

    model_labels = set(model.estimator.named_steps['logit'].classes_)
    consistent_pos_labels = set(pos_labels) <= model_labels
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
    big_dict = defaultdict(set)
    for dictionary in dicts:
        for key, value in dictionary.items():
            big_dict[key].add(value)
    lengths = [len(value) for value in big_dict.values()]
    return max(lengths) <= 1
