from random import random

def sort_dict_by_value(d: dict, desc=False) -> dict:
        return dict(sorted(d.items(), key=lambda item: item[1], reverse= desc))

def sort_labels(d: dict, desc=False) -> list:
    sorted_freqs = sort_dict_by_value(d, desc)
    labels_sorted_by_f = list(sorted_freqs.items())
    return labels_sorted_by_f

def decision(probability):
    return random() < probability