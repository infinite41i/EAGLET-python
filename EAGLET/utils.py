def sort_dict(d: dict, asc=True) -> dict:
        return dict(sorted(d.items(), key=lambda item: item[1], reverse= not asc))

def sort_labels(d: dict, asc=True) -> list:
    sorted_freqs = sort_dict(d, asc=False)#descending
    labels_sorted_by_f = list(sorted_freqs.items())
    return labels_sorted_by_f