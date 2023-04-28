from math import log10


# extract a python hash of active values from the numpy array
def get_active_values(row, keylist, means):
    active_values = {}
    certainties = {}
    for j, key in enumerate(keylist):
        if row[j * 2] > 0.25:
            active_values[key] = row[j * 2 + 1] * means[key]
            certainties[key] = row[j * 2]
            if key == "ph":
                # convert back to pH by taking the log10
                # active_values[key] = log10(active_values[key])
                pass
    return active_values, certainties
