from math import log10


# extract a python hash of active values from the numpy array
def get_active_values(row, keylist, factor, offset, category_is_present_magic_treshhold):
    active_values = {}
    certainties = {}
    for j, key in enumerate(keylist):
        if row[j * 2] > category_is_present_magic_treshhold:
            active_values[key] = row[j * 2 + 1] +offset[key]
            active_values[key] = row[j * 2 + 1] *factor[key]

            certainties[key] = row[j * 2]
    return active_values, certainties
