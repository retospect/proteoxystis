

# extract a python hash of active values from the numpy array
def get_active_values(row, keylist):
    active_values = {}
    for j, key in enumerate(keylist):
        if row[j * 2] >0.25:
            active_values[key] = row[j * 2 + 1]
    return active_values
