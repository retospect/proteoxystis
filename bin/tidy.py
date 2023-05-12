#! python

# Opens pdb.toml and converts it to a Python dictionary
# Then it processes each top level entry, and extracts the solvent condition.
# It shows a tqdl progress bar during processing
# There is a short option that only processes the first 1000 top level keys.
# These are not consistently written, so we have to try varios methods.
# It then writes a new training.toml file with the solvent conditions and the original sequence.

import argparse
import toml
import numpy as np
import tqdm
import sys
import gzip
import re
import datetime


class PdbParseException(Exception):
    pass


def getSequence(pdbid, pdb):
    # Get the value
    value = pdb[pdbid]
    l = len(value["sequence"])
    if l > 7000:
        raise PdbParseException(
            f"ERROR 53: [{key}] Sequence length {l} is too long".format(key=pdbid, l=l)
        )
    if l < 15:
        raise PdbParseException(
            f"ERROR 54: [{pdbid}] Sequence length {l} is too short".format()
        )
    # If sequence contains whitespace, something is wrong
    if re.search(r"\s", value["sequence"]):
        raise PdbParseException(
            f"ERROR 56: [{pdbid}] Sequence contains whitespace".format(pdbid)
        )
    return value["sequence"]


def parseEntry(value, pdbid):
    solvent = None
    # Try to get the solvent condition
    crco = value
    values = {}

    # Case insensitive match with regexp: "Conditions: .... PH: 7.0"
    if crco is not None:
        # This is all a big hack, and needs to be moved into a library with unit tests.

        # Extract the PH from PH 7.00, or pH 7.00
        m = re.search(r"PH:?\s+([0-9.]+)", crco, re.IGNORECASE)
        # check if it is a float, and store in the ph variable
        if m is not None:
            # remove trailing period.
            try:
                m = re.sub(r"\.$", "", m.group(1))
                ph = float(m)
                values["ph"] = ph
            except ValueError:
                raise PdbParseException(
                    f"ERROR 55: [{pdbid}] Could not parse PH from crystal conditions: {crco}".format()
                )

        # Extract the matthews coefficient from "38.45 MATTHEWS COEFFICIENT"
        m = re.search(r"([0-9.]+)\s+MATTHEWS COEFFICIENT", crco, re.IGNORECASE)
        # check that we have it, and store in the matthews variable
        if m is not None:
            matthews = float(m.group(1))
            values["matthews"] = matthews
        else:
            values["no_matthews"] = 1

        # extract the peg number of the form PEG 1000
        m = re.search(r"PEG\s+([0-9]+)", crco, re.IGNORECASE)
        if m is not None:
            peg = int(m.group(1))
            values["peg_length"] = peg

        # extract the PEG percentage and length from 26% PEG 6000
        m = re.search(r"([0-9]+)%\s+PEG\s+([0-9]+)", crco, re.IGNORECASE)
        if m is not None:
            peg = int(m.group(2))
            values["peg_length"] = peg
            values["peg_percent"] = int(m.group(1))

        # extract the names and the numbers in the format of "CRYSTALLIZATION CONDITIONS: 15MM MES, 0.5M NACL, 20MM BME"
        m = re.search(r"CRYSTALLIZATION CONDITIONS: (.*)", crco, re.IGNORECASE)
        if m is not None:
            # split on comma
            for c in m.group(1).split(","):
                # These segments can be of the form
                # "15MM MES"
                # "0.5M NACL"
                # "20MM BME"
                try:
                    # get the number and name for "15.3MM MES"
                    doublette = re.search(
                        r"([0-9.]+)\s+MM\s+([A-Z ]+)", c, re.IGNORECASE
                    )
                    if doublette is not None:
                        number = float(doublette.group(1))
                        name = doublette.group(2)
                        values[f"{name}_mM"] = number

                    # get the number and name for "15.3M MES"
                    doublette = re.search(
                        r"([0-9.]+)\s+M\s+([A-Z ]+)", c, re.IGNORECASE
                    )
                    if doublette is not None:
                        number = float(doublette.group(1))
                        name = doublette.group(2)
                        values[f"{name}_mM"] = number * 1000

                    # "0.5 M NACL"
                    doublette = re.search(
                        r"([0-9.]+)\s+M\s+([A-Z ]+)", c, re.IGNORECASE
                    )
                    if doublette is not None:
                        number = float(doublette.group(1))
                        name = doublette.group(2)
                        values[f"{name}_mM"] = number * 1000
                    # "0.5 MM NACL"
                    doublette = re.search(
                        r"([0-9.]+)\s+MM\s+([A-Z ]+)", c, re.IGNORECASE
                    )
                    if doublette is not None:
                        number = float(doublette.group(1))
                        name = doublette.group(2)
                        values[f"{name}_mM"] = number
                except ValueError:
                    raise PdbParseException(
                        f"ERROR 57: [{pdbid}] Could not parse crystallization conditions (doublette error): {crco}".format()
                    )

        # Get number for "VM (ANGSTROMS**3/DA): 2.70"
        m = re.search(r"VM \(ANGSTROMS\*\*3/DA\):\s+([0-9.]+)", crco, re.IGNORECASE)
        if m is not None:
            vm = float(m.group(1))
            values["vm_A_pwr_DA"] = vm

    # if there are fewer than 4 interesting things in the values hash, skip the writing and throw an error
    if len(values) < 4:
        raise PdbParseException(
            f"ERROR 58: [{pdbid}] Is not interesting, less than 4 values found: {crco}"
        )

    # if any value is greater or smaller than what numpy.half can hold, continue
    for val in values.values():
        if val >= np.finfo(np.half).max:
            raise PdbParseException(
                f"ERROR 59: [{pdbid}] Is not interesting, value too large: {crco}".format()
            )
        if val <= np.finfo(np.half).min:
            raise PdbParseException(
                f"ERROR 60: [{pdbid}] Is not interesting, value too small: {crco}".format()
            )

    # make sure all spaces in the key of the values hash are replaced by underscore

    # make sure all keys are lowercase
    values = {k.lower(): v for k, v in values.items()}

    values = {k.replace("tetrahydrate ", ""): v for k, v in values.items()}
    values = {k.replace("pentahydrate ", ""): v for k, v in values.items()}
    values = {k.replace("trihydrate ", ""): v for k, v in values.items()}
    values = {k.replace("bihydrate ", ""): v for k, v in values.items()}
    values = {k.replace("dihydrate ", ""): v for k, v in values.items()}
    values = {k.replace("hydrate ", ""): v for k, v in values.items()}
    values = {k.replace("buffering solution ", ""): v for k, v in values.items()}
    values = {k.replace("buffering ", ""): v for k, v in values.items()}
    values = {k.replace("buffer ", ""): v for k, v in values.items()}
    values = {k.replace(" +", " "): v for k, v in values.items()}
    values = {k.replace("-", "_"): v for k, v in values.items()}

    values = {k.replace(" ", "_"): v for k, v in values.items()}

    for key in values.keys():
        if "__" in key:
            raise PdbParseException(
                f"ERROR 61: [{pdbid}] Is problematic, key contains __ "
            )

    return values


def main(datafile, outfile):
    # Open the pdb.toml file
    with gzip.open(datafile, "rt") as f:
        print("Loading data...", end="", flush=True)
        pdb = toml.load(f)
        print("done")

    print("processing")
    err = open("tidy.err", "wt")

    # Sequence lenght for some stats
    records_processed = 0

    # Open the training.toml file
    with open(outfile, "wt") as f:
        # Write the header
        f.write("# This file is generated by tidy.py\n")
        f.write("# Generated on {date}\n".format(date=datetime.datetime.now()))
        f.write(
            "# It contains the sequence and solvent condition for each PDB entry that passes muster.\n"
        )
        # The set of all hash keys
        key_count = {}
        # Process each top level key
        print("Writing training.toml")
        seqlen = []
        for pdbid in tqdm.tqdm(pdb.keys()):
            try:
                sequence = getSequence(pdbid, pdb)
                values = parseEntry(pdb[pdbid]["crystal_conditions"], pdbid)
                seqlen.append(len(sequence))

                sequence = pdb[pdbid]["sequence"]
                f.write(f"[{pdbid}]\n")
                f.write(f'sequence="{sequence}"\n')

                # make the toml string for the values hash
                values_str = ", ".join([f"{k}={v}" for k, v in values.items()])
                f.write(f"values = {{{values_str}}}\n")
                # increment key count
                for k in values.keys():
                    if k not in key_count:
                        key_count[k] = 0
                    key_count[k] += 1
                records_processed += 1
            except PdbParseException as e:
                err.write(str(e) + "\n")
                continue
        f.write("\n")
        f.write("[metadata]\n")

        # count the key counts that are greater than 10 and collect them with their count in sig_key_count
        key_count_gt_10 = 0
        sig_key_count = {}
        for k, v in key_count.items():
            if v > 10:
                key_count_gt_10 += 1
                sig_key_count[k] = v
        f.write(
            f"sig_key_count = {{{', '.join([f'{k}={v}' for k, v in sig_key_count.items()])}}}\n"
        )
        # write the key count hash
        f.write(
            f"key_count = {{{', '.join([f'{k}={v}' for k, v in key_count.items()])}}}\n"
        )

        # write sequence length mean, median and stddev
        f.write(f"seqlen_mean = {np.mean(seqlen)}\n")
        f.write(f"seqlen_median = {np.median(seqlen)}\n")
        f.write(f"seqlen_stddev = {np.std(seqlen)}\n")
        f.write(f"record_count = {records_processed}\n")

        f.write(f"sig_key_count_gt_10 = {key_count_gt_10}\n")

    # print a list of all the significant keys to the sig_keys.csv file
    with open("sig_keys.csv", "w") as f:
        for k in sig_key_count.keys():
            f.write(f"{k}\n")


def test():
    print("Use nose2 to run tests")


if __name__ == "__main__":
    # Argparse ommandline options:
    # --test runs the test suite
    # --quick runs the truncated test suite
    # no option runs the main function

    parser = argparse.ArgumentParser(description="Tidy up the pdb.toml file")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument(
        "--quick", action="store_true", help="Run the truncated test suite"
    )
    args = parser.parse_args()

    if args.test:
        test()
    else:
        if args.quick:
            main("pdb-few.toml.gz", "training-few.toml")
        else:
            main("pdb.toml.gz", "training.toml")
