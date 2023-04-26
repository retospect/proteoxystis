#! python3
# Reads the PDB files and writes a toml file with the sequence and the experiment type and the solvent used if it was a crystallography experiment
# assuming the user already fetched all the PDB files with rsync
# See datawrangling.md to get exact instructions on how to get the files.
# Uses Bio.PDB to extract the sequence and the experiment type

import os
import sys
import toml
import gzip
from tqdm import tqdm
from Bio.PDB import PDBParser


def main(quick=False):
    print("Reading PDB files...")
    path = "data/pdb"
    files = os.listdir(path)
    # sort files and truncate list at 1000

    files = sorted(files)

    if quick:
        files = files[:1000]

    # Make a hash that translates 3 letter codons to 1 letter codons
    codon_hash = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "UNK": "u",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    fseq = 1
    # open the toml.gz file to write gzipped toml into
    bugger = gzip.open("pdb.err.gz", "wt")
    with gzip.open("pdb.toml.gz", "wt") as toml:
        # show a progress bar to indicate how many files have been processed

        for file in tqdm(files):
            if file.endswith(".ent.gz"):
                pdb_id = file[3:7]
                # make platform independent filename with path
                filename = os.path.join(".", path, file)
                # open and unzip the gzipped file and read the lines into an array
                with gzip.open(filename, "rb") as f:
                    file_content = f.read()
                lines = file_content.decode("utf-8").splitlines()

                # check if the file is a crystallography experiment
                experiment_type = ""
                crystal_conditions = ""

                sequence = ""

                for l in lines:
                    if l.startswith("EXPDTA"):
                        if "X-RAY" in l:
                            experiment_type = l
                    if "REMARK 280" in l:
                        crystal_conditions += l[11:]
                    # if the pattern matches "SEQRES <digits> A <3 letter codons>" then it is a sequence and we want to get the aa sequence
                    if "SEQRES" in l:
                        seqid = l[11]
                        if seqid == "A":
                            segment = l[19:].strip()
                            # Translate triple letter codons in segment to single letter codons
                            # and add them to the sequence
                            segment = segment.split()
                            for codon in segment:
                                if len(codon) == 3:
                                    try:
                                        sequence += codon_hash[codon]
                                    except KeyError:
                                        bugger.write(
                                            "ERROR 01: ["
                                            + file
                                            + "] KeyError codon "
                                            + codon
                                            + "\n"
                                        )
                                        continue
                                else:
                                    sequence += codon + " "
                        else:
                            # SEQID appart from A found; deal with later
                            bugger.write(
                                "ERROR 02: [" + file + "] SEQID in addition to A found\n"
                            )
                            continue

                # if the experiment type is not a crystallography experiment, does not have a sequence or does not have conditions, skip the file
                if crystal_conditions == "":
                    bugger.write(
                        "ERROR 03: ["
                        + file
                        + "] No crystal condition found (this may be not a crystallized protein, check)\n"
                    )
                    continue
                if sequence == "":
                    bugger.write(
                        "ERROR 04: ["
                        + file
                        + "] No protein sequence found. Something is weird here.\n"
                    )
                    continue

                # if experiment_type == '':
                #   continue

                # remove duplicate whitespaces from crystal_conditions
                crystal_conditions = " ".join(crystal_conditions.split())

                # remove double quotes in pdb_id, crystal_conditions, sequence, experiment_type, fseq
                pdb_id = pdb_id.replace('"', "")
                pdb_id = pdb_id.replace("\\", "")
                crystal_conditions = crystal_conditions.replace('"', "")
                crystal_conditions = crystal_conditions.replace("\\", "")

                # structure = parser.get_structure(pdb_id, filename)
                # write to toml file
                toml.write(f"[{pdb_id}]\n")
                toml.write(f'crystal_conditions = "{crystal_conditions}"\n')
                toml.write(f'sequence = "{sequence}"\n')
                toml.write(f'experiment_type = "{experiment_type}"\n')
                toml.write(f"fseq = {fseq}\n")
                toml.write("\n")
                fseq += 1

    bugger.close()


if __name__ == "__main__":
    main()
