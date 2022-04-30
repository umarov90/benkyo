import numpy as np
import re
import random

enc_mat = np.append(np.eye(4),
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], axis=0)
enc_mat = enc_mat.astype(np.bool)
mapping_pos = dict(zip("ACGTRYSWKMBDHVN", range(15)))


def encode_seq(seq):
    try:
        seq2 = [mapping_pos[i] for i in seq]
        return enc_mat[seq2]
    except:
        print(seq)
        return None


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


def parse_fasta(fasta_file):
    fasta = []
    seq = ""
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    fasta.append(encode_seq(seq))
                seq = ""
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            fasta.append(encode_seq(seq))
    return fasta


def random_dna(num, length):
    dnas = []
    for n in range(num):
        dna = ""
        acgt = ["A", "C", "G", "T"]
        for i in range(length):
            dna += random.choice(acgt)
        dnas.append(encode_seq(dna))
    return dnas
