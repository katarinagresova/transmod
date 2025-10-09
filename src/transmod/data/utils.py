from collections import defaultdict
import itertools
import numpy as np
import pandas as pd

def extract_features(df, features, include_fold=False):

    extracted_features = {}

    if "length" in features:

        if "cds_sequence" not in df.columns:
            df['cds_sequence'] = df.apply(extract_cds_sequence, axis=1)
        if "utr5_sequence" not in df.columns:
            df['utr5_sequence'] = df.apply(extract_utr5_sequence, axis=1)
        if "utr3_sequence" not in df.columns:
            df['utr3_sequence'] = df.apply(extract_utr3_sequence, axis=1)

        extracted_features['log10 tx length'] = df['tx_sequence'].apply(len).apply(np.log10)
        extracted_features['log10 cds length'] = df['cds_sequence'].apply(len).apply(np.log10)
        extracted_features['log10 utr5 length'] = df['utr5_sequence'].apply(len).apply(np.log10)
        extracted_features['log10 utr3 length'] = df['utr3_sequence'].apply(len).apply(np.log10)

    if "cds3mers" in features:
        if "cds_sequence" not in df.columns:
            df['cds_sequence'] = df.apply(extract_cds_sequence, axis=1)
        kmer_freqs = df['cds_sequence'].apply(kmer_frequencies, k=3, overlap=True)
        kmer_df = pd.DataFrame(kmer_freqs.tolist()).add_prefix('cds_3mer_')
        extracted_features.update(kmer_df.to_dict(orient='list'))

    if "5utr3mers" in features:
        if "utr5_sequence" not in df.columns:
            df['utr5_sequence'] = df.apply(extract_utr5_sequence, axis=1)

        kmer_freqs_5utr = df['utr5_sequence'].apply(kmer_frequencies, k=3, overlap=True)
        kmer_df_5utr = pd.DataFrame(kmer_freqs_5utr.tolist()).add_prefix('5utr_3mer_')
        extracted_features.update(kmer_df_5utr.to_dict(orient='list'))

    if "3utr3mers" in features:
        if "utr3_sequence" not in df.columns:
            df['utr3_sequence'] = df.apply(extract_utr3_sequence, axis=1)

        kmer_freqs_3utr = df['utr3_sequence'].apply(kmer_frequencies, k=3, overlap=True)
        kmer_df_3utr = pd.DataFrame(kmer_freqs_3utr.tolist()).add_prefix('3utr_3mer_')
        extracted_features.update(kmer_df_3utr.to_dict(orient='list'))

    if include_fold and 'fold' in df.columns:
        extracted_features['fold'] = df['fold']

    return pd.DataFrame(extracted_features)

def extract_cds_sequence(row):
    """
    Extract the CDS sequence from a DataFrame row.

    Parameters:
    row (pd.Series): A row from a pandas DataFrame containing 'tx_sequence', 'utr5_size', and 'cds_size'.

    Returns:
    str: The extracted CDS sequence.
    """
    start = row['utr5_size']
    end = start + row['cds_size']
    return row['tx_sequence'][start:end]

def extract_utr5_sequence(row):
    """
    Extract the 5' UTR sequence from a DataFrame row.

    Parameters:
    row (pd.Series): A row from a pandas DataFrame containing 'tx_sequence' and 'utr5_size'.

    Returns:
    str: The extracted 5' UTR sequence.
    """
    end = row['utr5_size']
    return row['tx_sequence'][:end]

def extract_utr3_sequence(row):
    """
    Extract the 3' UTR sequence from a DataFrame row.

    Parameters:
    row (pd.Series): A row from a pandas DataFrame containing 'tx_sequence', 'utr5_size', and 'cds_size'.

    Returns:
    str: The extracted 3' UTR sequence.
    """
    start = row['utr5_size'] + row['cds_size']
    return row['tx_sequence'][start:]

def gc_content(sequence):
    """
    Calculate the GC content of a nucleotide sequence.

    Parameters:
    sequence (str): A string representing the nucleotide sequence.

    Returns:
    float: The GC content as a fraction of the total sequence length.
    """
    if not sequence:
        return 0.0
    gc_count = sum(1 for base in sequence if base in 'GCgc')
    return gc_count / len(sequence)

def log_length(sequence):
    """
    Calculate the log10 of the length of a sequence.

    Parameters:
    sequence (str): A string representing the nucleotide sequence.

    Returns:
    float: The log10 of the sequence length.
    """
    import math
    length = len(sequence)
    return math.log10(length) if length > 0 else 0.0

def kmer_frequencies(sequence, k=2, overlap=True, bases='ATGC'):
    """
    Calculate the k-mer frequencies in a nucleotide sequence.

    Parameters:
    sequence (str): A string representing the nucleotide sequence.
    k (int): The length of the k-mers to consider.
    overlap (bool): Whether to consider overlapping k-mers.
    bases (str): The valid nucleotide bases to consider.

    Returns:
    dict: A dictionary with k-mers as keys and their frequencies as values.
    """

    if len(sequence) < k or k <= 0:
        return {}

    kmer_counts = defaultdict(int)
    step = 1 if overlap else k
    total_kmers = len(sequence) - k + 1 if overlap else len(sequence) // k

    for i in range(0, len(sequence) - k + 1, step):
        kmer = sequence[i:i + k]
        if len(kmer) == k:
            kmer_counts[kmer] += 1

    # Convert counts to frequencies
    kmer_frequencies = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}

    # Ensure all possible k-mers are represented, even if frequency is 0
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    for kmer in all_kmers:
        if kmer not in kmer_frequencies:
            kmer_frequencies[kmer] = 0.0
    
    return kmer_frequencies

