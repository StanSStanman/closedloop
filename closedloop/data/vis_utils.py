import pandas as pd
import os.path as op


def load_aparc(labels):
    aparc_fname = op.join(op.dirname(__file__), 'aparc.xlsx')
    df = pd.read_excel(aparc_fname, index_col=None, usecols=range(5),
                       engine='openpyxl')
    ord_labels = dict(lobe=[], roi=[], label=[], color=[])
    for _, l in df.iterrows():
        if l['roi'] in labels:
            ord_labels['lobe'].append(l['lobe'])
            ord_labels['roi'].append(l['roi'])
            ord_labels['label'].append(l['label'])
            ord_labels['color'].append(l['color'])

    return ord_labels


def scaling(x):
    a = 13 / 71
    b = 3 - a
    y = (a * x) + b
    return y
