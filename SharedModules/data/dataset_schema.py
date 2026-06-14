"""dataset_schema.py — SINGLE SOURCE OF TRUTH for per-dataset CSV schema.

Both the vocabulary generator (MotifBreakdown/generate_vocab_rules.py) and the
training data loader (SharedModules/data/loader.py) must agree on which CSV
column holds the label for each dataset. Previously each file kept its own copy
of this mapping and they DISAGREED for Fluoride_Carbonyl ('label' vs
'Fluoride_Carbonyl'), which could silently train/evaluate on a different target
than the vocabulary was built against. This module centralizes the mapping so
the two phases can never drift apart again.

DATASET_COLUMN[dataset] -> name of the CSV column to read as the label
                           (None for datasets whose label is provided out-of-band,
                            e.g. OGB / TUDataset).
TASK_TYPE[dataset]      -> 'BinaryClass' | 'Regression' | 'MultiLabel'

NOTE on Fluoride_Carbonyl: it is a synthetic benzene-style benchmark, built the
same way as Benzene and Alkane_Carbonyl, whose fold CSVs use a generic 'label'
column. It is therefore mapped to 'label' here (the value the vocab generator
already used and the value that makes it consistent with its sibling datasets).
If your Fluoride_Carbonyl CSVs genuinely use a native 'Fluoride_Carbonyl'
column, change the single line below — it now updates both phases at once.
"""
from typing import Dict, Optional

DATASET_COLUMN: Dict[str, Optional[str]] = {
    # real datasets — native column name
    'Mutagenicity':      'Mutagenicity',
    'BBBP':              'BBBP',
    'hERG':              'hERG',
    'Lipophilicity':     'Lipophilicity',
    'esol':              'esol',
    'tox21':             'tox21',
    'freesolv':          'freesolv',
    # synthetic / benchmark datasets — generic 'label' column
    'Benzene':           'label',
    'Alkane_Carbonyl':   'label',
    'Fluoride_Carbonyl': 'label',   # <- unified (was 'Fluoride_Carbonyl' in loader.py)
    # OGB: label provided directly by the OGB dataset object
    'ogbg-molhiv':       None,
    'ogbg-molbace':      None,
    'ogbg-molbbbp':      None,
    'ogbg-molclintox':   None,
    'ogbg-moltox21':     None,
    'ogbg-molsider':     None,
    'ogbg-molesol':      None,
    'ogbg-molfreesolv':  None,
    'ogbg-mollipo':      None,
    # TUDataset mutag: label from PKL
    'mutag':             None,
}

TASK_TYPE: Dict[str, str] = {
    'Mutagenicity':      'BinaryClass',
    'BBBP':              'BinaryClass',
    'hERG':              'BinaryClass',
    'Benzene':           'BinaryClass',
    'Alkane_Carbonyl':   'BinaryClass',
    'Fluoride_Carbonyl': 'BinaryClass',
    'Lipophilicity':     'Regression',
    'esol':              'Regression',
    'freesolv':          'Regression',
    'tox21':             'MultiLabel',
    'ogbg-molhiv':       'BinaryClass',
    'ogbg-molbace':      'BinaryClass',
    'ogbg-molbbbp':      'BinaryClass',
    'ogbg-molclintox':   'MultiLabel',
    'ogbg-moltox21':     'MultiLabel',
    'ogbg-molsider':     'MultiLabel',
    'ogbg-molesol':      'Regression',
    'ogbg-molfreesolv':  'Regression',
    'ogbg-mollipo':      'Regression',
    'mutag':             'BinaryClass',
}
