from .conv_layers import (
    GINConv,
    GCNConvWithAtten,
    GATConvWithAtten,
    SAGEConvWithAtten,
    PNAConvSimple,
    create_conv_layers,
    CONV_FACTORIES,
)
from .gnn_base import BaseGNN

# Aliases so any code that imported the old names still works
GCNConv  = GCNConvWithAtten
GATConv  = GATConvWithAtten
SAGEConv = SAGEConvWithAtten
PNAConv  = PNAConvSimple

__all__ = [
    'GINConv', 'GCNConvWithAtten', 'GATConvWithAtten',
    'SAGEConvWithAtten', 'PNAConvSimple',
    'GCNConv', 'GATConv', 'SAGEConv', 'PNAConv',  # aliases
    'create_conv_layers', 'CONV_FACTORIES',
    'BaseGNN',
]
