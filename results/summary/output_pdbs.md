# Output PDBs with escape scores as B factors
This Python Jupyter notebook outputs PDBs with the escape scores as B factors.

Though we will want more elaborate series of commands to codify our visualization of these RBD structures colored by escape, the series of commands below, when executed in a `PyMol` session with one of these PDBs open, will color the RBD surface according to escape scores.

For example, to normalize each structure colored by the max mut effect, we might want to have a white to red scale from 0 to 1:

     create RBD, chain E
     hide all; show cartoon, chain A; color gray20, chain A
     show surface, RBD; spectrum b, white red, RBD, minimum=0, maximum=1
     
For something like total escape, maybe we want each structure normalized to the maximum total escape in that structure, in which case we can just leave the maximum argument empty:

     create RBD, chain E
     hide all; show cartoon, chain A; color gray20, chain A
     show surface, RBD; spectrum b, white red, RBD, minimum=0
     
We write PDBs with B factors indicating the total site escape and maximum mutation escape at each site, and the same with these values normalized to the maximum for the full structure (the latter are easier to process in `Chimera`).

First, import Python modules:


```python
import collections
import copy
import os
import warnings

import Bio.PDB

import dms_variants.pdb_utils

from IPython.display import display, HTML

import pandas as pd

import yaml
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Read configuration for outputting PDBs:


```python
print(f"Reading PDB output configuration from {config['output_pdbs_config']}")
with open(config['output_pdbs_config']) as f:
    output_pdbs_config = yaml.safe_load(f)
```

    Reading PDB output configuration from data/output_pdbs_config.yaml


Make output directory:


```python
os.makedirs(config['pdb_outputs_dir'], exist_ok=True)
```

Read escape fractions and compute **total** and **maximum** escape at each site, and also the total and maximum escape at each site normalized to be between 0 and 1 for each selection:


```python
print(f"Reading escape fractions from {config['escape_fracs']}")

escape_fracs = (
    pd.read_csv(config['escape_fracs'])
    .query('library == "average"')
    .assign(site=lambda x: x['label_site'])
    .groupby(['selection', 'site'])
    .aggregate(total_escape=pd.NamedAgg(config['mut_metric'], 'sum'),
               max_escape=pd.NamedAgg(config['mut_metric'], 'max')
               )
    .reset_index()
    .assign(max_total_escape=lambda x: x.groupby('selection')['total_escape'].transform('max'),
            max_max_escape=lambda x: x.groupby('selection')['max_escape'].transform('max'),
            norm_total_escape=lambda x: x['total_escape'] / x['max_total_escape'],
            norm_max_escape=lambda x: x['max_escape'] / x['max_max_escape'])
    )

display(HTML(escape_fracs.head().to_html(index=False)))
```

    Reading escape fractions from results/escape_scores/escape_fracs.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>selection</th>
      <th>site</th>
      <th>total_escape</th>
      <th>max_escape</th>
      <th>max_total_escape</th>
      <th>max_max_escape</th>
      <th>norm_total_escape</th>
      <th>norm_max_escape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2196+2130_400</td>
      <td>331</td>
      <td>0.047778</td>
      <td>0.004191</td>
      <td>0.779822</td>
      <td>0.1913</td>
      <td>0.061268</td>
      <td>0.021908</td>
    </tr>
    <tr>
      <td>2196+2130_400</td>
      <td>332</td>
      <td>0.070106</td>
      <td>0.012930</td>
      <td>0.779822</td>
      <td>0.1913</td>
      <td>0.089900</td>
      <td>0.067590</td>
    </tr>
    <tr>
      <td>2196+2130_400</td>
      <td>333</td>
      <td>0.044737</td>
      <td>0.003998</td>
      <td>0.779822</td>
      <td>0.1913</td>
      <td>0.057368</td>
      <td>0.020899</td>
    </tr>
    <tr>
      <td>2196+2130_400</td>
      <td>334</td>
      <td>0.071901</td>
      <td>0.015000</td>
      <td>0.779822</td>
      <td>0.1913</td>
      <td>0.092202</td>
      <td>0.078411</td>
    </tr>
    <tr>
      <td>2196+2130_400</td>
      <td>335</td>
      <td>0.062080</td>
      <td>0.006922</td>
      <td>0.779822</td>
      <td>0.1913</td>
      <td>0.079608</td>
      <td>0.036184</td>
    </tr>
  </tbody>
</table>


Now map the escape metrics to the B-factors.
For sites where no mutations have escape scores:
 - In the RBD chain(s) fill the B-factor for non-normalized scores to -1 to enable collapsing to zero or callout as a a separate class, depending how we choose to color sites for different visualizations. For normalized scores, fill to 0.
 - In other chains, always fill missing B factors to 0.  


```python
for name, specs in output_pdbs_config.items():
    print(f"\nMaking PDB mappings for {name} to {specs['pdbfile']}")
    assert os.path.isfile(specs['pdbfile'])
    
    # get escape fracs just for conditions of interest
    if isinstance(specs['conditions'], str) and specs['conditions'].upper() == 'ALL':
        conditions = escape_fracs['selection'].unique().tolist()
    else:
        assert isinstance(specs['conditions'], list)
        conditions = specs['conditions']
    print(f"Making mappings for {len(conditions)} conditions.")
    df = escape_fracs.query('selection in @conditions')
    
    # get chains
    assert isinstance(specs['chains'], list)
    print('Mapping to the following chains: ' + ', '.join(specs['chains']))
    df = pd.concat([df.assign(chain=chain) for chain in specs['chains']], ignore_index=True)
    
    # make mappings for each condition and metric
    for condition, df in df.groupby('selection'):
        print(f"  Writing B-factor re-assigned PDBs for {condition} to:")
    
        for metric in ['total_escape', 'max_escape', 'norm_total_escape', 'norm_max_escape']:
        
            # what do we assign to missing sites?
            missing_metric = collections.defaultdict(lambda: 0)  # non-RBD chains always fill to zero
            for chain in specs['chains']:
                if 'norm' in metric:
                    missing_metric[chain] = 0  # missing sites in RBD are 0 for normalized metric PDBs
                else:
                    missing_metric[chain] = -1  # missing sites in RBD are -1 for non-normalized metric PDBs
        
            fname = os.path.join(config['pdb_outputs_dir'], f"{condition}_{name}_{metric}.pdb")
            print(f"    {fname}")
            
            dms_variants.pdb_utils.reassign_b_factor(input_pdbfile=specs['pdbfile'],
                                                     output_pdbfile=fname,
                                                     df=df,
                                                     metric_col=metric,
                                                     missing_metric=missing_metric)
```

    
    Making PDB mappings for 6m0j to data/pdbs/6M0J.pdb
    Making mappings for 3 conditions.
    Mapping to the following chains: E
      Writing B-factor re-assigned PDBs for 2196+2130_400 to:
        results/pdb_outputs/2196+2130_400_6m0j_total_escape.pdb
        results/pdb_outputs/2196+2130_400_6m0j_max_escape.pdb
        results/pdb_outputs/2196+2130_400_6m0j_norm_total_escape.pdb
        results/pdb_outputs/2196+2130_400_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for COV2-2130_400 to:
        results/pdb_outputs/COV2-2130_400_6m0j_total_escape.pdb
        results/pdb_outputs/COV2-2130_400_6m0j_max_escape.pdb
        results/pdb_outputs/COV2-2130_400_6m0j_norm_total_escape.pdb
        results/pdb_outputs/COV2-2130_400_6m0j_norm_max_escape.pdb
      Writing B-factor re-assigned PDBs for COV2-2196_400 to:
        results/pdb_outputs/COV2-2196_400_6m0j_total_escape.pdb
        results/pdb_outputs/COV2-2196_400_6m0j_max_escape.pdb
        results/pdb_outputs/COV2-2196_400_6m0j_norm_total_escape.pdb
        results/pdb_outputs/COV2-2196_400_6m0j_norm_max_escape.pdb
    
    Making PDB mappings for 6w41 to data/pdbs/6W41.pdb
    Making mappings for 3 conditions.
    Mapping to the following chains: C
    
    Making PDB mappings for 6xdg to data/pdbs/6xdg.pdb
    Making mappings for 3 conditions.
    Mapping to the following chains: E
    
    Making PDB mappings for 7c01 to data/pdbs/7c01_single.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A
    
    Making PDB mappings for 6wps to data/pdbs/6wps.pdb
    Making mappings for 2 conditions.
    Mapping to the following chains: A
    
    Making PDB mappings for 6xcm to data/pdbs/6xcm.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 6xcn to data/pdbs/6xcn.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, C, E
    
    Making PDB mappings for 7k90 to data/pdbs/7k90.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7k8s to data/pdbs/7k8s.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7k8t to data/pdbs/7k8t.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7k8x to data/pdbs/7k8x.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7k8y to data/pdbs/7k8y.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: B, D, E
    
    Making PDB mappings for 7k8z to data/pdbs/7k8z.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7k8v to data/pdbs/7k8v.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, C
    
    Making PDB mappings for 7K45 to data/pdbs/7k45.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: B
    
    Making PDB mappings for 7K43 to data/pdbs/7k43.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, E
    
    Making PDB mappings for 7JW0 to data/pdbs/7jw0.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: A, B, E
    
    Making PDB mappings for 7JX3 to data/pdbs/7jx3.pdb
    Making mappings for 2 conditions.
    Mapping to the following chains: R
    
    Making PDB mappings for 7jzu to data/pdbs/7jzu.pdb
    Making mappings for 1 conditions.
    Mapping to the following chains: B

