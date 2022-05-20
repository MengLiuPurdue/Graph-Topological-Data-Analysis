# Graph-Topological-Data-Analysis
This repo contains code for "Topological structure of deep learning predictions". 
### packages requirement
* pytorch
* torchvision
* scikit-learn
* numpy
* scipy
* timm
* tqdm
* torch_geometric
* networkx
* seaborn
* rembg

### To create a Reeb network

```python
from GTDA.GTDA_utils import compute_reeb, NN_model
from GTDA.GTDA import GTDA

nn_model = NN_model()
nn_model.A = G # graph to analyze
nn_model.preds = preds # prediction matrix 
```


## Swiss Roll experiment
### Files:
### Prerequisites: 
None, self contained

## Amazon Electronics experiment