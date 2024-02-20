from torch.utils.data import Dataset
import requests
import numpy as np 
from tqdm import tqdm 
import torch 
import pandas as pd 

 

from rdkit import Chem
import dgl.backend as F
from rdkit.Chem import AllChem
from dgl import save_graphs, load_graphs
from joblib import Parallel, delayed, cpu_count
import json
from dgllife.utils import smiles_to_bigraph,WeaveAtomFeaturizer,CanonicalBondFeaturizer
from functools import partial
from torch.utils.data import DataLoader
import dgl.backend as F
import dgl


 


class DrugComBVoxDataset(Dataset):
    def __init__(self, drug1s, drug2s, celllines, labels, drug2vox, molfeatures, cellfeatures, cell_gene):
         
        self.labels =  labels 
        self.length = len(self.labels)
        self.drug1s = drug1s
        self.drug2s = drug2s 
        self.celllines = celllines
        self.molfeatures = molfeatures
        self.cellfeatures = cellfeatures
        self.drug2vox = drug2vox
        self.cell_gene = cell_gene


   
    def __getitem__(self, idx):
        label = self.labels[idx]
        drug1id = self.drug1s[idx]
        drug2id = self.drug2s[idx]
        cellid = self.celllines[idx]
        # drug1_features =  self.molfeatures[ int(drug1id)-1]
        # drug2_features =  self.molfeatures[ int(drug2id)-1]
        vox1_features =  self.drug2vox[ int(drug1id)]
        vox2_features =  self.drug2vox[ int(drug2id)]
        # cf = self.cellfeatures [  int(cellid)-1 ]
        drug1_features =  self.molfeatures[ int(drug1id)]
        drug2_features =  self.molfeatures[ int(drug2id)]
        cf = self.cellfeatures [  int(cellid)  ]
        cell_gf = self.cell_gene[ int(cellid) ]
        # print(type(drug1_features), type(vox1_features), type(cf))

         
        return drug1_features, drug2_features, vox1_features,vox2_features, cf, cell_gf,  torch.FloatTensor([label]) 
 

    def __len__(self):
        return self.length


class DrugComBAGFPPPIDataset(Dataset):
    def __init__(self, drug1s, drug2s, celllines, labels, drug2agfp, molfeatures, cellfeatures, cell_gene):
        self.labels =  labels 
        self.length = len(self.labels)
        self.drug1s = drug1s
        self.drug2s = drug2s 
        self.celllines = celllines
        self.molfeatures = molfeatures
        self.cellfeatures = cellfeatures
        self.drug2agfp = drug2agfp
        self.cell_gene = cell_gene


   
    def __getitem__(self, idx):
        label = self.labels[idx]
        drug1id = self.drug1s[idx]
        drug2id = self.drug2s[idx]
        cellid = self.celllines[idx]
        # drug1_features =  self.molfeatures[ int(drug1id)-1]
        # drug2_features =  self.molfeatures[ int(drug2id)-1]
        agfp1_features =  self.drug2agfp[ int(drug1id)]
        agfp2_features =  self.drug2agfp[ int(drug2id)]
        # cf = self.cellfeatures [  int(cellid)-1 ]
        drug1_features =  self.molfeatures[ int(drug1id)]
        drug2_features =  self.molfeatures[ int(drug2id)]
        cf = self.cellfeatures [  int(cellid)  ]
        cell_gf = self.cell_gene[ int(cellid) ]
        # print(type(drug1_features), type(vox1_features), type(cf))

         
        return drug1_features, drug2_features, agfp1_features,agfp2_features, cf, cell_gf,  torch.FloatTensor([label]) 
 

    def __len__(self):
        return self.length
 
