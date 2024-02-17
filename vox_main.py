import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
os.environ["TOKENIZERS_PARALLELISM"]= 'false'
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
import glob
import sys
from dataset import * 
from torch.utils.data import DataLoader
from metrics import compute_cls_metrics, compute_reg_metrics
from vox_model import * 
from dgllife.utils import EarlyStopping, Meter,RandomSplitter
from prettytable import PrettyTable
from tqdm import tqdm 
# from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from dgllife.utils import smiles_to_bigraph,WeaveAtomFeaturizer,CanonicalBondFeaturizer,CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer,AttentiveFPBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer 
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
from scipy import stats
import numpy as np 
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit import Chem, DataStructs
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.smallmol.smallmol import SmallMol
import pickle
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from adan import * 
def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def obtain_vox(pdbfile, voxel_size=24):
    try:
        prot = SmallMol(pdbfile,force_reading=True, fixHs=False)      # 类操作小分子结构
         # 计算分子对象边界网格中体素的原子属性描述符。
        vox, centers, N = getVoxelDescriptors(prot,voxelsize=1, buffer=1,center=(0,0,0),boxsize=(voxel_size,voxel_size,voxel_size))
        nchannels = vox.shape[1]
            # Reshape Voxels
        vox_t = vox.transpose().reshape([1, nchannels, N[0], N[1], N[2]]) 
    except:
        vox_t = np.zeros( ( 1, 8, voxel_size, voxel_size, voxel_size ) )
    return vox_t


 

def compute_cls_metrics(y_true, y_prob):
    
    y_pred = np.array(y_prob) >= 0.5
   
    roc_auc = roc_auc_score(y_true, y_prob)
   
    F1 = f1_score(y_true, y_pred, average = 'binary')
   
    mcc = matthews_corrcoef(y_true, y_pred)

    aupr = average_precision_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    

    return   F1, roc_auc, aupr,  recall, acc 
 
 
 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
 
def read_graph(input, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
     
    G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
    # for edge in G.edges():
    #     G[edge[0]][edge[1]]['weight'] = 1     
    G = G.to_undirected()

    return G
'''--------'''
def get_data(dataset):

    if dataset == 'OncologyScreen':

        mol_features = pd.read_csv('./final_data/OncologyScreen/drug_f64.csv', header=None,low_memory=False).values
        cell_features = pd.read_csv('./final_data/OncologyScreen/cell_f64.csv', header=None,low_memory=False).values
        drug_synergy = pd.read_csv('./final_data/OncologyScreen/drug_combination_processed.csv')
        drug2protein =  pd.read_csv('./data/OncologyScreen/drug_protein.csv')
        cell2protein =  pd.read_csv('./data/OncologyScreen/cell_protein.csv')
        cell2id =  pd.read_csv('./data/OncologyScreen/cell_id.csv', header=None, names=['cell', 'id'])
    else:
        mol_features = pd.read_csv('./final_data/DrugComb/drug_f64.csv', header=None,low_memory=False).values
        cell_features = pd.read_csv('./final_data/DrugComb/cell_f64.csv', header=None,low_memory=False).values
        drug_synergy = pd.read_csv('./final_data/DrugComb/drug_combinations_processed.csv')
        drug2protein =  pd.read_csv('./data/DrugComb/drug_protein.csv')
        cell2protein =  pd.read_csv('./data/DrugComb/cell_protein.csv')
        cell2id =  pd.read_csv('./data/DrugComb/cell_id.csv', header=None, names=['cell', 'id'])


    drug2protein = dict(zip(drug2protein['drug'],  drug2protein['protein'] ))
    cell2protein = dict(zip(cell2protein['cell'],  cell2protein['protein'] ))
    cell2id = dict(zip(cell2id['cell'],  cell2id['id'] ))
    
    G = read_graph( './ppi.Full.edgelist' )
    # print(list(G.edges())[0:2])
    proteinid2index = dict(zip(G.nodes(), range(G.number_of_nodes())))


    # proteinid2index =  pd.read_csv('./ppi.Full.edgelist', sep='\t', header=None, names=['id1', 'id2'])
    # unique_proteinid =  list(set( list(proteinid2index['id1'].unique()) + list(proteinid2index['id2'].unique()) )) #.sort()
    # unique_proteinid.sort()
    # # print(unique_proteinid)
    # # proteinid2index = { int(cell2id[key]) : ppi_feature[v-1] for (key, v) in cell2protein.items()}
    # proteinid2index =  {  pid : index for (index, pid) in enumerate(unique_proteinid)  }
     
    with open('./ppi.npy', 'rb') as f:
        ppi_feature = np.load( f ) # ppi_feature [15970 x 256]

    
    cell_features = {}
    for key, v in cell2protein.items():
        print(key, v, cell2id[key])
        cell_features[ int(cell2id[key])  ] = ppi_feature[ proteinid2index[v]   ]
        
     

    '''以上是cell的embedding，data'''

    mol_features = {}
    drug2vox = {}     
    with open(f'./data/{dataset}/drug_id.csv', 'r') as f:
            for s in f:
                a,b = s.strip().split(',')
                drug2vox[int(b)] =  obtain_vox( f'druginfo/{a}.pdb'  )
                mol_features[ int(b) ] = ppi_feature[    proteinid2index[ drug2protein[a] ]  ]
    '''drug的vox，以及drug的ppi'''

    synergy = [[row[7], row[8],  row[9], float(row[10])] for index, row in
               drug_synergy.iterrows()]

    
    cell_gene = load_obj( f'{dataset}_cell_gene' )
    return synergy,  drug2vox, mol_features, cell_features, cell_gene
# mol_features 是drug的info，764个文件
  
'''---load_data---'''
def process_data(synergy, drug2smiles, cline2vec, mut2vec):
    processed_synergy = []
    # 将编号转化为对应的smile和细胞系向量
    for row in synergy:
        processed_synergy.append([drug2smiles[row[0]], drug2smiles[row[1]],
                                  cline2vec[row[2]],  mut2vec[row[2]],  float(row[3])])

    

    return np.array(processed_synergy, dtype=object)

 

def run_a_train_epoch(device, epoch,num_epochs, model, data_loader, loss_criterion, optimizer, scheduler):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    # print(len(data_loader))

 

    aux_crt = nn.MSELoss()
    for id,  (*x, y) in tbar:

        for i in range(len(x)):
            x[i] = x[i].float().to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output, aux_loss = model(*x)
        
        main_loss =  loss_criterion(output.view(-1), y.view(-1))   #+   10 *  aux_loss
        loss =  main_loss


        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        # scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}  AUX_loss={aux_loss.item()  :.3f}')
    

def run_an_eval_epoch(model, data_loader, criterion):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        preds =  torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].float().to(device)
            y = y.to(device)
            logits , _ =  model(*x)
            # logits = (logits1 + logits2)/2
            loss = loss_criterion(logits.view(-1), y.view(-1))
            
            logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    val_loss =  running_loss.get_average()
    return preds, trues, val_loss
 
def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def train_valid_split(synergy, rd_seed=0):
    synergy = np.array(synergy)
    train_size = 0.9
 
    synergy = pd.DataFrame(synergy)
    synergy_cv_data, synergy_test = np.split(np.array(synergy.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy))])

    return synergy_cv_data, synergy_test

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

if __name__ == '__main__':
    mode = 2
    if mode == 1:
        dataset_name = 'OncologyScreen'  # or ONEIL or # ALMANAC-COSMIC
        print(dataset_name)
         
    else:

        dataset_name = 'DrugComb'  # or ONEIL or # ALMANAC-COSMIC
        print(dataset_name)
         
   
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    synergy,  drug2vox, mol_features, cell_features, cell_gene = get_data(dataset_name)
    # drug的协同作用info，drug的vox embedding，drug的ppi embedding（疑似模型的参数），cell的ppi embedding，cell_gene data
    # drug的协同作用indo包括drug1s, drug2s, celllines, labels,
    # 即drug1s, drug2s, celllines, labels,drug的vox embedding，drug的ppi embedding（疑似模型的参数），cell的ppi embedding，cell_gene data
    seed = 42
    seed_everything(seed)

    lr =1e-5
    
        
    t_tables = PrettyTable(['method', 'F1', 'AUC', 'AUPR', 'Recall', 'Acc'  ])
        
            
    t_tables.float_format = '.3'   
    synergy = np.array(synergy)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf_count = 0
    for train_index, validation_index in kf.split(synergy):
        kf_count = kf_count + 1
        # if kf_count < 4:
        #     continue
        # ---construct train_set+validation_set
        synergy_train, synergy_test = synergy[train_index], synergy[validation_index]
        synergy_train, synergy_validation = train_valid_split(synergy_train)
        
        trn_ds = DrugComBVoxDataset(synergy_train[:,0], synergy_train[:,1], synergy_train[:,2], synergy_train[:,3],  drug2vox, mol_features, cell_features, cell_gene)
        val_ds = DrugComBVoxDataset(synergy_validation[:,0], synergy_validation[:,1], synergy_validation[:,2],   synergy_validation[:,3], drug2vox, mol_features, cell_features, cell_gene)
        test_ds = DrugComBVoxDataset(synergy_test[:,0], synergy_test[:,1], synergy_test[:,2], synergy_test[:,3] , drug2vox, mol_features, cell_features, cell_gene)



        train_loader = DataLoader(trn_ds, batch_size= BATCH_SIZE,   shuffle=True, num_workers=8 )
        test_loader = DataLoader(test_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=8  )
        valid_loader = DataLoader(val_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=8  )
            
        
        model =  DDS3DSMILESDescriptorNetV3(  ).to(device)  # 若干卷积+mlp+att+若干线性
       
        # optimizer = torch.optim.AdamW(model.parameters(), lr )

        optimizer = Adan(
                model.parameters(),
                lr = 5e-5,                  # learning rate (can be much higher than Adam, up to 5-10x)
                betas = (0.02, 0.08, 0.01), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
                weight_decay = 0.02        # weight decay 0.02 is optimal per author
            )
            
        stopper = EarlyStopping(mode='higher', patience=25, filename='models/cap-vox-tran'  + dataset_name )
        num_epochs = 500
            
        loss_criterion = nn.BCEWithLogitsLoss() # 二分类交叉熵损失函数
        # loss_criterion =    FocalLoss()
        for epoch in range(num_epochs):
            # Train
            run_a_train_epoch(device, epoch,num_epochs, model, train_loader, loss_criterion, optimizer, None)
            # Validation and early stop
            val_pred, val_true, val_loss = run_an_eval_epoch(model, valid_loader, loss_criterion)
            
            
            e_tables = PrettyTable(['epoch', 'F1', 'AUC', 'AUPR', 'Recall', 'Acc' ])
            F1, roc_auc, aupr, recall, acc = compute_cls_metrics(val_true,val_pred)
            row = [epoch, F1, roc_auc, aupr, recall, acc]
            

            early_stop = stopper.step(roc_auc, model)
            e_tables.float_format = '.3' 
            
            e_tables.add_row(row)
            print(e_tables)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        print('Test---------------')
        test_pred, test_y, test_loss= run_an_eval_epoch(model, test_loader, loss_criterion)
        
            
        F1, roc_auc, aupr, recall, acc = compute_cls_metrics(test_y,test_pred)
        row = [ 'test', F1, roc_auc, aupr, recall, acc]
    
        
        t_tables.add_row(row)
        print(t_tables)
            
        
        results_filename = 'result/v-'  +  dataset_name+ '.csv'
        ptable_to_csv(t_tables, results_filename)

df = pd.read_csv(results_filename)
print(df.describe())
