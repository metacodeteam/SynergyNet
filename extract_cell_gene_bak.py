import pandas as pd
import numpy as np
import pickle
import os



# Load Gene Attribute Matrix
gene_attribute_matrix = pd.read_table('/home/wjj/data/drugsyn/gene_attribute_matrix_standardized.txt', header=[0,1], index_col=[0,1])
# Drop un-needed indexes/headers
gene_attribute_matrix.index = gene_attribute_matrix.index.droplevel(1)
gene_attribute_matrix.columns = gene_attribute_matrix.columns.droplevel(1)
# Put column and index names in the right place
gene_attribute_matrix.index.name = gene_attribute_matrix.index[0]
gene_attribute_matrix.drop(gene_attribute_matrix.index[0], axis=0, inplace=True)
gene_attribute_matrix.columns.name = gene_attribute_matrix.columns[0]
gene_attribute_matrix.drop(gene_attribute_matrix.columns[0], axis=1, inplace=True)

print(gene_attribute_matrix.head())


# sc_gene_exp_file =   "/home/wjj/data/drugsyn/cell_line_Gene_expression_CCLE.csv"
cell2id1 =  pd.read_csv('../NEXGB/data/DrugComb/cell_id.csv', header=None, names=['cell', 'id'])
cell2id2 =  pd.read_csv('../NEXGB/data/OncologyScreen/cell_id.csv', header=None, names=['cell', 'id'])
cell2id1 = dict(zip(cell2id1['cell'],  cell2id1['id'] ))
cell2id2 = dict(zip(cell2id2['cell'],  cell2id2['id'] ))

# sc_gene_exp = pd.read_csv(sc_gene_exp_file)

cellid2gene = {}
for (key, v) in cell2id1.items():
    if key not in gene_attribute_matrix.columns:
        print(key)
         
    cell_line_feat_scores = gene_attribute_matrix[key]
    cellid2gene[v] = np.array(cell_line_feat_scores, dtype='float32') 
output_file = 'DrugComb_cell_gene.pkl'
with open(output_file, 'wb') as f:
   pickle.dump(cellid2gene, f)

cellid2gene = {}
for (key, v) in cell2id2.items():
    if key not in gene_attribute_matrix.columns:
        print(key)
    cell_line_feat_scores = gene_attribute_matrix[key]
    cellid2gene[v] = np.array(cell_line_feat_scores, dtype='float32') 

output_file = 'OncologyScreen_cell_gene.pkl'
with open(output_file, 'wb') as f:
   pickle.dump(cellid2gene, f)

    
 


