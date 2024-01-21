# MOGDx
[![DOI](https://zenodo.org/badge/622972427.svg)](https://zenodo.org/doi/10.5281/zenodo.10545043)
## Introduction
Multi-omic Graph Diagnosis (MOGDx) is a tool for the integration of omic data and classification of heterogeneous diseases. MOGDx exploits a Patient Similarity Network (PSN) framework to integrate omic data using Similarity Network Fusion (SNF) [^fn1]. A single PSN is built per modality. The PSN is built using the most informative features of that modality. The most informative features are found by performing a contrastive analysis between classification targets. For example, in mRNA, differentially expressed genes will be used to construct the PSN. Where suitable, Pearson correlation, otherwise Euclidean distance is measured between these informative features and the network is constructed using the K Nearest Neighbours (KNN) algorithm. SNF is used to combine individual PSN's into a single network. The fused PSN and the omic datasets are input into the Graph Convultional Network with Multi Modal Encoder, the architecture of which is shown in below. Each omic measure is compressed using a two layer encoder. The compressed encoded layer of each modality is then decoded to a shared latent space using mean pooling. This encourages each modality to learn the same latent space. The shared latent space is the node feature matrix, required for training the GCN, with each row forming a node feature vector. Classification is performed on the fused network using the Graph Convolutional Network (GCN) deep learning algorithm [^fn2]. 

GCN is a novel paradigm for learning from both network structure and node features. Heterogeneity in diseases confounds clinical trials, treatments, genetic association and more. Accurate stratification of these diseases is therefore critical to optimize treatment strategies for patients with heterogeneous diseases. Previous research has shown that accurate classification of heterogenous diseases has been achieved by integrating and classifying multi-omic data [^fn3][^fn4][^fn5]. MOGDx improves upon this research. The advantages of MOGDx is that it can handle both a variable number of modalities and missing patient data in one or more modalities. Performance of MOGDx was benchmarked on the BRCA TCGA dataset with competitive performance compared to its counterparts. In summary, MOGDx combines patient similarity network integration with graph neural network learning for accurate disease classification. 

## Workflow
### Full pipeline overview
![Code Overview](./workflow_diagrams/code_flowchart.png?raw=true)

### Pre-preocessing and Graph Generation
![R preprocess](./workflow_diagrams/pre-processing_modalities_inkscape.png?raw=true)

### Graph Convolutionl Network with Multi Modal Encoder
![Python](./workflow_diagrams/gcn-mme.png?raw=true)

## Installation
A working version of R and Python is required. R version 4.2.2 and Python version 3.9.2 was used to obtain all results.

Steps 1-4 are executed in R, while step 5 is designed to run on a cluster with a GPU but can be executed locally if there is sufficient memory availability. 

### Step 1 - Data Download
Create a folder called `data` and a folder for each dataset with naming convention 'TCGA-' e.g. 'TCGA-BRCA' inside this folder.

Use the R script `data_download.R` to download all data changing the project to BRCA/LGG/KICH/KIRC/KICH

### Step 2 - Preprocessing
Create a folder called for the dataset e.g. TCGA, and within this folder create a folder for each project.

Run the R script `Preprocessing.R` specifying the phenotypical trait and project, checking to ensure the paths point to the created `data` folder. 

Save each modalities processed folder with naming convention `modality_processed.RData`.

The options are \
BRCA : \
project = 'BRCA' \
trait = 'paper_paper_BRCA_Subtype_PAM50' 

LGG : \
project = 'LGG' \
trait = 'paper_Grade' 

Note : To create the KIPAN dataset, the KIRC, KICP and KICH datasets have to be combined. This can be achieved by copying the downloaded files 
from all three seperate datasets into a single dataset called KIPAN keeping the same naming structures. A column named 'subtype' specifying which dataset
the patient came from needs to be created in the Meta data file. A basic knowledge of R is required for this. 

KIPAN :  \
project = 'KIPAN' \
trait = 'subtype' 

### Step 3 - Graph Generation
Point the knn_graph_generation.R to the project folder containing the processed modalities.

Create a folder called raw. This is the folder from which MOGDx will be run.

Use the R script `knn_graph_generation.R` specifying the phenotypical trait, project and modalities downloaded in the for loop.

### Step 4 - SNF
Create a folder called Network outside data \
Copy each modalities `modality_graph.csv` to this folder \

Specify the modalities of interest in the list `mod_list` 

Point the SNF script to the new Network folder

Run the R script `SNF.R` 

### Example of directory structure for TCGA
- data
  - TCGA-BRCA
     - mRNA
       - mRNA.rda
     - miRNA
       - miRNA.rda
  - TCGA
    - BRCA
      - mRNA_processed.RData
      - miRNA_processed.RData
      - raw
        - datExpr_mRNA.csv
        - datMeta_mRNA.csv
        - mRNA_graph.csv
        - datExpr_miRNA.csv
        - datMeta_miRNA.csv
        - miRNA_graph.csv
        - mRNA_miRNA_graph.csv


### Step 5 - Execute MOGDx.py
Ensure all expression, meta and graph files for all modalities are in a single folder with naming convention 'modality_datExpr.csv'/'modality_datMeta.csv'/'modality_graph.csv' e.g. 'mRNA_datExpr.csv'/'mRNA_datMeta.csv'/'modality_graph.csv'. \
If performing an analysis on integrated modalities ensure all expression and meta files for the integrated modalities are in the folder. \
e.g. if analysing mRNA & miRNA, ensure mRNA_miRNA_graph.csv , datMeta_mRNA.csv, datMeta_miRNA.csv, datExpr_mRNA.csv and datExper_miRNA.csv are in the same folder. \

This process will have been done automatically by the creation of the raw folder and running of SNF.R and it is easiest to retain this folder.

MOGDx is a command line tool. A sample command is : \
`python MOGDx.py -i "/raw_data/raw_BRCA" -o "./Output/BRCA/"  -snf "mRNA_miRNA_graph.csv" --n-splits 5 -ld 32 16 --target "paper_BRCA_Subtype_PAM50" --index-col "patient" --epochs 2000 --lr 0.001 --h-feats 64 --decoder-dim 64`

-i is the location of the raw data containing all datExpr, datMeta and graph.csv files \
-o is the location where the output will be printed \
-snf is the name of the fused psn  \
--n-splits is the number of cross validation splits \
-ld is the latent dimension per modality. It is order alphabetically thus, mRNA will be compressed to dim of 32 and miRNA to 16  \
--target is the phenotype being classified \
--index-col is the column containing the patient ids  

All other arguments are related to the Graph Convolutional Network with Multi Modal Encoder

## Requirements
All requirements are specified in requirements.txt 

To create virtual env execute :  \
 `conda create --name <env> --file requirements.txt` 


## Contact
Barry Ryan 

PhD Candidate Biomedical AI, University of Edinburgh \
School of Informatics, 10 Crichton Street

Email: barry.ryan@ed.ac.uk \
LinkedIn: https://www.linkedin.com/in/barry-ryan/

## Citations
[^fn1]: Bo Wang et al. “Similarity network fusion for aggregating data types on a genomic scale”. en. In: Nature Methods 11.3 (Mar. 2014). Number: 3 Publisher: Nature Publishing Group, pp. 333–337. ISSN: 1548-7105. DOI: 10.1038/nmeth.2810. URL: https://www.nature.com/articles/nmeth.2810 (visited on 11/07/2022)
[^fn2]: Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. en. arXiv:1609.02907 [cs, stat]. Feb. 2017. URL:http://arxiv.org/abs/1609.02907 (visited on 09/26/2022).
[^fn3]: Shraddha Pai et al. “netDx: interpretable patient classification using integrated patient similarity networks”. In:
Molecular Systems Biology 15.3 (Mar. 2019). Publisher: John Wiley & Sons, Ltd, e8497. ISSN: 1744-4292. DOI: 10.15252/msb.20188497. URL: https://www.embopress.org/doi/full/10.15252/msb. 20188497 (visited on 12/05/2022).
[^fn4]: Xiao Li et al. “MoGCN: A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis”. eng. In: Frontiers in Genetics 13 (2022), p. 806842. ISSN: 1664-8021. DOI: 10.3389/fgene.2022.806842.
[^fn5]: Tongxin Wang et al. “MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification”. en. In: Nature Communications 12.1 (June 2021). Number: 1 Publisher: Nature Publishing Group, p. 3445. ISSN: 2041-1723. DOI: 10.1038/s41467-021-23774-w. URL: https://www.nature.com/articles/s41467-021-23774-w (visited on 01/26/2023).
