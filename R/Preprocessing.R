library(reshape2)
library(kableExtra)
library(plotly)
library(vsn)
library(tibble)
library(pheatmap)
library(SummarizedExperiment)

source('~/MOGDx/R/preprocess_functions.R')

setwd('~/MOGDx/') 

project <- 'BRCA'
trait <- 'paper_BRCA_Subtype_PAM50'
  
# -------------------------------------------------------------------------
# META File Generation ----------------------------------------------------
# -------------------------------------------------------------------------

# The meta data is (typically) located in the coldata of the mRNA gene expression
# experiment. 

load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))
# Create coldata and condition table --------------------------------------
coldata <- colData(data)
datMeta <- as.data.frame(coldata[,c('patient','race' , 'gender' , 'sample_type' , trait)])
datMeta <- datMeta[!(is.na(datMeta[[trait]])) , ]
datMeta <- datMeta[!(duplicated(datMeta[ , c('patient' , trait)])) , ] 
datMeta[[trait]] <- factor(datMeta[[trait]])
rownames(datMeta) <- datMeta$patient

write.csv(datMeta , file = paste0('./data/TCGA-',project,'/datMeta.csv'))

# -------------------------------------------------------------------------
# mRNA pre-processing -----------------------------------------------------
# -------------------------------------------------------------------------

# Pull in Count Matrices --------------------------------------------------
load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))
count_mtx <- assay(data)
colnames(count_mtx) <- substr(colnames(count_mtx) , 1, 12)
count_mtx <- count_mtx[, !(duplicated(colnames(count_mtx)))]

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get intersection of count and meta
common_idx <- intersect(colnames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[ , common_idx]
datMeta <- datMeta[common_idx , ]

# Perform differential expression analysis
diff_expr_res <- diff_expr(count_mtx , datMeta , trait , 500 , 'mRNA')

# Save differential expression results
datExpr <- diff_expr_res$datExpr
datMeta <- diff_expr_res$datMeta
dds <- diff_expr_res$dds
top_genes <- diff_expr_res$top_genes
save(datExpr, datMeta, dds, top_genes, file=paste0('~/MOGDx/data/TCGA-',project,'/mRNA/mRNA_processed.RData'))

# -------------------------------------------------------------------------
# miRNA preprocessing -----------------------------------------------------
# -------------------------------------------------------------------------
 
#Pull in count data
load(paste0('./data/TCGA-',project,'/miRNA/miRNA.rda'))

# Get Count Matrices and filter for reads
read_count <- data.frame(row.names = data$miRNA_ID)
read_per_million <- data.frame(row.names = data$miRNA_ID)
for (i in 2:dim(data)[2]) {
  
  if (i%%3 == 2) {
    read_count <- cbind(read_count , data[ , i] )
  }
  if (i%%3 == 0) {
    read_per_million <- cbind(read_per_million , data[ , i])
  }
}

colname_read_count <- c()
colname_read_per_million <- c()
for (i in 2:dim(data)[2]) {
  
  if (i%%3 == 2) {
    colname_read_count <- c(colname_read_count , substr(strsplit(colnames(data)[i] , '_')[[1]][3] , 1,12))
  }
  if (i%%3 == 0) {
    colname_read_per_million <- cbind(colname_read_per_million , substr(strsplit(colnames(data)[i] , '_')[[1]][6] , 1,12))
  }
}
colnames(read_count) <- colname_read_count
colnames(read_per_million) <- colname_read_per_million

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get Intersection of ID's
count_mtx <- read_count
count_mtx <- count_mtx[ , !(duplicated(colnames(count_mtx)))] 
common_idx <- intersect(colnames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[,common_idx]
datMeta <- datMeta[common_idx , ]

# Perform differential expression analysis
diff_expr_res <- diff_expr(count_mtx , datMeta , trait , 200 , 'miRNA')

# Save differential expression results
datExpr <- diff_expr_res$datExpr
datMeta <- diff_expr_res$datMeta
dds <- diff_expr_res$dds
top_genes <- diff_expr_res$top_genes
save(datExpr, datMeta, dds, top_genes, file=paste0('~/MOGDx/data/TCGA-',project,'/miRNA/miRNA_processed.RData'))

# -------------------------------------------------------------------------
# DNAm preprocessing ------------------------------------------------------
# -------------------------------------------------------------------------

# Load CpG Counts ---------------------------------------------------------
load(paste0('./data/TCGA-',project,'/DNAm/DNAm.rda'))
count_mtx <- assay(data)

to_keep = complete.cases(count_mtx) #removed 191928 cpg sites
length(to_keep) - sum(to_keep)

count_mtx <- t(count_mtx[to_keep,])
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get Intersection of ID's
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Run glmnet to get CpG's associated with phenotypes of interest ------------
phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, traits, nFolds = 10)
})

cpg_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  cpg_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    cpg_sites_tmp <- unique(c(cpg_sites_tmp ,class_coefs ))
  }
  cpg_sites[[res$trait]] <- cpg_sites_tmp
}

# Save CpG sites and expression data
datExpr <- count_mtx
save(cpg_sites , datExpr , datMeta , file = paste0('~/MOGDx/data/TCGA-',project,'/DNAm/DNAm_processed.RData'))

# ----------------------------------------------------------------------------------------
# Protein (RPPA) pre-processing ----------------------------------------------------------
# ----------------------------------------------------------------------------------------
load(paste0('./data/TCGA-',project,'/RPPA/RPPA.rda'))
count_mtx <- t(data[  , 6:ncol(data) ])
colnames(count_mtx) <- data$peptide_target
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# Load Meta Data -----------------------------------------------------------
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Subset to match ID's ----------------------------------------------------
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Run glmnet R2 regression to identify proteins of interest -------------------------------
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)] #remove columns with more than 50% NA

for(i in 1:ncol(count_mtx)){
  count_mtx[is.na(count_mtx[,i]), i] <- mean(count_mtx[,i], na.rm = TRUE)
}

phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, trait, nFolds = 10)
}) 

protein_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  protein_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    protein_sites_tmp <- unique(c(protein_sites_tmp ,class_coefs ))
  }
  protein_sites[[res$trait]] <- protein_sites_tmp
}

# Save proteins and expression data
datExpr <- count_mtx
save(protein_sites , datExpr , datMeta , file = paste0('~/MOGDx/data/TCGA-',project,'/RPPA/RPPA_processed.RData'))

# -------------------------------------------------------------------------
# CNV pre-processing ------------------------------------------------------
# -------------------------------------------------------------------------

# Read in Count Matrix
load(paste0('./data/TCGA-',project,'/CNV/CNV.rda'))

count_mtx <- t(assay(data))
count_mtx <- t(count_mtx)
rownames_mtx <- c()
for (name in strsplit(rownames(count_mtx) , ',')) {
  rownames_mtx <- c(rownames_mtx , substr(name[1] ,1, 12))
}
rownames(count_mtx) <- rownames_mtx
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)] #remove columns with more than 50% NA

# Read in Meta Data
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

#Intersect to common IDs
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Replace NA's with 0's 
count_mtx[is.na(count_mtx)] <- 0 

# Perform log transform on count matrix to give normal distribution resemblance
count_mtx_log <- log(count_mtx)

# Run glmnet R2 regression to get CNV's of interest
phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx_log, phenotypes, traits, nFolds = 10)
})

cnv_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  cnv_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    cnv_sites_tmp <- unique(c(cnv_sites_tmp ,class_coefs ))
  }
  cnv_sites[[res$trait]] <- cnv_sites_tmp
}

rm(count_mtx_log)

# Save CNV's and expression
datExpr <- count_mtx
save(cnv_sites , datExpr , datMeta , file = paste0('~/MOGDx/data/TCGA-',project,'/CNV/CNV_processed.RData'))
