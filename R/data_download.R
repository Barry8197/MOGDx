library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)

setwd('/data/PPMI/')

# -------------------------------------------------------------------------
# DNA Methylation ---------------------------------------------------------
# -------------------------------------------------------------------------

# 895 samples with 485577 cpg sites -> Does not need to be Normalised

query.met <- GDCquery(
      project = "TCGA-BRCA",
      data.category = "DNA Methylation",
      data.type = "Methylation Beta Value",
      platform = "Illumina Human Methylation 450"
)

#GDCdownload(query = query.met)
data <- GDCprepare(
  query = query.met ,
  save = TRUE,
  save.filename = '~/MOGDx/data/DNAm/DNAm.rda')

# -------------------------------------------------------------------------
# mRNA Gene Expression ----------------------------------------------------
# -------------------------------------------------------------------------

#1231 samples with 60666 genes -> Perform DESeq

query.exp <- GDCquery(
     project = "TCGA-BRCA",
     data.category = "Transcriptome Profiling",
     data.type = "Gene Expression Quantification",
     workflow.type = "STAR - Counts"
)

#GDCdownload(query.exp)
expdat <- GDCprepare(
      query = query.exp,
      save = TRUE,
      save.filename = "~/MOGDx/data/mRNA/mRNA.rda"
)

# -------------------------------------------------------------------------
# miRNA -------------------------------------------------------------------
# -------------------------------------------------------------------------

#1207 samples with 1882 genes -> Perform DESeq

query.mirna <- GDCquery(
     project = "TCGA-BRCA",
     experimental.strategy = "miRNA-Seq",
     data.category = "Transcriptome Profiling",
     data.type = "miRNA Expression Quantification"
)
 
#GDCdownload(query.mirna)
mirna <- GDCprepare(
     query = query.mirna,
     save = TRUE,
     save.filename = "~/MOGDx/data/miRNA/miRNA.rda"
)

# -------------------------------------------------------------------------
# RPPA  -------------------------------------------------------------------
# -------------------------------------------------------------------------

#919 samples with -> 488 features -> How to analyse protein data??? 

query.rppa <- GDCquery(
     project = "TCGA-BRCA",
     data.category = "Proteome Profiling",
     data.type = "Protein Expression Quantification"
)

#GDCdownload(query.rppa)
rppa <- GDCprepare(
    query = query.rppa,
    save = TRUE,
    save.filename = "~/MOGDx/data/RPPA/RPPA.rda"
)

# -------------------------------------------------------------------------
# CNV ---------------------------------------------------------------------
# -------------------------------------------------------------------------

# 1084 samples with -> 60624 features -> How to analyse CNV data

query.cnv <- GDCquery(
     project = "TCGA-BRCA",
     data.category = "Copy Number Variation",
     data.type = "Gene Level Copy Number"
)

#GDCdownload(query.cnv)
data <- GDCprepare(
  query.cnv,
  save = TRUE,
  save.filename = "~/MOGDx/data/CNV/CNV.rda"
  
)