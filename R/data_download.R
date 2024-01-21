library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)

setwd('~/MOGDx2.0/')
tcga_project <- 'KIRC'

# -------------------------------------------------------------------------
# DNA Methylation ---------------------------------------------------------
# -------------------------------------------------------------------------

query.met <- GDCquery(
      project = paste0("TCGA-",tcga_project),
      data.category = "DNA Methylation",
      data.type = "Methylation Beta Value",
      platform = "Illumina Human Methylation 450"
)

GDCdownload(query = query.met)
data <- GDCprepare(
  query = query.met ,
  save = TRUE,
  save.filename = paste0('/data/TCGA-',tcga_project,'/DNAm/DNAm.rda')
)

# -------------------------------------------------------------------------
# mRNA Gene Expression ----------------------------------------------------
# -------------------------------------------------------------------------

query.exp <- GDCquery(
  project = paste0("TCGA-",tcga_project),
     data.category = "Transcriptome Profiling",
     data.type = "Gene Expression Quantification",
     workflow.type = "STAR - Counts"
)

GDCdownload(query.exp)
expdat <- GDCprepare(
      query = query.exp,
      save = TRUE,
      save.filename = paste0('/data/TCGA-',tcga_project,'/mRNA/mRNA.rda')
)

# -------------------------------------------------------------------------
# miRNA -------------------------------------------------------------------
# -------------------------------------------------------------------------

query.mirna <- GDCquery(
  project = paste0("TCGA-",tcga_project),
     experimental.strategy = "miRNA-Seq",
     data.category = "Transcriptome Profiling",
     data.type = "miRNA Expression Quantification"
)
 
GDCdownload(query.mirna)
mirna <- GDCprepare(
     query = query.mirna,
     save = TRUE,
     save.filename = paste0('/data/TCGA-',tcga_project,'/miRNA/miRNA.rda')
)

# -------------------------------------------------------------------------
# RPPA  -------------------------------------------------------------------
# -------------------------------------------------------------------------

query.rppa <- GDCquery(
  project = paste0("TCGA-",tcga_project),
     data.category = "Proteome Profiling",
     data.type = "Protein Expression Quantification"
)

GDCdownload(query.rppa)
rppa <- GDCprepare(
    query = query.rppa,
    save = TRUE,
    save.filename = paste0('/data/TCGA-',tcga_project,'/RPPA/RPPA.rda')
)

# -------------------------------------------------------------------------
# CNV ---------------------------------------------------------------------
# -------------------------------------------------------------------------

query.cnv <- GDCquery(
     project = paste0("TCGA-",tcga_project),
     data.category = "Copy Number Variation",
     data.type = "Gene Level Copy Number"
)

GDCdownload(query.cnv)
data <- GDCprepare(
  query.cnv,
  save = TRUE,
  save.filename = paste0('/data/TCGA-',tcga_project,'/CNV/CNV.rda')
  
)
