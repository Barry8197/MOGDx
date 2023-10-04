source('~/Year2/MOGDx/R/preprocess_functions.R')

setwd('~/Year2/MOGDx/')

trait = 'CONCOHORT_DEFINITION'
TimeStep = 'V08'
dataset = 'PPMI'
for (modality in c( 'DNAm' )) {
  
  print(modality)
  load(paste0('./data/',dataset , '/raw/All/', TimeStep ,'/',modality,'_processed.RData'))
  
  cpg_sites_tmp <- read.csv('~/Year2/DNA_Clinical_Descriptors/data/descriptors/BMI.csv')
  cpg_sites[[trait]] <- intersect(cpg_sites_tmp$CpG , colnames(datExpr))
  print(length(cpg_sites[[trait]]))
  print(cpg_sites[[trait]][1:5])
    
  if (modality %in% c('miRNA' , 'mRNA')) {
    g <- expr.to.graph(datExpr , datMeta , trait , top_genes , modality)
  } else if (modality == 'DNAm') {
    g <- expr.to.graph(datExpr , datMeta , trait , cpg_sites , modality)
  } else if (modality == 'CNV') {
    g <- expr.to.graph(log(datExpr) , datMeta , trait , cnv_sites , modality)
  } else if (modality == 'RPPA') { 
    g <- expr.to.graph(datExpr , datMeta , trait , protein_sites , modality)
  } else if (modality == 'CSF') { 
    g <- expr.to.graph(datExpr , datMeta , trait , csf_sites , modality)
  } else if (modality == 'MOCA') { 
    g <- expr.to.graph(datExpr , datMeta , trait , q_sites , modality)
  } else if (modality == 'MDS-UPDRS') { 
    g <- expr.to.graph(datExpr , datMeta , trait , p_sites , modality)
  } else if (modality == 'Parkinsonism') {
    g <- expr.to.graph(datExpr , datMeta , trait , park_sites , modality)
  } else if (modality == 'Clinical') {
    g <- expr.to.graph(datExpr , datMeta , trait , clin_sites , modality)
  }
  
  write.csv(g, file = paste0('./Network/SNF/',TimeStep,'_BMI_graph.csv'))
  #write.csv(datExpr , file = paste0('./data/',dataset , '/raw/All/', TimeStep , '/output','/datExpr_', modality , '.csv'))
  #write.csv(datMeta , file = paste0('./data/',dataset , '/raw/All/', TimeStep , '/output','/datMeta_', modality , '.csv'))
  
}


