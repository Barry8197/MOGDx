source('~/Year2/MOGDx/R/preprocess_functions.R')

setwd('~/Year2/MOGDx/')

trait = 'CONCOHORT_DEFINITION'
TimeStep = 'V08'
dataset = 'PPMI'
for (modality in c('CSF' )) {
  
  print(modality)
  load(paste0('./data/',dataset , '/', TimeStep , '/' , modality,'/',modality,'_processed.RData'))
  
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
  }
  
  write.csv(g, file = paste0('./Network/',modality,'/graph.csv'))
  write.csv(datExpr , file = paste0('./data/',dataset , '/', TimeStep , '/' , modality,'/','/datExpr_', modality , '.csv'))
  write.csv(datMeta , file = paste0('./data/',dataset , '/', TimeStep , '/' , modality,'/','/datMeta_', modality , '.csv'))
  
}


