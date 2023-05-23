source('~/MOGDx/R/preprocess_functions.R')

setwd('~/MOGDx/')

trait = 'subtype'
dataset = 'TCGA-KIPAN'
for (modality in c('mRNA' , 'miRNA' , 'DNAm' , 'RPPA' , 'CNV' )) {
  
  print(modality)
  load(paste0('./data/',dataset , '/' , modality,'/',modality,'_processed.RData'))
  
  if (modality %in% c('miRNA' , 'mRNA')) {
    g <- expr.to.graph(datExpr , datMeta , trait , top_genes , modality)
  } else if (modality == 'DNAm') {
    g <- expr.to.graph(datExpr , datMeta , trait , cpg_sites , modality)
  } else if (modality == 'CNV') {
    g <- expr.to.graph(log(datExpr) , datMeta , trait , cnv_sites , modality)
  } else if (modality == 'RPPA') { 
    g <- expr.to.graph(datExpr , datMeta , trait , protein_sites , modality)
  }
  
  write.csv(g, file = paste0('./Network/',modality,'/graph.csv'))
  write.csv(datExpr , file = paste0('./data/',dataset, '/' ,modality,'/datExpr_', modality , '.csv'))
  write.csv(datMeta , file = paste0('./data/',dataset, '/' ,modality,'/datMeta_', modality , '.csv'))
  
}
