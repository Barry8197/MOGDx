library(MethylPipeR)
library(glmnet)
library(igraph)
library(DESeq2)
library(dplyr)
library(ggplot2)
library(edgeR)
library(MethylPipeR)
library(WGCNA)

removeTraitNAs <- function(traitDF, otherDFs, trait) {
  rowsToKeep <- !is.na(traitDF[[trait]])
  traitDF <- traitDF[rowsToKeep, ]
  otherDFs <- lapply(otherDFs, function(df) {
    if (is.data.frame(df) || is.matrix(df)) {
      df[rowsToKeep, ]
    } else if (is.null(df)) {
      # For example, if foldID is NULL in cvTrait
      df
    } else {
      # Assumes df is a vector
      df[rowsToKeep]
    }
  })
  list(traitDF = traitDF, otherDFs = otherDFs)
}


cvTrait <- function(trainMethyl, trainPhenotypes, trait, nFolds) {
  print(paste0('Removing rows with missing ', trait, ' from training data.'))
  trainRemoveNAResult <- removeTraitNAs(trainPhenotypes, list(trainMethyl = trainMethyl), trait)
  trainPhenotypes <- trainRemoveNAResult$traitDF
  trainMethyl <- trainRemoveNAResult$otherDFs$trainMethyl
  
  print('Fitting lasso model')
  methylModel <- cv.glmnet(x = trainMethyl,
                           y = as.factor(trainPhenotypes[[trait]]),
                           seed = 42,
                           family = 'multinomial',
                           type.measure = "class",
                           alpha = 1,
                           nFolds = nFolds,
                           parallel = TRUE,
                           trace.it = 1)
  print(methylModel)
  list(trait = trait, model = methylModel)
}

diff_expr <- function(count_mtx , datMeta , trait , n_genes , modality) {
  
  # Remove Genes with low level of expression -------------------------------
  to_keep = rowSums(count_mtx) > 0 #removed 1157 genes
  print(paste0('Removing ',length(to_keep) - sum(to_keep),' Genes with all 0'))
  
  count_mtx <- count_mtx[to_keep,]
  datExpr <- count_mtx
  
  if (modality != 'miRNA') {
    to_keep = filterByExpr(datExpr , group = datMeta[[trait]])
    
    print(paste0('keeping ',sum(to_keep) ,' genes'))
    print(paste0("Removing ",length(to_keep) - sum(to_keep)," Genes"))
    
    count_mtx = count_mtx[to_keep,]
  }
  
  # Remove Outliers ---------------------------------------------------------
  print('removing outliers')
  absadj = count_mtx %>% bicor %>% abs
  netsummary = fundamentalNetworkConcepts(absadj)
  ku = netsummary$Connectivity
  z.ku = (ku-mean(ku))/sqrt(var(ku))

  to_keep = z.ku > -2
  print(paste0("Keeping ",sum(to_keep)," Samples"))
  print(paste0("Removed ",length(to_keep) - sum(to_keep), " Samples"))

  count_mtx <- count_mtx[,to_keep] #removed 36
  datMeta <- datMeta[to_keep,]
  
  # Normalisation Using DESeq -----------------------------------------------
  plot_data = data.frame('ID'=rownames(count_mtx), 'Mean'=rowMeans(count_mtx), 'SD'=apply(count_mtx,1,sd))
  
  plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.1) + geom_abline(color='black') +
    scale_x_log10() + scale_y_log10() + theme_minimal()  + theme(plot.title = element_text(hjust = 0.5)) 
  
  datMeta[[trait]] <- as.factor(datMeta[[trait]])
  dds = DESeqDataSetFromMatrix(countData = count_mtx, colData = datMeta , design = formula(paste("~ 0 +",trait)))
  
  print('performing DESeq')
  dds = DESeq(dds)
  
  # DEA Plots ---------------------------------------------------------------
  DE_info = results(dds)
  DESeq2::plotMA(DE_info, main= 'Original LFC values')
  
  # VST Transformation of Data ----------------------------------------------
  nsub_check = sum( rowMeans( counts(dds, normalized=TRUE)) > 5 )
  if (nsub_check < 1000) {
    vsd = vst(dds , nsub= nsub_check)
  } else {
    vsd = vst(dds)
  }
  
  datExpr_vst = assay(vsd)
  datMeta_vst = colData(vsd)
  
  meanSdPlot(datExpr_vst, plot=FALSE)$gg + theme_minimal() + ylim(c(0,2))
  
  plot_data = data.frame('ID'=rownames(datExpr_vst), 'Mean'=rowMeans(datExpr_vst), 'SD'=apply(datExpr_vst,1,sd))
  
  plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.2) + geom_smooth(color = 'gray') +
    scale_x_log10() + scale_y_log10() + theme_minimal()
  
  subtypes <- levels(as.factor(datMeta[[trait]]))
  top_genes = c()
  for (subtype1 in subtypes[1:length(subtypes)-1]) {
    subtypes = subtypes[subtypes != subtype1]
    for (subtype2 in subtypes)  {
      if (subtype1 != subtype2) {
        res <- results(dds , contrast = c(trait , subtype1 , subtype2))
      }
      top_genes = unique(c(top_genes , head(order(res$padj) , n_genes) ))
    }
  }
  
  list(dds = dds , datExpr = datExpr_vst, datMeta = datMeta_vst , top_genes = top_genes)
}

make.knn.graph<-function(D,k){
  # calculate euclidean distances between cells
  dist<-as.matrix(dist(D))
  # make a list of edges to k nearest neighbors for each cell
  edges <- mat.or.vec(0,2)
  for (i in 1:nrow(dist)){
    # find closes neighbours
    matches <- setdiff(order(dist[i,],decreasing = F)[1:(k+1)],i)
    if (length(matches) > k) {
      edges <- rbind(edges,cbind(rep(i,length(matches)),matches))
      #edges <- rbind(edges,cbind(matches,rep(i,length(matches))))
    } else {
      edges <- rbind(edges,cbind(rep(i,k),matches))
      #edges <- rbind(edges,cbind(matches,rep(i,k)))
    }
    # add edges in both directions
    
    #edges <- rbind(edges,cbind(matches,rep(i,k)))  
  }
  # create a graph from the edgelist
  graph <- graph_from_edgelist(edges,directed=F)
  V(graph)$frame.color <- NA
  # make a layout for visualizing in 2D
  set.seed(1)
  g.layout<-layout_with_fr(graph)
  return(list(graph=graph,layout=g.layout))        
}

expr.to.graph<-function(datExpr , datMeta , trait , top_genes , modality){
  
  if (modality %in% c('mRNA' , 'miRNA')) {
    mat <- datExpr[top_genes, ]
  } else {
    mat <- t(datExpr[ , top_genes[[trait]]])
  }
  
  if (modality %in% c('mRNA' , 'miRNA' , 'DNAm' , 'RPPA' , 'CSF')) {
    mat <- mat - rowMeans(mat)
    corr_mat <- cor(mat, method="pearson")
  } else {
    corr_mat <- t(mat)
  }
  
  print(dim(mat))
  g <- make.knn.graph(corr_mat , 15)
  
  plot.igraph(g$graph,layout=g$layout, vertex.frame.color='black', vertex.color=as.factor(datMeta[[trait]]),
              vertex.size=5,vertex.label=NA, vertex.label.cex = 0.3 , main=modality )
  
  g <- g$graph
  g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)
  
  # Remove any vertices remaining that have no edges
  g <- delete.vertices(g, degree(g)==0)
  
  # Assign names to the graph vertices 
  V(g)$name <- rownames(datMeta)
  V(g)$class <- as.character(datMeta[[trait]])
  V(g)$color <- as.numeric(as.factor(V(g)$class))
  V(g)$vertex.frame.color <- "black"
  
  return(as_long_data_frame(g))
}

snf.to.graph <- function(W , datMeta , trait , idx , sub_mod_list) {
  
  g <- make.knn.graph(W , 15)
  
  plot.igraph(g$graph,layout=g$layout, vertex.frame.color='black', vertex.color=as.numeric(as.factor(datMeta[idx,][[trait]])),
              vertex.size=5,vertex.label=NA,main=paste0(sub_mod_list , collapse = '_'))
  
  g <- g$graph
  g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)
  
  # Remove any vertices remaining that have no edges
  g <- delete.vertices(g, degree(g)==0)
  
  # Assign names to the graph vertices 
  V(g)$name <- rownames(datMeta[idx,])
  V(g)$class <- as.character(datMeta[idx,][[trait]])
  V(g)$color <- as.numeric(as.factor(V(g)$class))
  V(g)$vertex.frame.color <- "black"
  
  return(g)
}
