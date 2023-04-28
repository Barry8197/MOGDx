library(igraph)
library(DESeq2)
library(dplyr)

setwd('~/MOGDx/')
modality <- 'RPPA'
load(paste0('./data/',modality,'/',modality,'_processed.RData'))

length(protein_sites[[1]])
make.knn.graph<-function(D,k){
  # calculate euclidean distances between cells
  dist<-as.matrix(dist(D))
  # make a list of edges to k nearest neighbors for each cell
  edges <- mat.or.vec(0,2)
  for (i in 1:nrow(dist)){
    # find closes neighbours
    matches <- setdiff(order(dist[i,],decreasing = F)[1:(k+1)],i)
    # add edges in both directions
    edges <- rbind(edges,cbind(rep(i,k),matches))  
    edges <- rbind(edges,cbind(matches,rep(i,k)))  
  }
  # create a graph from the edgelist
  graph <- graph_from_edgelist(edges,directed=F)
  V(graph)$frame.color <- NA
  # make a layout for visualizing in 2D
  set.seed(1)
  g.layout<-layout_with_fr(graph)
  return(list(graph=graph,layout=g.layout))        
}


# mRNA  -------------------------------------------------------------------
vsd <- vst(dds , nsub = 349)
res <- results(dds)


# miRNA -------------------------------------------------------------------
datExpr <- log(t(read_per_million))

dim(datExpr)
hist(datExpr[,500])

# Graph Generaion ---------------------------------------------------------
mat <- t(datExpr[, cpg_sites[[1]]])
mat <- datExpr[head(order(res$padj), 200), ]
mat <- mat - rowMeans(mat)
corr_mat <- as.matrix(as.dist(cor(mat, method="pearson")))
heatmap(corr_mat)

g <- make.knn.graph(corr_mat , 15)

plot.igraph(g$graph,layout=g$layout,vertex.color=as.numeric(as.factor(datMeta[,'paper_BRCA_Subtype_PAM50'])),
            vertex.size=5,vertex.label=NA,main="padj KNN network")

g <- g$graph

g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)


# Remove any vertices remaining that have no edges
g <- delete.vertices(g, degree(g)==0)

# Assign names to the graph vertices (optional)
V(g)$name <- rownames(datMeta)
V(g)$class <- datMeta$paper_BRCA_Subtype_PAM50

# Change shape of graph vertices
V(g)$shape <- "sphere"

# Change colour of graph vertices
V(g)$color <- as.numeric(as.factor(V(g)$class))

# Change colour of vertex frames
V(g)$vertex.frame.color <- "white"

# Scale the size of the vertices to be proportional to the level of expression of each gene represented by each vertex
# Multiply scaled vales by a factor of 10
scale01 <- function(x){(x-min(x))/(max(x)-min(x))}
vSizes <- (scale01(apply(corr_mat, 1, mean)) + 1.0) * 10

# Plot the tree object
plot(
  g,
  layout=layout.fruchterman.reingold,
  edge.curved=TRUE,
  vertex.size=5,
  vertex.label = NA,
  asp=FALSE,
  main="gPD vs. HC Correlation Network")

write.csv(as_long_data_frame(g) , file = paste0('./Network/',modality,'/graph.csv'))
write.csv(datExpr , file = paste0('./data/',modality,'/datExpr_', modality , '.csv'))
write.csv(datMeta , file = paste0('./data/',modality,'/datMeta_', modality , '.csv'))
