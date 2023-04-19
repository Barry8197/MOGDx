library(igraph)
library(DESeq2)
library(dplyr)

setwd('~/MOGDx/')
load('./data/RPPA/RPPA_processed.RData')

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
vsd <- vst(dds)
res <- results(dds)


# miRNA -------------------------------------------------------------------
datExpr <- read_per_million




# Graph Generaion ---------------------------------------------------------
mat <- t(count_mtx[, protein_sites[[1]]]) 
mat <- mat[head(order(res$padj), 1500), ]
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
  vertex.size=vSizes,
  vertex.label.dist=-0.5,
  vertex.label.color="black",
  asp=FALSE,
  vertex.label.cex=0.6,
  main="gPD vs. HC Correlation Network")

write.csv(as_long_data_frame(g) , file = './Network/RPPA/graph.csv')
write.csv(count_mtx , file = './data/RPPA/RPPA_datExpr.csv')
write.csv(datMeta , file = './data/RPPA/RPPA_datMeta.csv')
