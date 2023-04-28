library(SNFtool)
library(ANF)
library(igraph)
library(data.table)

datMeta <- t(data.frame( row.names = c('patient' ,  'race' , 'gender' , 'sample_type paper_BRCA_Subtype_PAM50')))
for (mod in c('CNV' , 'RPPA' , 'mRNA' , 'miRNA' , 'DNAm')) {
  print(mod)
  datMeta <- rbind(datMeta , read.csv(paste0('./data/',mod,'/',mod,'_datMeta.csv') , row.names = 1))
  
}
datMeta <- datMeta[!(duplicated(datMeta)),]
# This pulls in all my networks from the directory. Note - while the example uses two features and then
# calculates its similarity (affinity) matrix, I do this in another script to construct a patient similarity 
# network. My affinity matrix is made in the next for loop. My guess is that this loop wont be useful and 
# you can ignore.

all_idx <- c()
g_list <- list()
for (net in list.files('./Network/SNF/')) {
  print(net)
  net_graph <- read.csv(paste0('./Network/SNF/',net) , row.names = 1)
  patients <- unique(data.frame(id = c(net_graph$from_name , net_graph$to_name) ,
                                class = c(net_graph$from_class , net_graph$to_class)))
  relation <- data.frame(from = net_graph$from_name , 
                         to = net_graph$to_name )
  
  g_net <- graph_from_data_frame(relation , directed = FALSE , vertices = patients)
  g_net <- simplify(g_net, remove.multiple=TRUE, remove.loops=TRUE)
  
  g_list[[net]] <- g_net
  all_idx <- unique(append(all_idx,V(g_net)$name))
}

# This for loop extracts the adjacency (similarity/affinity) matrix from each graph.
adjacency_graphs <- list()
for (graph_names in names(g_list)) {
  
  missing_idx <- setdiff(all_idx , V(g_list[[graph_names]])$name)
  g_list[[graph_names]] <- add_vertices(g_list[[graph_names]] , length(missing_idx) , name = missing_idx)
  
  graph_adj <- as.matrix(as_adjacency_matrix(g_list[[graph_names]]))[all_idx,all_idx]
  
  adjacency_graphs[[graph_names]] <- graph_adj
  
}

## First, set all the parameters:
K = 20;		# number of neighbors, usually (10~30)
#alpha = 0.5;  	# hyperparameter, usually (0.3~0.8)
T = 3; 	# Number of Iterations, usually (10~20)

#change this to similarity matrix
W = SNF(adjacency_graphs, K , T)
W <- W - diag(0.5 , dim(W)[1]) 

# This constructs a network using the SNF matrix W as weights. Youll see with the weights are all very low
# so im currently working to understand these better. However, as per SNF_ex.R, performing clustering on W
# seems to work v nicely.
g <- graph.adjacency(
  W,
  mode="undirected",
  weighted=TRUE,
  diag=FALSE
)

V(g)$class <- datMeta[all_idx,]$paper_BRCA_Subtype_PAM50

g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)

# Remove edges below absolute Pearson correlation 0.8
threshold <- quantile(W , 0.99)
g <- delete_edges(g, E(g)[which(E(g)$weight<threshold)])

# Remove any vertices remaining that have no edges
g <- delete.vertices(g, degree(g)==0)

# Change shape of graph vertices
V(g)$shape <- "sphere"

# Change colour of graph vertices
V(g)$color <- as.numeric(as.factor(V(g)$class))

# Change colour of vertex frames
V(g)$vertex.frame.color <- "white"

# Plot the tree object
plot(
  g,
  layout=layout.fruchterman.reingold,
  vertex.size=5,
  vertex.label = NA,
  main="BL Correlation Network")

write.csv(as_long_data_frame(g) , file = './Network/SNF/graph.csv')

