library(clusterProfiler)
library(DESeq2)
library(enrichplot)

project <- 'KIPAN'
trait <- 'subtype'
dataset = 'TCGA-KIPAN'
modality <- 'mRNA'
  
load(paste0('./data/',dataset , '/' , modality,'/',modality,'_processed.RData'))

subtypes <- levels(as.factor(datMeta[[trait]]))
top_genes = c()
for (subtype1 in subtypes[1:length(subtypes)-1]) {
  subtypes = subtypes[subtypes != subtype1]
  for (subtype2 in subtypes)  {
    if (subtype1 != subtype2) {
      res <- results(dds , contrast = c(trait , subtype1 , subtype2))
    }
    top_genes_tmp <-  res$padj
    names(top_genes_tmp) <- substr(rownames(res) , 1,15)
    top_genes_tmp <- head(sort(top_genes_tmp , decreasing = TRUE) , 500)
    top_genes <- c(top_genes , top_genes_tmp)
  }
}
top_genes <- sort(top_genes , decreasing = TRUE)

organism = "org.Hs.eg.db"
BiocManager::install(organism, character.only = TRUE)
library(organism, character.only = TRUE)

gene_list <- top_genes[!duplicated(top_genes)]

gse <- gseGO(geneList=gene_list, 
             ont ="ALL", 
             keyType = "ENSEMBL", 
             minGSSize = 3, 
             maxGSSize = 800, 
             pvalueCutoff = 0.05, 
             verbose = TRUE, 
             OrgDb = org.Hs.eg.db, 
             pAdjustMethod = "none",
             scoreType = "pos")

require(DOSE)
dotplot(gse, showCategory=10, split=".sign") + facet_grid(.~.sign)

geneSetId
gseaplot(gse, by = "all", title = gse$Description[2], geneSetID = 2)

terms <- gse$Description[1:3]
pmcplot(terms, 2015:2022, proportion=FALSE)
