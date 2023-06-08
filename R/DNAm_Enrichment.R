library(mCSEA)

project <- 'BRCA'
trait <- 'paper_BRCA_Subtype_PAM50'
dataset = 'TCGA-BRCA'
modality <- 'DNAm'

load(paste0('./data/',dataset , '/' , modality,'/',modality,'_processed.RData'))
## -----------------------------------------------------------------------------
betaTest <- t(datExpr)
subtypes <- levels(as.factor(datMeta[[trait]]))
for (subtype in subtypes[c(1,4,5)]) {
  phenoTest <- as.data.frame(datMeta[[trait]])
  colnames(phenoTest) <- trait
  rownames(phenoTest) <- colnames(betaTest)
  phenoTest[ which(phenoTest[[trait]] != subtype) , ] <- 'Other'
  phenoTest[[trait]] <- as.factor(phenoTest[[trait]])
  
  myRank <- rankProbes(betaTest, phenoTest, refGroup = 'Other')
  
  myRank[!(names(myRank) %in% cpg_sites[[trait]])] <- 0.00000001
  
  myResults <- mCSEATest(myRank, betaTest, phenoTest, 
                         regionsTypes = "genes", platform = "EPIC")
  
  save(myResults , file = paste0('./data/',dataset,'/DNAm/',subtype,'_mCSEA.RData'))
}

myResults_all <- list()
for (subtype in subtypes) {
  load(paste0('./data/',dataset,'/DNAm/',subtype,'_mCSEA.RData'))
  myResults_all[[subtype]] <- myResults
} 

top_genes <- c()
out_table <- c()
for (result in myResults_all) {
  
  top_genes <- c(top_genes , rownames(result$genes)[order(result$genes$padj)[1:5]])
  result_table <- cbind(result$genes[order(result$genes$padj)[1:5] , c(1,2,5,6)] , 'Subtype' = rep(unique(result$pheno[result$pheno != 'Other']) , 5))
  out_table <- rbind(out_table , result_table )
  
}


write.csv(out_table , file = paste0('./',project,'_DNAm_enrichment.csv')
