library(DESeq2)
library(WGCNA)
library(dplyr)
library(ggplot2)
library(edgeR)
library(reshape2)
library(kableExtra)
library(plotly)
library(vsn)
library(tibble)
library(pheatmap)
library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)

setwd('~/MOGDx/') 


# mRNA pre-processing -----------------------------------------------------

# Pull in Count Matrices --------------------------------------------------
load('~/MOGDx/data/mRNA.rda')
count_mtx <- assay(data)

# Create coldata and condition table --------------------------------------
coldata <- colData(data)

# Count Distribution ------------------------------------------------------
counts = count_mtx %>% melt

count_distr = data.frame('Statistic' = c('Min', '1st Quartile', 'Median', 'Mean', '3rd Quartile', 'Max'),
                         'Values' = c(min(counts$value), quantile(counts$value, probs = c(.25, .5)) %>% unname,
                                      mean(counts$value), quantile(counts$value, probs = c(.75)) %>% unname,
                                       max(counts$value)))

count_distr %>% kable(digits = 2, format.args = list(scientific = FALSE)) %>% kable_styling(full_width = F)

rm(counts, count_distr)


# Remove Samples with NA in BRCA type -------------------------------------
to_keep = !(is.na(coldata$paper_BRCA_Subtype_PAM50))
coldata <- coldata[to_keep,]
count_mtx <- count_mtx[,to_keep]

# Remove Genes with low level of expression -------------------------------
#load('./../gPD_HC/preprocessedData/V08/raw_counts.RData')
to_keep = rowSums(count_mtx) > 0 #removed 1157 genes
length(to_keep) - sum(to_keep)

count_mtx <- count_mtx[to_keep,]

datExpr <- count_mtx
datMeta <- coldata

#to_keep = apply(datExpr, 1, function(x) 100*mean(x>0)) >= threshold
to_keep = filterByExpr(datExpr , group = coldata$paper_BRCA_Subtype_PAM50)
print('keeping genes')
print(sum(to_keep))
length(to_keep) - sum(to_keep)
count_mtx = count_mtx[to_keep,]

rm( to_keep , datExpr , datMeta)


# Remove Outliers ---------------------------------------------------------
print('removing outliers')
absadj = count_mtx %>% bicor %>% abs
netsummary = fundamentalNetworkConcepts(absadj)
ku = netsummary$Connectivity
z.ku = (ku-mean(ku))/sqrt(var(ku))

plot_data = data.frame('sample'=1:length(z.ku), 'distance'=z.ku, 'Sample_ID'=coldata$sample,
                       'Subject_ID'=coldata$patient, 'Sex'=coldata$gender, 'Age'=coldata$days_to_birth,
                       'Diagnosis'=coldata$paper_BRCA_Subtype_PAM50)

ggplot(plot_data) + geom_point(aes(distance , Subject_ID , color = Diagnosis )  ) +
  theme(axis.title.y=element_blank(), axis.text.y=element_blank(),axis.ticks.y=element_blank())

to_keep = z.ku > -2
sum(to_keep)
length(to_keep) - sum(to_keep)

count_mtx <- count_mtx[,to_keep] #removed 95
coldata <- coldata[to_keep,]

rm(absadj, netsummary, ku, z.ku, plot_data , to_keep)


# Normalisation Using DESeq -----------------------------------------------
plot_data = data.frame('ID'=rownames(count_mtx), 'Mean'=rowMeans(count_mtx), 'SD'=apply(count_mtx,1,sd))

plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.1) + geom_abline(color='black') +
  scale_x_log10() + scale_y_log10() + theme_minimal() + ggtitle('gPD vs. HC V08 gene counts post low expression filtering') + theme(plot.title = element_text(hjust = 0.5)) 

rm(plot_data)


# Design Matrix Correlations ----------------------------------------------
out <- c()
for (i in c(45,47,27,46,75)) {
  row_tmp <- c()
  for (a in c(45,47,27,46,75)) {
    row_tmp <- rbind(row_tmp , cor(as.numeric(coldata[,i]) , as.numeric(coldata[,a]) , use = 'complete.obs'))
  }
  out <- cbind(out , row_tmp)
}
rownames(out) <- colnames(coldata[, c(45,47,27,46,75)])
colnames(out) <- colnames(coldata[, c(45,47,27,46,75)])
pal <- wes_palette("Zissou1", 100, type = "continuous")
pheatmap(round(out,2) , cluster_rows = FALSE , cluster_cols = FALSE , display_numbers = round(out , 2) )
out

dds = DESeqDataSetFromMatrix(countData = count_mtx, colData = coldata , design = ~ paper_BRCA_Subtype_PAM50)

print('performing DESeq')
dds = DESeq(dds)

# DEA Plots ---------------------------------------------------------------
DE_info = results(dds)
DESeq2::plotMA(DE_info, main= 'Original LFC values')

rm(DE_info)
# VST Transformation of Data ----------------------------------------------
vsd = vst(dds)

datExpr_vst = assay(vsd)
datMeta_vst = colData(vsd)

rm(vsd)

meanSdPlot(datExpr_vst, plot=FALSE)$gg + theme_minimal() + ylim(c(0,2))

plot_data = data.frame('ID'=rownames(datExpr_vst), 'Mean'=rowMeans(datExpr_vst), 'SD'=apply(datExpr_vst,1,sd))

plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.2) + geom_smooth(color = 'gray') +
  scale_x_log10() + scale_y_log10() + theme_minimal()

rm(plot_data)


# Save Expression & Meta data ---------------------------------------------
datExpr = datExpr_vst
datMeta = datMeta_vst %>% data.frame

rm(datExpr_vst, datMeta_vst , count_mtx , coldata)

genes_info = DE_info %>% data.frame  %>% 
  mutate(significant=padj<0.05 & !is.na(padj) )
genes_info$ID <- rownames(genes_info)
genes_info$ID <- gsub("\\..*","", genes_info$ID)

save(datExpr, datMeta, dds, genes_info, file=paste0('~/MOGDx/data/mRNA_processed.RData'))


# miRNA preprocessing -----------------------------------------------------
load('data/miRNA/miRNA.rda')

read_count <- data.frame(row.names = data$miRNA_ID)
read_per_million <- data.frame(row.names = data$miRNA_ID)
for (i in 2:dim(data)[2]) {
  
  if (i%%3 == 2) {
    read_count <- cbind(read_count , data[ , i] )
  }
  if (i%%3 == 0) {
    read_per_million <- cbind(read_per_million , data[ , i])
  }
}

colname_read_count <- c()
colname_read_per_million <- c()
for (i in 2:dim(data)[2]) {
  
  if (i%%3 == 2) {
    colname_read_count <- c(colname_read_count , substr(strsplit(colnames(data)[i] , '_')[[1]][3] , 1,12))
  }
  if (i%%3 == 0) {
    colname_read_per_million <- cbind(colname_read_per_million , substr(strsplit(colnames(data)[i] , '_')[[1]][6] , 1,12))
  }
}
colnames(read_count) <- colname_read_count
colnames(read_per_million) <- colname_read_per_million

save(read_count , read_per_million , file=paste0('~/MOGDx/data/miRNA/miRNA_preprocessed.RData'))


# DNAm preprocessing ------------------------------------------------------
library(MethylPipeR)


# Create Meta File --------------------------------------------------------
load('data//mRNA/mRNA.rda')
coldata <- colData(data)
datMeta <- as.data.frame(coldata[,c('patient','race' , 'gender' , 'sample_type' , 'paper_BRCA_Subtype_PAM50')])
datMeta <- datMeta[!(is.na(datMeta$paper_BRCA_Subtype_PAM50)) , ]
datMeta <- datMeta[!(duplicated(datMeta[ , c('patient' , 'paper_BRCA_Subtype_PAM50')])) , ] 
rownames(datMeta) <- datMeta$patient


# Load CpG Counts ---------------------------------------------------------
load('data/DNAm/DNAm.rda')
count_mtx <- assay(data)

to_keep = complete.cases(count_mtx) #removed 191928 cpg sites
length(to_keep) - sum(to_keep)

count_mtx <- t(count_mtx[to_keep,])
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)


# Run RTFS to get CpG's associated with phenotypes of interest ------------
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]


phenotypes <- datMeta[,c('patient' , 'paper_BRCA_Subtype_PAM50' , 'race' , 'gender')]
colnames(phenotypes)

traits <- c('paper_BRCA_Subtype_PAM50' , 'race' )

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
  methylModel <- fitMPRModelCV(type = 'continuous',
                               method = 'glmnet',
                               trainXs = trainMethyl,
                               trainY = as.numeric(as.factor(trainPhenotypes[[trait]])),
                               seed = 42,
                               alpha = 1,
                               nFolds = nFolds,
                               parallel = TRUE,
                               trace.it = 1)
  list(trait = trait, model = methylModel)
}

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, trait, nFolds = 10)
})

cpg_sites = c()
for (i in 1:length(traitResults)) {
  trait_coefs <- coef(traitResults[[i]]$model$model , s = "lambda.min")
  cpg_sites[[i]] <- trait_coefs@Dimnames[[1]][which(trait_coefs != 0)]
  cpg_sites[[i]] <- cpg_sites[[i]][2:length(cpg_sites[[i]])]
}


save(cpg_sites , count_mtx , datMeta , file = '~/MOGDx/data/DNAm/DNAm_processed.RData')


# Protein (RPPA) pre-processing ----------------------------------------------------------
library(mice)
library(VIM)

load('data/RPPA/RPPA.rda')
count_mtx <- t(data[  , 6:ncol(data) ])
colnames(count_mtx) <- data$peptide_target
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)


# Create Meta Data -----------------------------------------------------------
load('data/mRNA/mRNA.rda')
coldata <- colData(data)
datMeta <- as.data.frame(coldata[,c('patient','race' , 'gender' , 'sample_type' , 'paper_BRCA_Subtype_PAM50')])
datMeta <- datMeta[!(is.na(datMeta$paper_BRCA_Subtype_PAM50)) , ]
datMeta <- datMeta[!(duplicated(datMeta[ , c('patient' , 'paper_BRCA_Subtype_PAM50')])) , ] 
rownames(datMeta) <- datMeta$patient


# Run RTFS to identify proteins of interest -------------------------------
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

mice_plot <- aggr(count_mtx, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(count_mtx), cex.axis=.7,
                  gap=5, ylab=c("Missing data","Pattern"))

missing_50 <- colnames(count_mtx[ , mice_plot$missings[,'Count'] > 0.5*ncol(count_mtx)]) #remove proteins with more than 50% missing
count_mtx <- count_mtx[, -which(colnames(count_mtx) %in% missing_50)]

#count_mtx_colnames <-colnames(count_mtx)
#colnames(count_mtx) <- NULL
#count_mtx_imputed <- mice(count_mtx, m=5, maxit = 50, method = 'pmm', seed = 500)
for(i in 1:ncol(count_mtx)){
  count_mtx[is.na(count_mtx[,i]), i] <- mean(count_mtx[,i], na.rm = TRUE)
}

#summary(count_mtx_imputed)

phenotypes <- datMeta[,c('patient' , 'paper_BRCA_Subtype_PAM50' , 'race' , 'gender')]
colnames(phenotypes)

traits <- c('paper_BRCA_Subtype_PAM50' , 'race' )

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, trait, nFolds = 10)
}) ## Maybe run RTFS on difficult to extract subtypes i.e. Luminal A vs. Luminal B. 

protein_sites = c()
for (i in 1:length(traitResults)) {
  trait_coefs <- coef(traitResults[[i]]$model$model , s = "lambda.min")
  protein_sites[[i]] <- trait_coefs@Dimnames[[1]][which(trait_coefs != 0)]
  protein_sites[[i]] <- protein_sites[[i]][2:length(protein_sites[[i]])]
}


save(protein_sites , count_mtx , datMeta , file = '~/MOGDx/data/RPPA/RPPA_processed.RData')


# CNV pre-processing ------------------------------------------------------
load('data/CNV/CNV.rda')

count_mtx <- t(assay(data))
rownames_mtx <- c()
for (name in strsplit(rownames(count_mtx) , ',')) {
  rownames_mtx <- c(rownames_mtx , substr(name[1] ,1, 12))
}
rownames(count_mtx) <- rownames_mtx
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<nrow(count_mtx)]

