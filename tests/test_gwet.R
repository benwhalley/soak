# install.packages('irrCAC')
library(irrCAC)
this_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(this_dir)

df <- readr::read_csv('gwet_test.csv')
res <- irrCAC::gwet.ac1.raw(df)
res$est