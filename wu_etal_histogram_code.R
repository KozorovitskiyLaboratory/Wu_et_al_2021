
## Wu, M. et al eLife 2021
## histogram fit

library(tidyverse)
library(readxl)
library(fitdistrplus)

############################
## Load raw data
############################
# %failure (Baseline.LH, LH, Baseline.wLH, wLH)
raw.input_LH  <- read_excel(path="mzw_eLife_histogram_LH.xlsx")
raw.input_wLH <- read_excel(path="mzw_eLife_histogram_wLH.xlsx")

# Drop any NA's
# %failure data for Baseline.LH, LH, Baseline.wLH, and wLH
Baseline.LH <- drop_na(raw.input_LH)
Baseline.LH <- Baseline.LH$Baseline.LH
LH <- raw.input_LH$LH

Baseline.wLH <- raw.input_wLH$Baseline.wLH
wLH <- raw.input_wLH$wLH

############################
## Histogram parameters
############################
# Histogram parameters
# Bin width
width = 4
bins <- seq(0,100,width)

############################
## Fitting and plotting histograms
############################
## Using fitdistrplus for fitting data with the Gaussian distribution

## LH
fitnorm_LH <- fitdist(LH,
                     distr = "norm",
                     method = "mle")
summary(fitnorm_LH)

{hist(x=fitnorm_LH$data,
      breaks=bins,
      freq=FALSE, main="LH - norm")
  lines(seq(0,100,1),
        dnorm(seq(0,100,1), mean=fitnorm_LH$estimate['mean'], sd=fitnorm_LH$estimate['sd']),
        col='firebrick')}

#for exporting histogram into .txt files
#res1_hist <- hist(x=LH,
#                  breaks=bins,
#                  freq=FALSE, main="LH", xlab='% failure')
#res1_hist_df <- data.frame(res1_hist$mids, res1_hist$counts, res1_hist$density)
#res1_den_df <- data.frame(seq(0,100,1),
#                       dnorm(seq(0,100,1), mean=fitnorm_LH$estimate['mean'], sd=fitnorm_LH$estimate['sd']))
# write.table(res1_hist_df, "/Users/vasdumrong/Box/res1_hist_df.txt", sep="\t", row.names = FALSE)
# write.table(res1_den_df, "/Users/vasdumrong/Box/res1_den_df.txt", sep="\t", row.names = FALSE)

## Baseline.LH
fitnorm_BLH <- fitdist(Baseline.LH,
                     distr = "norm",
                     method = "mle")
summary(fitnorm_BLH)

{hist(x=fitnorm_BLH$data,
      breaks=bins,
      freq=FALSE, main="Baseline.LH - norm")
  lines(seq(0,100,1),
        dnorm(seq(0,100,1), mean=fitnorm_BLH$estimate['mean'], sd=fitnorm_BLH$estimate['sd']),
        col='firebrick')}


#res1_hist <- hist(x=Baseline.LH,
#                  breaks=bins,
#                  freq=FALSE, main="Baseline.LH", xlab='% failure')
#res1_hist_df <- data.frame(res1_hist$mids, res1_hist$counts, res1_hist$density)
#res1_den_df <- data.frame(seq(0,100,1),
#                          dnorm(seq(0,100,1), mean=fitnorm_BLH$estimate['mean'], sd=fitnorm_BLH$estimate['sd']))
#write.table(res1_hist_df, "/Users/vasdumrong/Box/res1_hist_df.txt", sep="\t", row.names = FALSE)
#write.table(res1_den_df, "/Users/vasdumrong/Box/res1_den_df.txt", sep="\t", row.names = FALSE)

## wLH
fitnorm_wLH <- fitdist(wLH,
                     distr = "norm",
                     method = "mle")
summary(fitnorm_wLH)

{hist(x=fitnorm_wLH$data,
      breaks=bins,
      freq=FALSE, main="wLH - norm")
  lines(seq(0,100,1),
        dnorm(seq(0,100,1), mean=fitnorm_wLH$estimate['mean'], sd=fitnorm_wLH$estimate['sd']),
        col='firebrick')}

#res1_hist <- hist(x=wLH,
#                  breaks=bins,
#                  freq=FALSE, main="wLH", xlab='% failure')
#res1_hist_df <- data.frame(res1_hist$mids, res1_hist$counts, res1_hist$density)
#res1_den_df <- data.frame(seq(0,100,1),
#                          dnorm(seq(0,100,1), mean=fitnorm_wLH$estimate['mean'], sd=fitnorm_wLH$estimate['sd']))
#write.table(res1_hist_df, "/Users/vasdumrong/Box/res1_hist_df.txt", sep="\t", row.names = FALSE)
#write.table(res1_den_df, "/Users/vasdumrong/Box/res1_den_df.txt", sep="\t", row.names = FALSE)


## Baseline.wLH
fitnorm_BwLH <- fitdist(Baseline.wLH,
distr = "norm",
method = "mle")
summary(fitnorm_BwLH)

{hist(x=fitnorm_BwLH$data,
      breaks=bins,
      freq=FALSE, main="Baseline.wLH - norm")
  lines(seq(0,100,1),
        dnorm(seq(0,100,1), mean=fitnorm_BwLH$estimate['mean'], sd=fitnorm_BwLH$estimate['sd']),
        col='firebrick')}

#res1_hist <- hist(x=Baseline.wLH,
#                  breaks=bins,
#                  freq=FALSE, main="Baseline.wLH", xlab='% failure')
#res1_hist_df <- data.frame(res1_hist$mids, res1_hist$counts, res1_hist$density)
#res1_den_df <- data.frame(seq(0,100,1),
#                          dnorm(seq(0,100,1), mean=fitnorm_BwLH$estimate['mean'], sd=fitnorm_BwLH$estimate['sd']))
#write.table(res1_hist_df, "/Users/vasdumrong/Box/res1_hist_df.txt", sep="\t", row.names = FALSE)
#write.table(res1_den_df, "/Users/vasdumrong/Box/res1_den_df.txt", sep="\t", row.names = FALSE)

