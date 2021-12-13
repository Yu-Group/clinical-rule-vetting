# load in useful packages
library(tidyverse)
library(forcats)
library(lubridate)
library(stringr)
library(patchwork)
library(scales)
library(ggrepel)
setwd('/Users/Yaxuan/Github/stat-215-final/csi-pecarn-rule-vetting/rulevetting/projects/csi_pecarn/')
rm(list = ls())

data_0 = read.csv('notebooks/figs/nonsense.csv')

data <- read.csv('notebooks/figs/2-5vlist.csv') %>%
  filter(TPR < 0.96) %>%
  bind_rows(data_0)
p = data %>%
  ggplot(aes(x = FPR, y = TPR)) +
  geom_point() +
  geom_line() +
  geom_text_repel(aes(label = variable)) +
  geom_hline(yintercept = 0.95, linetype = 'dashed', color = 'red')+
  xlim(0,1) + ylim(0,1) +
  xlab('Fasle Possitive Rate (1 - Specificity)') +
  ylab('True Possitive Rate (Sensitivity)') +
  ggtitle("2 <= Age < 5")+
  theme_bw()
p
ggsave(p, filename = 'notebooks/figs/2-5vlist.pdf', height = 3, width = 4)

data <- read.csv('notebooks/figs/2-vlist.csv') %>%
  filter(FPR < 0.86) %>%
  bind_rows(data_0)
p = data %>%
  ggplot(aes(x = FPR, y = TPR)) +
  geom_point() +
  geom_line() +
  geom_text_repel(aes(label = variable)) +
  geom_hline(yintercept = 0.95, linetype = 'dashed', color = 'red')+
  xlim(0,1) + ylim(0,1) +
  xlab('Fasle Possitive Rate (1 - Specificity)') +
  ylab('True Possitive Rate (Sensitivity)') +
  ggtitle("Age < 2")+
  theme_bw()
ggsave(p, filename = 'notebooks/figs/2-vlist.pdf', height = 3, width = 4)


data <- read.csv('notebooks/figs/5-12vlist.csv') %>%
  filter(TPR < 0.96) %>%
  bind_rows(data_0)
p = data %>%
  ggplot(aes(x = FPR, y = TPR)) +
  geom_point() +
  geom_line() +
  geom_text_repel(aes(label = variable)) +
  geom_hline(yintercept = 0.95, linetype = 'dashed', color = 'red')+
  xlim(0,1) + ylim(0,1) +
  xlab('Fasle Possitive Rate (1 - Specificity)') +
  ylab('True Possitive Rate (Sensitivity)') +
  ggtitle("5 <= Age < 12")+
  theme_bw()
ggsave(p, filename = 'notebooks/figs/5-12vlist.pdf', height = 3, width = 4)

data <- read.csv('notebooks/figs/12+vlist.csv') %>%
  filter(TPR < 0.98) %>%
  bind_rows(data_0)
p = data %>%
  ggplot(aes(x = FPR, y = TPR)) +
  geom_point() +
  geom_line() +
  geom_text_repel(aes(label = variable)) +
  geom_hline(yintercept = 0.95, linetype = 'dashed', color = 'red')+
  xlim(0,1) + ylim(0,1) +
  xlab('Fasle Possitive Rate (1 - Specificity)') +
  ylab('True Possitive Rate (Sensitivity)') +
  ggtitle(" Age >= 12")+
  theme_bw()
ggsave(p, filename = 'notebooks/figs/12+vlist.pdf', height = 3, width = 4)

