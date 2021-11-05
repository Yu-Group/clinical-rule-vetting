# download data from https://data.mendeley.com/datasets/3868pbf375/2
# need to convert from IBM SPSS Statistics Files sav file to csv
install.packages('foreign')

library(foreign)

d = read.spss("exaggerate_rawdata.sav", to.data.frame=TRUE)
write.table(d, "exaggerate_rawdata.csv", sep = ",")

