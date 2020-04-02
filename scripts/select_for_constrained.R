library(readr)
library(dplyr)

df.bounds  <- pipe("grep -v 'OK' results/crcns_bounds_check.tbl | perl -pe 's/.*crcns_(.+?)_xval.*/$1/g'") %>%
    read_tsv(col_names=c("cell"))

df.stats  <- pipe("grep -v error < results/crcns_data.tbl") %>%
    read_tsv()

df.regions  <- read_csv("/home/data/crcns/cell_regions.csv", col_names=c("cell", "area"))

to.run  <- filter(df.stats, duration > 40000, spikes > 500, eo.cc > 0.4) %>%
    inner_join(filter(df.regions, area!="None"), by="cell") %>%
    inner_join(df.bounds)

write_tsv(to.run, "config/crcns_torun.tbl")
