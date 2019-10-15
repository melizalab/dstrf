library(readr)
library(dplyr)

df.bounds  <- pipe("perl -pe 's/.*crcns_(.+?)_xval.*/$1/g' < build/crcns_initial_params.tbl") %>%
    read_tsv(col_names=c("cell"))

df.stats  <- pipe("grep -v error < build/crcns_data.tbl") %>%
    read_tsv()

df.regions  <- read_csv("crcns/cell_regions.csv", col_names=c("cell", "area"))

to.run  <- filter(df.stats, duration > 40000, spikes > 500, eo.cc > 0.4) %>%
    inner_join(filter(df.regions, area!="None"), by="cell") %>%
    inner_join(df.bounds)

write_tsv(to.run, "build/crcns_needs_constrained.csv")
