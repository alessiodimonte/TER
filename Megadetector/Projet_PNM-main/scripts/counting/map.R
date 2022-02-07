library(leaflet)
library(readODS)
library(dplyr)

coord_pieges <- read_ods(path='workspace/fsimoes/videos_mercantour_23_06_2020/coordonnes_pieges-photos_2020_4.ods', sheet=2)

csv_chevreuil <- 'workspace/fsimoes/counting/final_number_species_chevreuil_2020-03-10.csv'
csv_chamois <- 'workspace/fsimoes/counting/final_number_species_chamois_2020-03-10.csv'
csv_humain <- 'workspace/fsimoes/counting/final_number_species_humain_2020-03-10.csv'

#########################################################################

coord_pieges <- coord_pieges[,c('Maille','N','E')]
colnames(coord_pieges) <- c('maille','N','E')
coord_pieges[c(1:3),]$maille <- sub("^", "0", coord_pieges[c(1:3),]$maille)
coord_pieges$maille <- sub("^", "maille", coord_pieges$maille)


# Chevreuil
data_cheuv <- read.csv2(csv_chevreuil, sep=",",header = TRUE, stringsAsFactors = FALSE)
#remove "vide"
data_cheuv <- data_cheuv %>% filter(detection_classes != 'vide')
data_cheuv <- data_cheuv[,c('maille','date','detection_classes','nbr_species_vid')]

# Final data
df_final <- merge(coord_pieges,data_cheuv,by="maille",all=T)
df_final[is.na(df_final)] <- 0
df_final$nbr_species_vid <- round(as.numeric(df_final$nbr_species_vid),0) 

getColor <- function(df_final) {
  sapply(df_final$nbr_species_vid, function(nbr_species_vid) {
    if(nbr_species_vid > 0) {
      "red"
    } else {
      "blue"
    } })
}

icons <- awesomeIcons(
  icon = 'camera',
  library = 'ion',
  markerColor = getColor(df_final)
)

labs <- lapply(seq(nrow(df_final)), function(i) {
  paste0( '<p>', df_final[i, "maille"], '<p></p>', 
          df_final[i, "nbr_species_vid"])
})

leaflet(df_final) %>% addTiles() %>% addAwesomeMarkers(lng = ~E,lat = ~N, icon=icons, label=~lapply(labs, htmltools::HTML), labelOptions = labelOptions(noHide = T,direction = "upper",
                                                                                                                                     style = list(
                                                                                                                                       "font-family" = "serif",
                                                                                                                                       "box-shadow" = "3px 3px rgba(0,0,0,0.25)",
                                                                                                                                       "font-size" = "7px",
                                                                                                                                       "border-color" = "rgba(0,0,0,0.5)"
                                                                                                                                     )))

# Chamois
data_cham <- read.csv2(csv_chamois, sep=",",header = TRUE, stringsAsFactors = FALSE)
#remove "vide"
data_cham <- data_cham %>% filter(detection_classes != 'vide')
data_cham <- data_cham %>% group_by(date,maille,detection_classes) %>% mutate(total_species=sum(as.numeric(nbr_species_vid)))
data_cham <- data_cham[,c('maille','date','detection_classes','total_species')]
data_cham <- unique(data_cham)
data_cham <- data_cham %>% filter(detection_classes=='chamois')

# Final data
df_final <- merge(coord_pieges,data_cham,by="maille",all=T)
df_final[is.na(df_final)] <- 0
df_final$total_species <- round(as.numeric(df_final$total_species),2) 

getColor <- function(df_final) {
  sapply(df_final$total_species, function(total_species) {
    if(total_species > 0) {
      "red"
    } else {
      "blue"
    } })
}

icons <- awesomeIcons(
  icon = 'camera',
  library = 'ion',
  markerColor = getColor(df_final)
)

labs <- lapply(seq(nrow(df_final)), function(i) {
  paste0( '<p>', df_final[i, "maille"], '<p></p>', 
          df_final[i, "total_species"])
})

leaflet(df_final) %>% addTiles() %>% addAwesomeMarkers(lng = ~E,lat = ~N, icon=icons, label=~lapply(labs, htmltools::HTML), labelOptions = labelOptions(noHide = T,direction = "upper",
                                                                                                                                     style = list(
                                                                                                                                       "font-family" = "serif",
                                                                                                                                       "box-shadow" = "3px 3px rgba(0,0,0,0.25)",
                                                                                                                                       "font-size" = "7px",
                                                                                                                                       "border-color" = "rgba(0,0,0,0.5)"
                                                                                                                                     )))




# Humains
data_hum <- read.csv2(csv_humain, sep=",",header = TRUE, stringsAsFactors = FALSE)
#remove "vide"
data_hum <- data_hum %>% filter(detection_classes != 'vide')
data_hum <- data_hum %>% group_by(date,maille,detection_classes) %>% mutate(total_species=sum(as.numeric(nbr_species_vid)))
data_hum <- data_hum[,c('maille','date','detection_classes','total_species')]
data_hum <- unique(data_hum)
data_hum <- data_hum %>% filter(detection_classes=='humain')

# Final data
df_final <- merge(coord_pieges,data_hum,by="maille",all=T)
df_final[is.na(df_final)] <- 0
df_final$total_species <- round(as.numeric(df_final$total_species),0) 

getColor <- function(df_final) {
  sapply(df_final$total_species, function(total_species) {
    if(total_species > 0) {
      "red"
    } else {
      "blue"
    } })
}

icons <- awesomeIcons(
  icon = 'camera',
  library = 'ion',
  markerColor = getColor(df_final)
)

labs <- lapply(seq(nrow(df_final)), function(i) {
  paste0( '<p>', df_final[i, "maille"], '<p></p>', 
          df_final[i, "total_species"])
})

leaflet(df_final) %>% addTiles() %>% addAwesomeMarkers(lng = ~E,lat = ~N, icon=icons, label=~lapply(labs, htmltools::HTML), labelOptions = labelOptions(noHide = T,direction = "upper",
                                                                                                                                     style = list(
                                                                                                                                       "font-family" = "serif",
                                                                                                                                       "box-shadow" = "3px 3px rgba(0,0,0,0.25)",
                                                                                                                                       "font-size" = "7px",
                                                                                                                                       "border-color" = "rgba(0,0,0,0.5)"
                                                                                                                                     )))






