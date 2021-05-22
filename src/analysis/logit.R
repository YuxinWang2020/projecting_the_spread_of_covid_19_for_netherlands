library(pglm)
library(haven)
library(dplyr)
options(scipen = 10)

ROOT <- getwd() # set it to your repository root path
SRC <- paste0(ROOT, "/src/")
BLD <- paste0(ROOT, "/bld/")
merge_data <- read_dta(paste0(BLD,"data/liss/logit_merge_data.dta"))
#merge_data <- na.omit(merge_data)
#merge_data['intercept'] <- 1

panal_data <- pdata.frame(merge_data, index = c('personal_id','month'))

x_var <- colnames(panal_data)[! colnames(panal_data) %in% c('infected','personal_id','month','_const')]
fit <- pglm(paste0("infected ~ ", paste(x_var, collapse='+'))
            , data=panal_data, model="random",family = binomial('logit'))
summary(fit)