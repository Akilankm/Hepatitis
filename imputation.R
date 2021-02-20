
library(mice)
df <- read.csv('hepatitis_csv.csv',na.strings=c("na","NA"," ",""))
res <- mice(data=df,m=5,method="pmm",maxiter=5)
complete(res,1) -> result
write.csv(result,"hepatitis_csv_imputation.csv",row.names=FALSE)
