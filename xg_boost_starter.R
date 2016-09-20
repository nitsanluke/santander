# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(e1071)
suppressMessages(library(AUC))
suppressMessages(library(randomForest))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

source('/home/nitsan/research/Kaggle/santander/scripts/binarize.r')
set.seed(123456)


# Reading the data
if(FALSE)
{
train_file = '/home/nitsan/research/Kaggle/santander/data/train.csv'
test_file = '/home/nitsan/research/Kaggle/santander/data/test.csv'
dat_train <- read.csv(train_file, stringsAsFactors = F)
dat_test <- read.csv(test_file, stringsAsFactors = F)

# Mergin the test and train data
dat_test$TARGET <- NA
all_dat <- rbind(dat_train, dat_test)

# Removing the constant variables
train_names <- names(dat_train)[-1]
for (i in train_names)
{
  #if (class(all_dat[[i]]) == "integer") 
  #{
    u <- unique(all_dat[[i]])
    if (length(u) == 1) 
    {
      all_dat[[i]] <- NULL
    } 
  #}
  #else
  #{
  #  cat(i, '\n')
  #}
}

#Removing duplicate columns
train_names <- names(all_dat)[-1]
fac <- data.frame(fac = integer())    

for(i in 1:length(train_names))
{
  if (i != length(train_names))
  {
    for (k in (i + 1):length(train_names))
    {
      if (identical(all_dat[,i], all_dat[,k]) == TRUE)
      {
        fac <- rbind(fac, data.frame(fac = k))
      }
    }
  }
}
same <- unique(fac$fac)
all_dat <- all_dat[,-same]
cat ("Num Cols:", ncol(all_dat), '\n')
}

train_file = '/home/nitsan/research/Kaggle/santander/data/train.csv-processed.csv'
test_file = '/home/nitsan/research/Kaggle/santander/data/test.csv-processed.csv'
train = read.table(train_file, header = TRUE, sep=',')
test  = read.table(test_file, header = TRUE, sep=',')
#all_dat <- rbind(train, test)
#rm(train)
#rm(test)
#gc()


#print("Running %")
#total_90_rows = nrow(all_dat) * 0.99
#droped_cols = c()
#v = 1
#for(i in 2:ncol(all_dat))
#{
#  if(is.na(all_dat[, i]))
#  {
#    print(colnames(all_dat)[i])
#  }
#  if(sum(all_dat[, i] == 0) >   total_90_rows)
#  {
#    cat(colnames(all_dat)[i], ', ')
#    droped_cols[v] = i
#    v = v + 1
#  }
#}

#exit()
  #Binarization
if(FALSE)
{
  ID = all_dat[, 1]
  Taget = all_dat[, ncol(all_dat)]
  newdata = all_dat[,1]
  num_catogorical_f = 0
  
  for (i in 2:ncol(all_dat)-1)
  {
    #print(colnames(all_dat)[i])
    if (length(unique(all_dat[,i])) < 100)
    {
      binarized_features = binarize(all_dat[, i], colnames(all_dat)[i])
      newdata = cbind(newdata, as.matrix(binarized_features$train))
      num_catogorical_f = num_catogorical_f + 1
    }
    else
    {
      newdata = cbind(newdata, all_dat[, i])
      colnames(newdata)[ncol(newdata)] = colnames(all_dat)[i]
    }
  }
  
  rm(all_dat)
  gc()
  print("B4")
  all_dat = cbind(newdata, Taget)
  print("A4")
  all_dat = all_dat[, 2:ncol(all_dat)]
  colnames(all_dat)[ncol(all_dat)] = "TARGET"
  cat ("Num Cols:", ncol(all_dat), "\tNum catogorical F: ", num_catogorical_f, '\n')
  #cat(colnames(all_dat))
  
  rm(newdata)
  gc()
}

if(FALSE){
# Splitting the data for model
train <- all_dat[1:nrow(dat_train), ]
test <- all_dat[-(1:nrow(dat_train)), ]

rm(all_dat)
gc()

write.csv(train, file = paste(train_file, '-binarize-2.csv', sep = ''), row.names = FALSE)
write.csv(test, file = paste(test_file, '-binarized-2.csv', sep = ''), row.names = FALSE)
}

#Read data
#train_file = '/home/nitsan/research/Kaggle/santander/data/train.csv-processed.csv'
#test_file = '/home/nitsan/research/Kaggle/santander/data/test.csv-processed.csv'
#train = read.table(train_file, header = TRUE, sep=',')
#test  = read.table(test_file, header = TRUE, sep=',')
print(dim(train))

if(FALSE)
{
#Oversampling
#P:N => 1:24
pos_index = c(1:nrow(train))[train$TARGET == 1]
neg_index = c(1:nrow(train))[train$TARGET == 0]

#print(length(pos_index))
#print(length(neg_index))
#for(i in 1:10) {
#cat("nrounds: ", i*25, '\t')
#set.seed(123456)

#sub_test_neg = sample(neg_index, 25000, replace = FALSE)
#sub_test_pos = sample(pos_index, 1000, replace = FALSE)

#sub_train_pos = setdiff(pos_index, sub_test_pos)
#sub_train_neg = setdiff(neg_index, sub_test_neg)

#print(length(sub_train_pos))

#sub_train = train[c(sub_train_pos, sub_train_pos, sub_train_pos, 
#                    sub_train_pos, sub_train_pos,
#                    sub_train_neg), ]

#sub_train = train[c(sub_train_pos, sub_train_neg), ]
#sub_train = sub_train[sample(nrow(sub_train)), ]
#print(dim(sub_train))

#sub_test = train[c(sub_test_pos, sub_test_neg), ]
#sub_test = sub_test[sample(nrow(sub_test)), ]

#print(dim(sub_train))
#print(dim(sub_test))
#create new copy of positive data

#iCopies = 5

#pos_data = train[sub_train_pos, ]

#FullData
pos_data = train[pos_index, ]
duplicatedCopies = matrix(0, ncol=(ncol(pos_data)-2), 0)

for (iCopies in 1:2){ 
for(k in 1:1){
  duplicatedCopies = matrix(0, ncol=(ncol(pos_data)-2), 0)
  for (n in 1:iCopies)
  {
    
    duplicateData = matrix(0, ncol=ncol(pos_data)-2, nrow=nrow(pos_data))
    #for (i in 1:nrow(duplicateData))
    #{
        for (j in 1:ncol(duplicateData))
        {
            if (length(unique(pos_data[,j+1])) < 500)
            {
              duplicateData[, j] = pos_data[, j+1] 
            }
            else
            {
              duplicateData[, j] = pos_data[, j+1] + rnorm(n = nrow(duplicateData), mean = 0.0, sd = 0.01) 
            }
        }
        
        duplicatedCopies = rbind(duplicatedCopies, duplicateData)
        rm(duplicateData)
    #}
  }
  duplicateData = duplicatedCopies
  rm(duplicatedCopies)
  gc()
  
  colnames(duplicateData) = colnames(pos_data)[2:(ncol(pos_data)-1)]
  duplicateData = cbind(ID = pos_data$ID, duplicateData, TARGET = pos_data$TARGET)

  #Full data
  train = rbind(train, duplicateData)
  train = train[sample(nrow(train)), ]	

  #print(colnames(duplicateData))
  #print(colnames(sub_train))
  
  #sub_train = rbind(sub_train, duplicateData)
  #sub_train = sub_train[sample(nrow(sub_train)), ]
  #print(dim(sub_train))
  
  #write.csv(sub_train, 'test')


#param <- list( "objective" = "binary:logistic",booster = "gbtree",
#               "eval_metric" = "auc", colsample_bytree = 0.85,
 #              subsample = 0.95, nthread = 10)

#y <- as.numeric(sub_train[, ncol(sub_train)])

#xgbmodel <- xgboost(data = as.matrix(sub_train[,-c(1,ncol(sub_train))]), 
#                    params = param,
#                    max.depth = 5, eta = 0.03,
#                    label = y, verbose = 0,
#                    nrounds = k*50)

#res <- predict(xgbmodel, newdata = data.matrix(sub_test[,-c(1,ncol(sub_test))]))

#cat("nrounds: ", (k*50), "\tNum Copy: ", iCopies, '\t', "Test AUC: ")
#print(auc(roc(res, as.factor(sub_test$TARGET))))

}
#}
}
}

print(dim(train))
train = train[, c(1,3,53,ncol(train))]
#train[, 3] = as.factor(train[, 3])
print(colnames(train))
test = test[, c(1,3,53,ncol(test))]
#test[, 3] = as.factor(test[, 3])
print(colnames(train[ , -c(1,ncol(train))]))
#pos_index = c(1:nrow(train))[train$TARGET == 1]
#neg_index = c(1:nrow(train))[train$TARGET == 0]
#train = train[c(pos_index, pos_index, pos_index,
#                neg_index), ]
#train = train[sample(nrow(train)), ]
#print(dim(train))


###########################################################################################################

#Building the model
param <- list( "objective" = "binary:logistic",booster = "gbtree",
               "eval_metric" = "auc", colsample_bytree = 0.85,
               subsample = 0.95, nthread = 10)

y <- as.factor(train[, ncol(train)])


#xgbm_cv <- xgb.cv(params = param , data = as.matrix(train[,-c(1,ncol(train))]),
 #                 nrounds = 300, nfold=10, label = y,
  #                prediction = FALSE, showsd = TRUE,
   #               verbose = TRUE)

#print(xgbm_cv)
#AUC was highest in 288th round during cross validation
#AUC was highest in 27th round during cross validation

if(TRUE){
xgbmodel <- xgboost(data = as.matrix(train[,-c(1,ncol(train))]), 
                    params = param,
                    max.depth = 5, eta = 0.03,
                    label = y,
                    nrounds = 300)

#Prediction
#Remove ID & Target
res <- predict(xgbmodel, newdata = data.matrix(test[,-c(1,ncol(test))]))

rm(xgbmodel)
gc()
}

if(FALSE){
#Random Forest
rf_cv = rfcv(trainx = ,data.frame(train[, -c(1, ncol(train))]),
             trainy = as.factor(y), cv.fold=10, 
             scale="log")

print(rf_cv)
with(rf_cv, plot(n.var, error.cv, log="x", type="o", lwd=2))
  
#rf_train = randomForest(x = data.frame(train[, -c(1, 310)]), 
#                              y = as.factor(y),
#                              ntree = 1000, scale="log",
#                              do.trace=1)

#res_rf = predict(rf_train, newdata =  data.matrix(test[-c(1, 310)]), type = "prob")

#res = sqrt(res * res_rf)
#res = res_rf
}

res <- data.frame(ID = test[, 1], TARGET = res)

write.csv(res, "lk-xgboost-submission.csv", row.names = FALSE)
