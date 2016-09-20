# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(e1071)
library(cluster)
#library(mclust)
#library(fpc)
suppressMessages(library(AUC))
suppressMessages(library(randomForest))


options(error=recover)

clusters <- function(x, centers) {
  # compute squared euclidean distance from each sample to each cluster center
  tmp <- sapply(seq_len(nrow(x)), function(i) apply(centers, 1, function(v) sum((x[i, ]-v)^2)))
  #print(head(t(tmp)))
  #print(head(max.col(-t(tmp))))
  return(max.col(-t(tmp)))  # find index of min distance
}


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

source('/home/nitsan/research/Kaggle/santander/scripts/binarize.r')
set.seed(123456)


train_file = '/home/nitsan/research/Kaggle/santander/data/train.csv-processed.csv'
test_file = '/home/nitsan/research/Kaggle/santander/data/test.csv-processed.csv'
#train = read.table(train_file, header = TRUE, sep=',')
#save(train, file = 'train')
load('train')

#Train and Validation set

pos_index = c(1:nrow(train))[train$TARGET == 1]
neg_index = c(1:nrow(train))[train$TARGET == 0]
validation_neg = sample(neg_index, length(neg_index)/2, replace = FALSE)
validation_pos = sample(pos_index, length(pos_index)/2, replace = FALSE)
newtrain_neg = setdiff(neg_index, validation_neg)
newtrain_pos = setdiff(pos_index, validation_pos)
newtrain = rbind(train[newtrain_neg, ], train[newtrain_pos, ])
validation = rbind(train[validation_neg, ], train[validation_pos, ])

print(dim(newtrain))
print(dim(validation))

#XGBoost
res_xgb = c()
if(FALSE)
{
  param <- list( "objective" = "binary:logistic",booster = "gbtree",
                 "eval_metric" = "auc", colsample_bytree = 0.85,
                 subsample = 0.95, nthread = 10)
  
  #xgbm_cv <- xgb.cv(params = param , data = as.matrix(newtrain[,-c(1,ncol(newtrain))]),
  #                  nrounds = 300, nfold=10, label = newtrain[, ncol(newtrain)],
  #                  prediction = FALSE, showsd = TRUE, verbose = TRUE)
  #print(xgbm_cv) 
  #AUC was highest in 15th round during cross validation - 6-04-2016
  #Training auc was 0.9 in 272 round
  
  xgbmodel <- xgboost(data = as.matrix(newtrain[, -c(1,ncol(newtrain))]),
                      params = param, max.depth = 5, eta = 0.03,
                      label = newtrain[, ncol(newtrain)], nrounds = 272)
  
  res_xgb <- predict(xgbmodel, newdata = data.matrix(validation[, -c(1, ncol(validation))]))
  
  print("XGB: ")
  print(auc(roc(res_xgb, as.factor(validation$TARGET))))
  #rm(xgbm_cv)
  rm(xgbmodel)
  gc()
  
}

#Random Forest
res_rf = c()
if(FALSE)
{
  #rf_cv = rfcv(trainx = ,data.frame(newtrain[, -c(1, ncol(newtrain))]),
  #             trainy = as.factor(newtrain[, ncol(newtrain)]), cv.fold=10,
  #             scale="log")
  
  #print(rf_cv)
  #with(rf_cv, plot(n.var, error.cv, log="x", type="o", lwd=2))
  #ntree 36
  
  ###############################################
  #NOT Getting Importance Does better 
  #predictors with Gini importance value
  #import <- importance(rf_train, sort = TRUE)
  #print(dim(import))
  #print(head(import))
  #reordering of predictors by importance
  #import <- import[order(import[, 4], decreasing=TRUE), ]
  
  #19 predictors listed by Gini importance
  #print((import[1:19, ]))
  #print(rownames(import)[1:19])
  #rf_train = randomForest(x = data.frame(newtrain[, rownames(import)[1:19]]),
  #                        y = as.factor(newtrain[, ncol(newtrain)]), ntree = 36, scale="log", do.trace=1)
  ################################################
  
  #Mtry Tune
  #rf_train = tuneRF(x = data.frame(newtrain[, -c(1, ncol(newtrain))]), y = as.factor(newtrain[, ncol(newtrain)]),
  #                  stepFactor=2, improve=0.05, trace=TRUE, plot=TRUE, doBest=TRUE)
  
  ######################################################
  
  #Defult RF seems to be good 0.55 AUC
  rf_train = randomForest(x = data.frame(newtrain[, -c(1, ncol(newtrain))]),
                          y = as.factor(newtrain[, ncol(newtrain)]), 
                         scale="log", do.trace=1, importance = TRUE)
  

  res_rf = predict(rf_train, newdata =  data.matrix(validation[-c(1, ncol(validation))]), type = "prob")
  res_rf = res_rf[, 2]
  
  print("RF:")
  print(auc(roc(res_rf, as.factor(validation$TARGET))))
  
  #rm(rf_cv)
  rm(rf_train)
  gc()
}

#SVM
if(TRUE)
{
  svm_tune = tune(svm, train.x = data.frame(newtrain[, -c(1, ncol(newtrain))]), train.y = as.factor(newtrain[, ncol(newtrain)]),
                  ranges = list(gamma = 2^(-1:2), cost = 2^(-3:4)),
                  cross = 5, cachesize = 5000, kernel = "radial", probability = TRUE,
                  tunecontrol = tune.control(sampling = "cross"), trace = TRUE)
  
  print(summary(svm_tune))
  plot(svm_tune)
  print(svm_tune)

  #svm_model = svm(x_train, y_train, type = 'C-classification', kernel = "radial", 
  #                gamma =  1 / ncol(x_train), cost = 1,
  #                cachesize = 1000, probability = TRUE, fitted = TRUE)
  
  #print(summary(svm_model))
  
  res_svm = predict(svm_tune.best.model, newdata = data.matrix(validation[-c(1, ncol(validation))]),
                    probability = TRUE)
  
  print(head(res_svm))
  
  print(auc(roc(res_svm, as.factor(validation$TARGET))))
}

#K-MEans
if(FALSE)
{
  
  km_train = kmeans(data.frame(newtrain[, -c(1, ncol(newtrain))]), 2, nstart=100, iter.max = 1000)
  #print(km_train)
  #plotcluster(data.frame(newtrain[, -c(1, ncol(newtrain))]), k_cluster$cluster)
  #plotcluster(data.frame(newtrain[, -c(1, ncol(newtrain))]), newtrain[, ncol(newtrain)])
  #plot(newtrain[2, ], newtrain[3, ], col = km_train$cluster)
  print(head(km_train$cluster))
  print('KMeans: ')
  print(auc(roc(km_train$cluster, newtran$TARGET)))
  
}


#Combined Ensumble
if(FALSE)
{
  #print("RF")
  #print(head(res_rf))
  #print("XGB")
  #print(head(res_xgb))
  res_ensumble_mean = (res_rf + res_xgb)/2
  #res_ensumble_mean = -1 * res_ensumble_mean
  print(head(res_ensumble_mean))
  print("EN: Mean: ")
  print(auc(roc(res_ensumble_mean, as.factor(validation$TARGET))))
  
  res_ensumble_gmean = sqrt(res_rf * res_xgb)
  print(head(res_ensumble_gmean))
  #res_ensumble_gmean = -1 * res_ensumble_gmean
  print("EN: Geo Mean: ")
  print(auc(roc(res_ensumble_gmean, as.factor(validation$TARGET))))
  
  if(TRUE)
  {
      pos_index = c(1:nrow(validation))[validation$TARGET == 1]
      neg_index = c(1:nrow(validation))[validation$TARGET == 0]
      sub_neg = sample(neg_index, length(neg_index)/2, replace = FALSE)
      sub_pos = sample(pos_index, length(pos_index)/2, replace = FALSE)
      subtrain_neg = setdiff(neg_index, sub_neg)
      subtrain_pos = setdiff(pos_index, sub_pos)
      #subtrain = rbind(validation[subtrain_neg, ], validation[subtrain_pos, ])
      #subvalidation = rbind(validation[sub_neg, ], validation[sub_pos, ])
      #print(dim(subtrain))
      #print(dim(subvalidation))
      
      rf_subtrain = c(res_rf[subtrain_neg], res_rf[subtrain_pos])
      rf_subvalidation = c(res_rf[sub_neg], res_rf[sub_pos])
      xgb_subtrain = c(res_xgb[subtrain_neg], res_xgb[subtrain_pos])
      xgb_subvalidation = c(res_xgb[sub_neg], res_xgb[sub_pos])
      
      en_sub_train = cbind(rf_subtrain, xgb_subtrain)
      en_sub_test = cbind(rf_subvalidation, xgb_subvalidation)
      en_y_train = c(validation$TARGET[subtrain_neg], validation$TARGET[subtrain_pos])
      en_y_test = c(validation$TARGET[sub_neg], validation$TARGET[sub_pos])
      
      param <- list( "objective" = "binary:logistic",booster = "gbtree",
                     "eval_metric" = "auc", colsample_bytree = 0.85,
                     subsample = 0.95, nthread = 10)
      
      #en_xgbm_cv <- xgb.cv(params = param , data = as.matrix(en_sub_train),
      #                  nrounds = 300, nfold=10, label = en_y_train,
      #                  prediction = FALSE, showsd = TRUE, verbose = TRUE)
      
      en_xgbmodel <- xgboost(data = as.matrix(en_sub_train),
                          params = param, max.depth = 5, eta = 0.03,
                          label = en_y_train, nrounds = 220)
      
      res_en_xgb <- predict(en_xgbmodel, newdata = data.matrix(en_sub_test))
      
      print("EN XGB: ")
      print(auc(roc(res_en_xgb, as.factor(en_y_test))))
      
      #rm(en_xgbm_cv)
      rm(en_xgbmodel)
      gc()
      
  }
  
}

rm(newtrain)
rm(validation)
gc()

###########################################################################################################
if(FALSE)
{
  test  = read.table(test_file, header = TRUE, sep=',')
  save(test, file='test')
  
  print(dim(train))
  print(dim(test))
  
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
  
  if(FALSE)
  {
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
  
  #res <- data.frame(ID = test[, 1], TARGET = res)
  
  #write.csv(res, "lk-xgboost-submission.csv", row.names = FALSE)
}
