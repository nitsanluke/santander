# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(e1071)
library(cluster)
library(party)
library(glmnet)
library(methods)

#library(mclust)
#library(fpc)
suppressMessages(library(AUC))
suppressMessages(library(randomForest))


#options(error=recover)

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

#write.csv(newtrain, "newtrain.csv", row.names = FALSE)
#write.csv(validation, "validation.csv", row.names = FALSE)

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
                      label = newtrain[, ncol(newtrain)], nrounds = 272, verbose = F)
  
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

#CTree
res_ctree = c()
if(FALSE)
{
  varNameVec = colnames(newtrain)[-c(1, ncol(newtrain))]
  formula = as.formula(paste("TARGET ~", paste(varNameVec, collapse = "+")))
  #print(formula)
  
  ctree_model = ctree(formula, data=as.data.frame(newtrain))
  
  plot(ctree_model)
  
  res_ctree = predict(ctree_model, newdata = as.data.frame(validation[-c(1, ncol(validation))]), 
                      type = "prob")
  
  res_ctree = unlist(res_ctree)
  #print(head(res_ctree))
  print("CTREE:")
  print(auc(roc(res_ctree, as.factor(validation$TARGET))))
  rm(ctree_model)
  gc()
}

#Logistic Regression
res_glm = c()
if(FALSE)
{
  varNameVec = colnames(newtrain)[-c(1, ncol(newtrain))]
  formula = as.formula(paste("TARGET ~ 1 + ", paste(varNameVec, collapse = "+")))
  #print(formula)
  
  #glm_model = glm(formula, family=binomial(link='logit'), data=as.data.frame(newtrain))
  
  glm_model = cv.glmnet(x=model.matrix(formula, newtrain),
                        y=newtrain$TARGET,
                        family="binomial", nfolds=5, alpha=0.5,
                        parallel=TRUE, thresh=1E-10)
  

  #print(glm_model)
  #print(glm_model$fit)
  print(glm_model$lambda.min)
  res_glm = predict(glm_model, newx = model.matrix(formula, validation), type="response", s = "lambda.min")
  #print(res_glm)
  
  print("GLM:")
  print(auc(roc(res_glm, as.factor(validation$TARGET))))
  rm(glm_model)
  gc()
}

#Combined Ensumble
if(FALSE)
{
  #print("RF")
  #print(head(res_rf))
  #print("XGB")
  #print(head(res_xgb))
  #res_ensumble_mean = (res_ctree + res_xgb + res_glm)/3
  res_ensumble_mean = (res_xgb + res_glm)/2
  #res_ensumble_mean = -1 * res_ensumble_mean
  #print(head(res_ensumble_mean))
  print("EN: Mean: ")
  print(auc(roc(res_ensumble_mean, as.factor(validation$TARGET))))
  
  #res_ensumble_gmean = (res_ctree * res_xgb * res_glm)^(1/3)
  res_ensumble_gmean = (res_xgb * res_glm)^(1/2)
  #print(head(res_ensumble_gmean))
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
      
      res_rf = res_ctree
      
      rf_subtrain = c(res_rf[subtrain_neg], res_rf[subtrain_pos])
      rf_subvalidation = c(res_rf[sub_neg], res_rf[sub_pos])
      xgb_subtrain = c(res_xgb[subtrain_neg], res_xgb[subtrain_pos])
      xgb_subvalidation = c(res_xgb[sub_neg], res_xgb[sub_pos])
      glm_subtrain = c(res_glm[subtrain_neg], res_glm[subtrain_pos])
      glm_subvalidation = c(res_glm[sub_neg], res_glm[sub_pos])
      
      #en_sub_train = cbind(rf_subtrain, xgb_subtrain, glm_subtrain)
      #en_sub_test = cbind(rf_subvalidation, xgb_subvalidation, glm_subvalidation)
      #en_y_train = c(validation$TARGET[subtrain_neg], validation$TARGET[subtrain_pos])
      #en_y_test = c(validation$TARGET[sub_neg], validation$TARGET[sub_pos])
      
      en_sub_train = cbind(xgb_subtrain, glm_subtrain)
      en_sub_test = cbind(xgb_subvalidation, glm_subvalidation)
      en_y_train = c(validation$TARGET[subtrain_neg], validation$TARGET[subtrain_pos])
      en_y_test = c(validation$TARGET[sub_neg], validation$TARGET[sub_pos])
      
      param <- list( "objective" = "binary:logistic",booster = "gbtree",
                     "eval_metric" = "auc", colsample_bytree = 0.85,
                     subsample = 0.95, nthread = 10)
      
      #en_xgbm_cv <- xgb.cv(params = param , data = as.matrix(en_sub_train),
      #                  nrounds = 500, nfold=10, label = en_y_train,
      #                  prediction = FALSE, showsd = TRUE, verbose = TRUE)
      #iterations - 8
      
      #colnames(en_sub_train) = c('x1', 'x2', 'x3')
      colnames(en_sub_train) = c('x1', 'x2')
      #colnames(en_y_train) = c('y')
      #colnames(en_sub_test) = c('x1', 'x2', 'x3')
      colnames(en_sub_test) = c('x1', 'x2')
      
      em_combined = cbind(en_y_train, en_sub_train)
      #colnames(em_combined) = c('y', 'x1', 'x2', 'x3')
      colnames(em_combined) = c('y', 'x1', 'x2')
      
      en_xgbmodel <- xgboost(data = as.matrix(en_sub_train),
                          params = param, max.depth = 5, eta = 0.03,
                          label = en_y_train, nrounds = 8, verbose = F)
      
      res_en_xgb <- predict(en_xgbmodel, newdata = data.matrix(en_sub_test))
  
      print("EN XGB: ")
      print(auc(roc(res_en_xgb, as.factor(en_y_test))))
      
      en_linmodel <- lm(y ~ x1 + x2, data=as.data.frame(em_combined))
      res_en_lm = predict(en_linmodel, newdata = as.data.frame(en_sub_test))    
      print("EN Lin")
      print(auc(roc(res_en_lm, as.factor(en_y_test))))
      
      formula = as.formula('y ~ 1 + x1 + x2')
      en_glmmodel = cv.glmnet(x=model.matrix(formula, as.data.frame(em_combined)),
                              y=en_y_train, family="binomial", nfolds=5, alpha=0.5,
                              parallel=TRUE, thresh=1E-10)
      
      #en_glmmodel = glm(y ~ x1 + x2, family=binomial(link='logit'), data=as.data.frame(em_combined))
      #res_en_glm = predict(en_glmmodel, newdata = as.data.frame(en_sub_test))
      
      res_en_glm = predict(en_glmmodel, newx = model.matrix(formula, en_sub_test), type="response", s = "lambda.min")
      print("EN GLM")
      print(auc(roc(res_en_glm, as.factor(en_y_test))))
      
      
      #en_rfmodel = randomForest(x = as.data.frame(en_sub_train),
       #                       y = as.factor(en_y_train), 
        #                      scale="log", do.trace=1, importance = TRUE)
      
      #res_en_rf = predict(en_rfmodel, newdata = as.data.frame(en_sub_test), type = "prob")
      #res_en_rf = res_en_rf[, 2]
      
      #print("EN RF")
      #print(auc(roc(res_en_rf, as.factor(en_y_test))))

      #rm(en_xgbm_cv)
      rm(en_xgbmodel)
      gc()
      
  }
  
}

#rm(newtrain)
#rm(validation)
#gc()

###########################################################################################################
if(TRUE)
{
  #test  = read.table(test_file, header = TRUE, sep=',')
  #save(test, file='test')
  load('test')
  
  #print(dim(train))
  print(dim(test))
  
  #Building the model
  param <- list( "objective" = "binary:logistic",booster = "gbtree",
                 "eval_metric" = "auc", colsample_bytree = 0.85,
                 subsample = 0.95, nthread = 10)
  
  
 # y <- as.factor(newtrain[, ncol(newtrain)])
  
  
  #xgbm_cv <- xgb.cv(params = param , data = as.matrix(train[,-c(1,ncol(train))]),
   #                 nrounds = 300, nfold=10, label = y,
    #                prediction = FALSE, showsd = TRUE,
     #               verbose = TRUE)
  
  #print(xgbm_cv)
  
  if(TRUE)
  {
    xgbmodel <- xgboost(data = as.matrix(newtrain[,-c(1,ncol(newtrain))]), 
                        params = param,
                        max.depth = 5, eta = 0.03,
                        label = newtrain[, ncol(newtrain)],
                        nrounds = 272, verbose = F)
    
    #Prediction
    #Remove ID & Target
    en_res_xgb <- predict(xgbmodel, newdata = data.matrix(validation[,-c(1, ncol(validation))]) )
    res_xgb <- predict(xgbmodel, newdata = data.matrix(test[, -c(1, ncol(test))]))
    print("XGB:")
    print(auc(roc(en_res_xgb, as.factor(validation$TARGET))))
    #rm(xgbmodel)
    #gc()
  }
  
  #CTree
  if(FALSE){
    
    varNameVec = colnames(newtrain)[-c(1, ncol(newtrain))]
    formula = as.formula(paste("TARGET ~ ", paste(varNameVec, collapse = "+")))
    #ctree_model = ctree(formula, data=as.data.frame(newtrain))
    
    #print(ctree_model) 
    
    #en_res_ctree = predict(ctree_model, newdata = as.data.frame(validation), type = "prob")
    #en_res_ctree = unlist(en_res_ctree)
    
    miss_match_cols = which(sapply(test, class) != sapply(newtrain, class))
    print(sapply(newtrain[, miss_match_cols], unique))
    print(sapply(test[, miss_match_cols], unique))    
    #levels(test$TARGET) = levels(newtrain$TARGET) 
    res_ctree = predict(ctree_model, newdata = (test), type = "prob")
    res_ctree = unlist(res_ctree)
    print("CTREE:")
    print(auc(roc(en_res_ctree, as.factor(validation$TARGET))))
  
  }
  
  #Glmnet
  if(TRUE){
    varNameVec = colnames(newtrain)[-c(1, ncol(newtrain))]
    formula = as.formula(paste("TARGET ~ 1 + ", paste(varNameVec, collapse = "+")))
    glm_model = cv.glmnet(x=model.matrix(formula, newtrain),
                          y=newtrain$TARGET,
                          family="binomial", nfolds=5, alpha=0.5,
                          parallel=TRUE, thresh=1E-10)
    
    en_res_glm = predict(glm_model, newx = model.matrix(formula, validation), type="response", s = "lambda.min")
    res_glm = predict(glm_model, newx = model.matrix(formula, test), type="response", s = "lambda.min")
    print("GLment:")
    print(auc(roc(en_res_glm, as.factor(validation$TARGET))))
  }
  
  en_train = cbind(validation$TARGET, en_res_xgb, en_res_glm)
  colnames(en_train) = c('y', 'x1', 'x2')
  en_test = cbind(res_xgb, res_glm)
  colnames(en_test) = c('x1', 'x2')
  print(dim(en_test))

  en_linmodel <- lm(y ~ x1 + x2, data=as.data.frame(en_train))
  print(en_linmodel)
  res_en_lm = predict(en_linmodel, newdata = as.data.frame(en_test))    
  
  print(length(res_en_lm))
  res <- data.frame(ID = test[, 1], TARGET = res_en_lm)
  write.csv(res, "lk-en-xgboost-submission.csv", row.names = FALSE)
  print("finish")

}
