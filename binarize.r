binarize <- function(data, sFeatureName='', testData=c())
{
  lstCatogeries = unique(data)
  #print(sFeatureName)
  #sort(lstCatogeries)
  newData = matrix(0, nrow=length(data), ncol=length(lstCatogeries))
  newTestData = matrix(0,0,0)
  
  for (i in 1:length(data))
  {
    newData[i, match(data[i], lstCatogeries)] = 1
  }
  
  if (sFeatureName != '')
  {
    newFeatureNames = c()
    for(i in 1:length(lstCatogeries))
    {
      newFeatureNames[i] = paste(sFeatureName, '_', i, sep='')
    }
    colnames(newData) = newFeatureNames
  }
  
  if (length(testData) != 0)
  { 
    newTestData = matrix(0, nrow=length(testData), ncol=length(lstCatogeries))
    
    for(j in 1:length(testData))
    {
      newTestData[j, match(testData[j], lstCatogeries)] = 1
    }
    
    colnames(newTestData) = newFeatureNames
    
    return(list('train' = newData, 'test' = newTestData))
  }
  
  return(list('train' = newData, 'test' = testData))
}

clusters <- function(x, centers) {
  # compute squared euclidean distance from each sample to each cluster center
  tmp <- sapply(seq_len(nrow(x)), function(i) apply(centers, 1, function(v) sum((x[i, ]-v)^2)))
  #print(head(t(tmp)))
  #print(head(max.col(-t(tmp))))
  return(max.col(-t(tmp)))  # find index of min distance
}
