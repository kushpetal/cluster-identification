# SF Validity Index
###### A Python implementation of a bounded validity index for clustering

The SF (score function) Index is based off maximizing the inter-cluster distance while minimizing the intra-cluster distance.

A **higher** value for the SF score indicates a more suitable number of clusters. 

## Demonstration

| Simulated Data  | SF Scores Plot      
|----------|:-------------:|
| <img src="/example/clusters.png" width="350" height="300">  |  <img src="/example/sfscore.png" width="350" height="300"> |
  
- The highest SF Index is measured when the number of clusters is 3 which accurately identifies the true number of clusters

**Study: [A Comprehensive Validity Index For Clustering](https://www.semanticscholar.org/paper/A-comprehensive-validity-index-for-clustering-Saitta-Raphael/d9330f539f2c96c43a2967ab9b1db8cdbfc7f572)**
