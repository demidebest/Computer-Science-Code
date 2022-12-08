# Computer-Science-Code
In this research, we focus on the detection of duplicate products. The similarity of product information indicates the likeliness of products to be the same. However, product descriptions are unstructured and often incomplete. This makes recognizing duplicates more complicated. Another challenge is the large number of descriptions that need to be compared. Therefore, a scalable solution is necessary. In this research, Local Sensitivity Hashing (LSH) is used to eliminate pairs of product descriptions that do not contain similar information. Further, we apply a Detection heuristic to detect duplicates for the remaining candidate pairs.

The different functions in the code:
- bootstrap: Get training and test sample 
- getID: Get IDs
- getTitle: Get titles
- getCleanTitle: Get clean titles
- getCleanFeature: Get clean feature 
- getShingles: Get Shingles
- getMinhasing: Get signature matrix
- getBuckets: Get number of matches for all pairs
- getBucketsThreshold: Get candidate pairs
- findBrand: Find brand name of each product
- checkWebsiteAndBrandAndID: Get final pairs by comparing the website, brand and model ID for all candidate pairs
- findModelID: Get (possible) model ID for all products
- getDuplicates: Get threshold for training data
- getDuplicatesTest: Get duplicates for test data
- checkFeature: Check whether feature consists of at least one number and one letter
- checkResults: Get TP, FP, TN and FN
- getNumberPairs: Get total number of pairs
- main: Main method which evaluates 5 bootstrap samples

To use the code:
- the path should be adjusted to use the correct data set
- the number of bootstraps (bootstrapIt) can be adjusted
- the number of permutations for min hashing (numberMinhashing) can be adjusted 
- the minimum percentage of the true duplicates that are labeled as candidate pairs (beta) (to optimize the threshold) can be adjuisted 

