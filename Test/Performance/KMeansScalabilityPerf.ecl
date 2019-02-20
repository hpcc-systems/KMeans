/**
  * The use of this code is to test the scalability of ECL KMeans Clustering Algorithm
  * using the standard public dataset Iris[1].
  * Reference
  * [1] Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
  *     [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
  *     School of Information and Computer Science.
  */
IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT Test;
IMPORT Cluster;

//Data Preperation
//Load Iris Dataset
ds := Test.Datasets.DSIris.ds;
//Test the scalability by increasing the size of the original data
//Define N -- increase the volume of the data by N times.
N := 1000000;
multi_ds := NORMALIZE(ds, N, TRANSFORM(RECORDOF(ds), SELF := LEFT));
ML_Core.AppendSeqId(multi_ds, id, ids);
ML_Core.ToField(ids, idsWi);
d01 := idsWi(number < 5);
centroidsID := [1,51,101];
d02 := d01(id IN centroidsID);
//Set up the parameters
max_iteratons := 30;
tolerance := 0.0;
//Fit KMeans model with the samples d01 and the centroids d02
Model := Cluster.KMeans(max_iteratons, tolerance).fit(d01, d02);
//Coordinates of cluster centers
Centroids := Cluster.KMeans().Centers(Model);
OUTPUT(Centroids);