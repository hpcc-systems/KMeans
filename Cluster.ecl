/**
  * Classic KMeans Clustering Algorithm.
  *
  * <p>Clustering Algorithms is a branch of unsupervised machine learning algorithms
  * to automatically catogarize observations(points) into groups without pre-defined
  * lables. KMeans[1] is one of the most well-known clustering algorithms. Giving the data
  * points for clustering and the K initial centroids of each cluster, KMeans algorithm
  * can automatically group each sample into one cluster.
  *
  * <p>Reference.
  * [1] Hartigan, J. A., & Wong, M. A. (1979). Algorithm AS 136: A k-means clustering algorithm.
  * Journal of the Royal Statistical Society. Series C (Applied Statistics), 28(1), 100-108.
*/

IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT ML_Core.ModelOps2;
IMPORT PBblas.Types AS pTypes;
IMPORT Cluster_Types.KMeans_Model AS KTypes;
IMPORT Cluster_Types.KMeans_Model.Ind1 as KM1;


EXPORT Cluster := MODULE
  /**
    * KMeans Algorithm is a popular clustering method for cluster analysis in data mining.
    * It iteratively update the cluster centroids until it reaches the tolerance.
    * KMeans module is both highly data scalable and model scalable on HPCC Systems Platform.
    * @param max_iter    The maxinum number of iterations to run KMeans. It's an integer scalar
    *                    value. The default value is 10.
    * @param t           The converage rates. It's a real value scalar. The defaul value is 0.0.
    */
  EXPORT KMeans(INTEGER max_iter = 10 , REAL t = 0.0) := MODULE
    // Dataframe to hold the intermediate/final results of distance calculation
    SHARED ClusterPair:=RECORD
      Types.t_Work_Item   wi;
      Types.t_RecordID    id;
      Types.t_RecordID    clusterid;
      Types.t_FieldNumber number;
      Types.t_FieldReal   value01 := 0;
      Types.t_FieldReal   value02 := 0;
      Types.t_FieldReal   value03 := 0;
    END;
    // Dataframe to hold the centroids coordinates of each iteration
    SHARED lIterations:=RECORD
      Types.t_work_item wi;
      Types.NumericField.id;
      Types.NumericField.number;
      SET OF TYPEOF(Types.NumericField.value) values;
      BOOLEAN ifConverge := FALSE;
      INTEGER iter := 0;
    END;
    /**
      * Function Euclidean() is the distance matric used in the classic KMeans algorithm to
      * calcualte the distances between the samples to the centroids.
      * @param x           The samples for distance calcultion in DATASET(NumericField) format.
      *                    Each observation (e.g. record) is identified by
      *                    'id', and each feature is identified by field number (i.e.
      *                    'number').
      * @param y           The centers for distance calcultion in DATASET(NumericField) format.
      *                    Each observation (e.g. record) is identified by
      *                    'id', and each feature is identified by field number (i.e.
      *                    'number').
      * @param mode        A discrecte value 1 or 0 to define how to calculate the Euclean
      *                    distances between x and y in each model identified by wi.
      *                    Mode 1: calculate the distances of each data pair of x and y.
      *                    Mode 2: calcualte the distances of data pairs with same id in x and y.
      * @return            The distances of data pairs.
      * @see               PBblas.Types.Layout_Cell
      */
    SHARED Euclidean(DATASET(Types.NumericField) x, DATASET(Types.NumericField) y,
                                                    Types.t_Discrete mode = 0) := FUNCTION
      //Two modes: 0 --> one to many Euclidean, 1 --> one to one Euclidean.
      //Mode 1: one to many
      //Brute force method to calculate all data pair of x and y.
      //If the number of records of x and y are a and b respectively,
      //the number of calcluated distances will be a*b.
      dx := DISTRIBUTE(x, HASH32(wi, id));
      m0_axis := JOIN(dx, y, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                        TRANSFORM(ClusterPair,
                                    SELF.wi:= LEFT.wi,
                                    SELF.id := LEFT.id,
                                    SELF.clusterid := RIGHT.id,
                                    SELF.number :=LEFT.number,
                                    SELF.value01 := LEFT.value,
                                    SELF.value02 := RIGHT.value,
                                    SELF.value03 := POWER(LEFT.value - RIGHT.value, 2)),
                                    MANY LOOKUP);
      m0_g := GROUP(SORT(m0_axis, wi, id, clusterid, LOCAL),wi, id, clusterid,LOCAL);
      ClusterPair take(ClusterPair le, DATASET(ClusterPair) ri) := TRANSFORM
            SELF.value03 := SQRT(SUM(ri,value03));
            SELF.number := 0;
            SELF.value01 := 0;
            SELF.value02 := 0;
            SELF := le;
      END;
      one2many := PROJECT(ROLLUP(m0_g, GROUP, take(LEFT, ROWS(LEFT))),
                                                  TRANSFORM(pTypes.Layout_Cell,
                                                      SELF.wi_id := LEFT.wi,
                                                      SELF.x := LEFT.id,
                                                      SELF.y := LEFT.clusterid,
                                                      SELF.v := LEFT.value03));
      //Mode 2: one to one
      //Only calculate the data pair of x and y with the same id.
      //If the number of records of x and y are a and b respectively,
      //the maximum number of calcluated distances will MAX(a, b).
      m1_axis := JOIN(x, y, LEFT.wi = RIGHT.wi AND
                        LEFT.id = RIGHT.id AND
                        LEFT.number = RIGHT.number,
                        TRANSFORM(Types.NumericField,
                        SELF.value := POWER((LEFT.value - RIGHT.value), 2),
                        SELF := LEFT),
              HASH);
      m1_g := GROUP(m1_axis, wi, id, ALL);
      Types.NumericField take1(Types.NumericField le, DATASET(Types.NumericField) ri) := TRANSFORM
        SELF.value := SQRT(SUM(ri, value));
        SELF.number := ri.id;
        SELF := le;
      END;
      one2one := PROJECT(ROLLUP(m1_g, GROUP, take1(LEFT, ROWS(LEFT))),
                                                  TRANSFORM(pTypes.Layout_Cell,
                                                        SELF.wi_id := LEFT.wi,
                                                        SELF.x := LEFT.id,
                                                        SELF.y := LEFT.number,
                                                        SELF.v := LEFT.value));
      RETURN IF(MODE = 0,one2many, one2one);
    END;
    /**
      * Function getClosest() is used to calcalate the closest center of sample
      * via Euclidean distance.
      * @parm x            The samples for distance calcultion in DATASET(NumericField) format.
      *                    Each observation (e.g. record) is identified by 'id', and each
      *                    feature is identified by field number (i.e. 'number').
      * @param y           The centers for distance calcultion in DATASET(NumericField) format.
      *                    Each observation (e.g. record) is identified by 'id', and each
      *                    feature is identified by field number (i.e. 'number').
      * @return            The closest center of each sample and their distance.
      * @see               PBblas.Types.Layout_Cell
      */
    SHARED getClosest(DATASET(Types.NumericField) x, DATASET(Types.NumericField) y) := FUNCTION
      dx := DISTRIBUTE(x, HASH32(wi, id));
      m0_axis := JOIN(dx, y, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                        TRANSFORM(ClusterPair,
                                    SELF.wi:= LEFT.wi,
                                    SELF.id := LEFT.id,
                                    SELF.clusterid := RIGHT.id,
                                    SELF.number :=LEFT.number,
                                    SELF.value01 := LEFT.value,
                                    SELF.value02 := RIGHT.value,
                                    SELF.value03 := POWER(LEFT.value - RIGHT.value, 2)),
                                    MANY LOOKUP);
      m0_g := GROUP(SORT(m0_axis, wi, id, clusterid, LOCAL),wi, id, clusterid,LOCAL);
      ClusterPair take(ClusterPair le, DATASET(ClusterPair) ri) := TRANSFORM
            SELF.value03 := SQRT(SUM(ri,value03));
            SELF.number := 0;
            SELF.value01 := 0;
            SELF.value02 := 0;
            SELF := le;
      END;
      one2many := PROJECT(ROLLUP(m0_g, GROUP, take(LEFT, ROWS(LEFT))),
                                                  TRANSFORM(pTypes.Layout_Cell,
                                                      SELF.wi_id := LEFT.wi,
                                                      SELF.x := LEFT.id,
                                                      SELF.y := LEFT.clusterid,
                                                      SELF.v := LEFT.value03));
      //Calculate the closest centroid of each sample
      s :=SORT(one2many, wi_id, x, v, LOCAL);
      rst := DEDUP(s, wi_id, x, LOCAL);
      RETURN rst;
    END;

    /**
      * Fit function takes the samples d01 and initial centroids d02 to train KMeans model.
      * @param d1          The samples to be clustered in DATASET(NumericField) format.
      *                    Each observation (e.g. record) is identified by 'id', and each
      *                    feature is identified by field number (i.e. 'number').
      * @param d2          The initial K centroids for clustering in DATASET(NumericField)
      *                    format. Each observation (e.g. record) is identified by
      *                    'id', and each feature is identified by field number.
      * @return            KMeans Model in the format of ML_Core.Types.Layout_Model2.
      * @see               ML_Core.Types.Layout_Model2.
      * @see               Cluster_Types.KMeans_Model
      */
    EXPORT Fit(DATASET(Types.NumericField) d1, DATASET(Types.NumericField) d2) := FUNCTION
      //Distribute the samples d1 to each node. Same id of each wi will be on the same node.
      dp := DISTRIBUTE(d1, HASH32(wi, id));
      //Hold the initial centroids in the right dataframe: lIterations.
      d2itr := PROJECT(d2, TRANSFORM(lIterations,
                                              SELF.values := [LEFT.value],
                                              SELF := LEFT));
      //EM function: LOOP body to iteratively update the coordinates of the centroids
      EM(DATASET(literations) ds, INTEGER c) := FUNCTION
        //Distribute the samples evenly on the cluster
        dc := DISTRIBUTE(ds, HASH32(wi, id));
        //Extract the result of last iteration
        d02 := PROJECT(dc, TRANSFORM(Types.NumericField,
                                      SELF.value := LEFT.values[c],
                                      SELF := LEFT));
        //Calculate the distances of the samples to each centroid
        inner_axis := JOIN(dp, d02, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                          TRANSFORM(ClusterPair,
                                      SELF.wi:= LEFT.wi,
                                      SELF.id := LEFT.id,
                                      SELF.clusterid := RIGHT.id,
                                      SELF.number :=LEFT.number,
                                      SELF.value01 := LEFT.value,
                                      SELF.value02 := RIGHT.value,
                                      SELF.value03 := POWER(LEFT.value - RIGHT.value, 2)),
                                      MANY LOOKUP);
        inner_g := GROUP(SORT(inner_axis, wi, id, clusterid, LOCAL),wi, id, clusterid,LOCAL);
        ClusterPair take(ClusterPair le, DATASET(ClusterPair) ri) := TRANSFORM
              SELF.value03 := SQRT(SUM(ri,value03));
              SELF.number := 0;
              SELF.value01 := 0;
              SELF.value02 := 0;
              SELF := le;
        END;
        inner_onetomany := ROLLUP(inner_g, GROUP, take(LEFT, ROWS(LEFT)));
        //Calculate the closest centroid of each sample
        inner_closest :=SORT(inner_onetomany, wi, id, value03, LOCAL);
        closest := DEDUP(inner_closest, wi, id, LOCAL);
        //Calculate the number of members of each centroid/cluster
        inner_members_count:= TABLE(closest,
                                    {wi, clusterid, cnt := COUNT(GROUP)},
                                    wi, clusterid,
                                    FEW);
        //Calculate the new cordinates of each centroid
        members_local_value :=JOIN(dp,closest,
                                LEFT.wi = RIGHT.wi
                                AND
                                LEFT.id = RIGHT.id,
                                      TRANSFORM(Types.NumericField,
                                                  SELF.id := RIGHT.clusterid,
                                                  SELF := LEFT),
                          LOCAL);
        members_value :=TABLE(members_local_value,
                              {wi, id, number, s := SUM(GROUP, value)},
                              wi, id, number,
                              LOCAL);
        centroidUpdate_local := JOIN(members_value, inner_members_count,
                                      LEFT.wi = RIGHT.wi
                                      AND
                                      LEFT.id = RIGHT.clusterid,
                                      TRANSFORM(Types.NumericField,
                                      SELF.value := LEFT.s/RIGHT.cnt,
                                      SELF := LEFT), LOOKUP);
        centroidupdate := PROJECT( TABLE(centroidUpdate_local,
                                          {wi, id, number, sv := SUM(GROUP, value)},
                                          wi, id, number,
                                          FEW),
                                  TRANSFORM(Types.NumericField,
                                              SELF.value := LEFT.sv,
                                              SELF:= LEFT));
        //Filter the centroids without members
        dpass := JOIN(d02, inner_members_count,
                        LEFT.wi = RIGHT.wi AND
                        LEFT.id = RIGHT.clusterid,
                        TRANSFORM(LEFT),
                                    LEFT ONLY,
                                    HASH);
        //The new coordinates of each centroid
        result := dpass + centroidUpdate;
        //Calculate the movement of each centroid
        new_centroids := DISTRIBUTE(dpass + centroidUpdate, HASH32(wi, id));
        movement_axis := JOIN(d02, new_centroids, LEFT.wi = RIGHT.wi AND
                          LEFT.id = RIGHT.id AND
                          LEFT.number = RIGHT.number,
                          TRANSFORM(Types.NumericField,
                          SELF.value := POWER((LEFT.value - RIGHT.value), 2),
                          SELF := LEFT),
                LOCAL);
        g_movement := GROUP(movement_axis, wi, id, LOCAL);
        Types.NumericField take1(Types.NumericField le,
                                                DATASET(Types.NumericField) ri) := TRANSFORM
          SELF.value := SQRT(SUM(ri, value));
          SELF.number := 0;
          SELF := le;
        END;
        inner_movement := ROLLUP(g_movement, GROUP, take1(LEFT, ROWS(LEFT)));
        //Decide if the current iteration is converged
        WiConverges := TABLE(inner_movement,
                              {wi, BOOLEAN ifConverge := IF(MAX(GROUP, value)>t, FALSE, TRUE)},
                              wi);
        //Update the iterating dataset and feed into next iteration
        updateDS:=  JOIN(dc, new_centroids, LEFT.wi = RIGHT.wi AND
                          LEFT.id = RIGHT.id AND
                          LEFT.number = RIGHT.number,
                        TRANSFORM(literations,
                        SELF.values := LEFT.values + [RIGHT.value],
                        SELF := LEFT),
                    LOCAL);
        updated := JOIN(updateDS, WiConverges, LEFT.wi = RIGHT.wi, TRANSFORM(
                                          literations, SELF.ifConverge := RIGHT.ifConverge,
                                          SELF.iter := c,
                                          SELF := LEFT),
                      LOOKUP);
        RETURN updated;
      END;
      //LOOP function to start the EM iteration steps.
      result := LOOP(d2itr,
                      max_iter,
                      LEFT.ifConverge = FALSE,
                      EXISTS(ROWS(LEFT)(ifConverge = FALSE)),
                      em(ROWS(LEFT), COUNTER));
      //Below are the results of KMeans
      //#iterations
      tol_iters :=TABLE(result, {wi, iter}, wi, iter, MERGE);
      //cluster_centers
      centroids := PROJECT(result, TRANSFORM(Types.NumericField,
                                              SELF.value := LEFT.values[LEFT.iter + 1],
                                              SELF := LEFT));
      //labels
      movement := Euclidean(dp, centroids);
      g :=SORT(GROUP(movement,wi_id, x,ALL),wi_id, x, v);
      closest := DEDUP(g, wi_id, x);
      Labels := PROJECT(DEDUP(g, wi_id, x),TRANSFORM(RECORDOF(g)-v, SELF := LEFT));
      //inertia
      members_count:= TABLE(closest,
                              {wi_id, y, cnt := COUNT(GROUP), s := SUM(GROUP, POWER(v,2))},
                              wi_id, y, FEW);
      inertia := SUM(members_count, s);
      cluster_members := PROJECT(members_count,
                                  TRANSFORM({INTEGER wi, INTEGER id, INTEGER members_count},
                                              SELF.wi := LEFT.wi_id,
                                              SELF.id := LEFT.y,
                                              SELF.members_count := LEFT.cnt));
      /**
        * Get KMeans model
        *
        * The model includes below information:
        * 1. The coordinates of each center.
        * 2. The samples of each Cluster/center.
        * 3. The iteration runs of each wi.
        *
        * KMeans uses the Layout_Model2 format.
        * See Cluster_Types.ecl for the format of the model.
        *
        */
      DATASET(Types.Layout_Model2) getModel := FUNCTION
        centers := ModelOps2.FromNumericField(Centroids, [KM1.Centers]);
        samples := PROJECT(Labels, TRANSFORM(Types.Layout_Model2,
                    SELF.wi := LEFT.wi_id,
                    SELF.value := LEFT.y,
                    SELF.indexes := [KM1.samples, LEFT.x],
                    ));
        iterations := PROJECT(tol_iters, TRANSFORM(Types.Layout_Model2,
                        SELF.wi := LEFT.wi,
                        SELF.value := LEFT.iter,
                        SELF.indexes := [KM1.iterations]));
        mod := centers + samples + iterations;
      RETURN mod;
      END;
    RETURN getmodel;
    END;
    /**
      * Function Centers() return the final coordinates of the centers from the trained model.
      * @param mod        The fitted/trained KMeans model.
      * @return centers   The Final coordinates of the centers in NumericField format.
      */
    EXPORT Centers(DATASET(Types.Layout_Model2) mod) := FUNCTION
      index_centers := KM1.centers;
      centers := ModelOps2.ToNumericField(mod, [index_centers]);
      RETURN centers;
    END;
    /**
      * Function Predict() computes the cluster center for each new sample.
      * @param mod          The fitted/trained KMeans model.
      * @param newSamples   The new samples to be clustered.
      * @return             The index of the closest center for each new sample.
      * @see                Cluster_Types.KMeans_Model.Labels
      */
    EXPORT DATASET(KTypes.Labels) Predict(DATASET(Types.Layout_Model2) mod,
                                              DATASET(Types.NumericField) newSamples) := FUNCTION
      //Get the coordinates of the centers from the trained model
      final_centers := Centers(mod);
      //Calculate the distances between each new sample to each center
      new_closest := getClosest(newSamples, final_centers);
      //The closest center index for each sample
      Labels := SORT(PROJECT(new_closest, TRANSFORM(KTypes.Labels,
                                                      SELF.wi := LEFT.wi_id,
                                                      SELF.id := LEFT.x,
                                                      SELF.label := LEFT.y)),
                    wi, id);
      RETURN Labels;
    END;
    /**
      * Function Labels() computes the closest center of each training sample
      * from the trained Model.
      * @param mod    The fitted/trained KMeans model.
      * @return       The closest center index for each training sample.
      * @see          Cluster_Types.KMeans_Model.Labels
      */
    EXPORT DATASET(KTypes.Labels) Labels(DATASET(Types.Layout_Model2) mod) := FUNCTION
      l := ModelOps2.Extract(mod, [KM1.samples]);
      Labels := PROJECT(l, TRANSFORM(KTypes.Labels,
                                SELF.wi := LEFT.wi,
                                SELF.id := LEFT.indexes[1],
                                SELF.label := LEFT.value));
      RETURN Labels;
    END;
    /**
      * Function Iterations() computes the iteration runs of each wi.
      * @param mod              The fitted/trained KMeans model.
      * @return iterations      The total number of iterations for each wi.
      * @see                    Cluster_Types.KMeans_Model.n_Iters
      */
    EXPORT DATASET(KTypes.n_Iters) Iterations(DATASET(Types.Layout_Model2) mod) := FUNCTION
      l := ModelOps2.Extract(mod, [KM1.iterations]);
      iterations := PROJECT(l, TRANSFORM(KTypes.n_Iters,
                                SELF.wi := LEFT.wi,
                                SELF.iters := LEFT.value,
                                ));
      RETURN iterations;
    END;
  END;
END;