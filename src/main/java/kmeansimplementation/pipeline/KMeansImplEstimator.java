package kmeansimplementation.pipeline;


import kmeansimplementation.DataModel;
import kmeansimplementation.DistanceName;
import kmeansimplementation.KMeansImpl;
import kmeansimplementation.Util;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;

/**
 * Created by as on 16.04.2018.
 */
public class KMeansImplEstimator extends Estimator<KMeansImplModel> {

    private static final long serialVersionUID = 5345470610951989479L;
    private String featuresCol = "features";
    private String predictionCol = "prediction";
    private ArrayList<Vector> initialCenters;
    private long seed;
    private double epsilon;
    private int maxIterations;
    private int k;
    private DistanceName distanceName;

    public KMeansImplEstimator() {
        this.initialCenters = new ArrayList<>();
        this.seed = 20L;
        this.epsilon =1e-4;
        this.maxIterations = 20;
        this.k = 2;
        this.distanceName = DistanceName.EUCLIDEAN;
    }

    public String getFeaturesCol() {
        return featuresCol;
    }

    public KMeansImplEstimator setFeaturesCol(String featuresCol) {
        this.featuresCol = featuresCol;
        return this;
    }

    public String getPredictionCol() {
        return predictionCol;
    }

    public KMeansImplEstimator setPredictionCol(String predictionCol) {
        this.predictionCol = predictionCol;
        return this;
    }

    public ArrayList<Vector> getInitialCenters() {
        return initialCenters;
    }

    public KMeansImplEstimator setInitialCenters(ArrayList<Vector> initialCenters) {
        this.initialCenters = initialCenters;
        return this;
    }

    public long getSeed() {
        return seed;
    }

    public KMeansImplEstimator setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public KMeansImplEstimator setEpsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public KMeansImplEstimator setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    public int getK() {
        return k;
    }

    public KMeansImplEstimator setK(int k) {
        this.k = k;
        return this;
    }

    public DistanceName getDistanceName() { return distanceName; }

    public KMeansImplEstimator setDistanceName(DistanceName distanceName) {
        this.distanceName = distanceName;
        return this;
    }

    @Override
    public KMeansImplModel fit(Dataset<?> dataset) {
        //this.transformSchema(dataset.schema());
        JavaRDD<DataModel> x3 = Util.DatasetToRDD(dataset.select(this.featuresCol));
        if(this.initialCenters.isEmpty()){
            this.initialCenters = KMeansImpl.initializeCenters(x3,this.k, this.seed);
        }
        ArrayList<Vector> finalCenters = KMeansImpl.computeCenters(x3, this.initialCenters, this.epsilon, this.maxIterations, this.distanceName);
        KMeansImplModel KMeansImplModel = new KMeansImplModel()
                .setDistanceName(this.distanceName)
                .setClusterCenters(finalCenters)
                .setPredictionCol(this.predictionCol)
                .setFeaturesCol(this.featuresCol);

        return KMeansImplModel;
    }

    @Override
    public StructType transformSchema(StructType structType) {
        return structType;
    }

    @Override
    public Estimator<KMeansImplModel> copy(ParamMap paramMap) {
        return defaultCopy(paramMap);
    }

    @Override
    public String uid() {
        return "CustomTransformer" + serialVersionUID;
    }
}
