package kmeansimplementation.pipeline;


import kmeansimplementation.DataModel;
import kmeansimplementation.DistanceName;
import kmeansimplementation.KMeansImpl;
import kmeansimplementation.Util;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;

/**
 * Created by as on 16.04.2018.
 */
public class KMeansImplModel extends Model<KMeansImplModel> {

    private static final long serialVersionUID = 5542470640921989462L;
    private ArrayList<Vector> clusterCenters;
    private String featuresCol = "features";
    private String predictionCol = "prediction";
    private DistanceName distanceName = DistanceName.EUCLIDEAN;

    public KMeansImplModel setClusterCenters(ArrayList<Vector> clusterCenters) {
        this.clusterCenters = clusterCenters;
        return this;
    }

    public KMeansImplModel setFeaturesCol(String featuresCol) {
        this.featuresCol = featuresCol;
        return this;
    }

    public KMeansImplModel setPredictionCol(String predictionCol) {
        this.predictionCol = predictionCol;
        return this;
    }

    public ArrayList<Vector> getClusterCenters() {
        return clusterCenters;
    }

    public String getFeaturesCol() {
        return featuresCol;
    }

    public String getPredictionCol() {
        return predictionCol;
    }

    public DistanceName getDistanceName() {
        return distanceName;
    }

    public KMeansImplModel setDistanceName(DistanceName distanceName) {
        this.distanceName = distanceName;
        return this;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {

        JavaRDD<DataModel> x3 = Util.DatasetToRDD(dataset.select(this.featuresCol));
        JavaPairRDD<Integer, Vector> x5 = KMeansImpl.predictCluster2(x3, this.clusterCenters, this.distanceName);
        Dataset<Row> dm = Util.RDDToDataset(x5, SparkSession.getActiveSession().get(), this.featuresCol, this.predictionCol);
        return dm;
    }

    @Override
    public StructType transformSchema(StructType structType) {
        return structType;
    }

    @Override
    public KMeansImplModel copy(ParamMap paramMap) {
        return defaultCopy(paramMap);
    }

    @Override
    public String uid() {
        return "CustomTransformer" + serialVersionUID;
    }
}
