package kmeansimplementation;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;

/**
 * Created by as on 26.04.2018.
 */
public class Util {

    /**
     * Konwersja z Dataset na JavaRDD.
     *
     * @param ds dane RDD
     * @return
     */
    public static JavaRDD<DataModel> DatasetToRDD(Dataset<Row> ds) {
        JavaRDD<DataModel> x3 = ds.toJavaRDD().map(row -> {
            KMeansImplModel KMeansImplModel = new KMeansImplModel();
            KMeansImplModel.setData((Vector) row.get(0));
            return KMeansImplModel;
        });
        return x3;
    }

    /**
     * Konwersja z JavaRDD na Dataset.
     *
     * @param x dane RDD
     * @param spark obiekt SparkSession
     * @param featuresCol nazwa kolumny atrybutow
     * @param predictionCol nazwa kolumny predykcji
     * @return dane Dataset
     */
    public static Dataset<Row> RDDToDataset(JavaPairRDD<Integer, Vector> x, SparkSession spark, String featuresCol, String predictionCol) {
        JavaRDD<Row> ss = x.map(v1 -> RowFactory.create(v1._2(), v1._1()));
        StructType schema = new StructType(new StructField[]{
                new StructField(featuresCol, new VectorUDT(), false, Metadata.empty()),
                new StructField(predictionCol, DataTypes.IntegerType, true, Metadata.empty())
        });
        Dataset<Row> dm = spark.createDataFrame(ss, schema);
        return dm;
    }

    /**
     * Zapis Dataset do pliku CSV.
     *
     * @param dm dane Dataset
     * @param featuresCol nazwa kolumny z atrybutami
     * @param predictionCol nazwa kolumny z predykcjami
     * @param path sciezka zapisu pliku
     */
    public static void saveAsCSV(Dataset<Row> dm, String featuresCol, String predictionCol, String path) {
        JavaRDD<Row> rr = dm.select(featuresCol, predictionCol).toJavaRDD().map(value -> {
            Vector vector = (Vector) value.get(0);
            Integer s = (Integer) value.get(1);
            Vector vector2 = new DenseVector(ArrayUtils.addAll(vector.toArray(), new double[]{s.doubleValue()}));
            return RowFactory.create(Arrays.toString(vector2.toArray())
                    .replace("[", "")
                    .replace("]", "")
                    .replaceAll(" ", ""));
        });
        StructType structType = new StructType().add("data", DataTypes.StringType);

        // Save Dataset (overwrite allowed).
        dm.sqlContext().createDataFrame(rr, structType)
                .coalesce(1) // file in 1 part
                .write()
                .mode(SaveMode.Overwrite)
                .text(path);
    }

}
