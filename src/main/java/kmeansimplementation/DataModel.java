package kmeansimplementation;

import org.apache.spark.ml.linalg.Vector;

/**
 * Created by as on 02.06.2018.
 */
public interface DataModel {

    Vector getData();

    void setData(Vector data);

}
