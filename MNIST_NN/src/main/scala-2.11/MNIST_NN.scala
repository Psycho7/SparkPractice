import java.io.{File, FileInputStream}
import java.nio.ByteBuffer
import java.util.Date
import java.text.{DateFormat, SimpleDateFormat}
import java.util.Locale

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint

/**
  * Created by Psycho7 on 8/18/16.
  */
object MNIST_NN {
  def main(args: Array[String]) {
    println("Load Train DataSet")
    val trainImageData = loadImageData("train-images-idx3-ubyte")
    val trainLabelData = loadLabelData("train-labels-idx1-ubyte")
    val trainDataSet = combineData(trainImageData, trainLabelData)
    println()
    println("Load Test DataSet")
    val testImageData = loadImageData("t10k-images-idx3-ubyte")
    val testLabelData = loadLabelData("t10k-labels-idx1-ubyte")
    val testDataSet = combineData(testImageData, testLabelData)

    val spark = SparkSession.builder
      .appName("MNIST Neural Network")
      .getOrCreate

    import spark.implicits._

    val train = spark.sparkContext.parallelize(trainDataSet).toDS()
    val test = spark.sparkContext.parallelize(testDataSet).toDS()

    // Input layer: 28 x 28 = 784
    val layers = Array[Int](784, 500, 150, 10)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setSeed(1234L)

    val model = trainer.fit(train)
    val result = model.transform(test)
    model.save("MNIST_NN_MODEL_%s".format(new SimpleDateFormat("yyyy-MM-dd-HH-ss").format(new Date())))
    result.show()
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
  }

  def loadImageData(filePath: String): Array[Array[Int]] = {
    val file = new File(filePath)
    val in = new FileInputStream(file)
    val bytes = new Array[Byte](file.length.toInt)
    in.read(bytes)
    in.close()


    val numberOfImages = ByteBuffer.wrap(bytes.slice(4, 8)).getInt
    val numberOfRows = ByteBuffer.wrap(bytes.slice(8, 12)).getInt
    val numberOfCols = ByteBuffer.wrap(bytes.slice(12, 16)).getInt
    val sizeOfImage = numberOfRows * numberOfCols
    println("Magic Number: %d".format(ByteBuffer.wrap(bytes.take(4)).getInt))
    println("Number of images: %d".format(numberOfImages))
    println("Size of image: %d x %d".format(numberOfRows, numberOfCols))

    val imageDataBuffer = new ArrayBuffer[Array[Int]]
    var i = 0
    while (i < numberOfImages) {
      imageDataBuffer += bytes.slice(16 + i * sizeOfImage, 16 + (i + 1) * sizeOfImage).map(_ & 0xff)
      i += 1
    }

    imageDataBuffer.toArray
  }

  def loadLabelData(filePath: String): Array[Int] = {
    val file = new File(filePath)
    val in = new FileInputStream(file)
    val bytes = new Array[Byte](file.length.toInt)
    in.read(bytes)
    in.close()

    val numberOfImages = ByteBuffer.wrap(bytes.slice(4, 8)).getInt
    println("Magic Number: %d".format(ByteBuffer.wrap(bytes.take(4)).getInt))
    println("Number of images: %d".format(numberOfImages))

    val labelData = bytes.drop(8).map(_ & 0xff)
    labelData
  }

  def combineData(imageData: Array[Array[Int]], labelData: Array[Int]) = {
    val dataSet = labelData.zip(imageData).map(x =>
      new LabeledPoint(x._1, Vectors.dense(x._2.map(_.toDouble)))
    )
    dataSet
  }

}
