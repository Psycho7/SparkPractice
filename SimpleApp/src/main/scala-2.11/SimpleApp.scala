/**
  * Created by Psycho7 on 8/17/16.
  */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "/usr/local/Cellar/apache-spark/2.0.0/README.md"
    val conf = new SparkConf().setAppName("My Simple Application")
    val sc = new SparkContext(conf)
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(_.contains("a")).count
    val numBs = logData.filter(_.contains("b")).count
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
