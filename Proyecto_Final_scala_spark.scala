// Databricks notebook source
// MAGIC %md
// MAGIC ## Preparación, Importación, funciones utilitarias

// COMMAND ----------

//Reserva de recursos computacionales
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.
appName("Proyecto Final").
config("spark.driver.memory", "1g").
config("spark.dynamicAllocation.maxExecutors", "20").
config("spark.executor.cores", "2").
config("spark.executor.memory", "5g").
config("spark.executor.memoryOverhead", "500m").
config("spark.default.parallelism", "100").
getOrCreate()


// COMMAND ----------

//Librerias
import org.apache.spark.sql.expressions.Window
import spark.sqlContext.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// COMMAND ----------

// Obtenemos el número de cores para repartición.
val numCores = spark.sparkContext.defaultParallelism

// COMMAND ----------

//Lectura de los datos
val df1 = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("dbfs:/FileStore/proyectoFinal/world_happiness_report_2021.csv")
.repartition(numCores)
val df2 = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("dbfs:/FileStore/proyectoFinal/world_happiness_report.csv")
.repartition(numCores)

// COMMAND ----------

//Funciónes para cacher y librar de cache
def cache(df : DataFrame) = {
  print("Almacenando en cache...")
  df.cache()
  println(", almacenado en cache!")
}

def liberarTodoElCache(spark : SparkSession) = {
  print("Liberando todo el cache...")
  spark.sqlContext.clearCache()
  println(", todo el cache liberado!")
}

// COMMAND ----------

// Usamos un parametro para cambiar el show.
var PARAM_SHOW_HABILITADO = true

//Definimos la función show 
def show(df : DataFrame) = {
  if(PARAM_SHOW_HABILITADO == true){
    df.show(false)
  }
}

// COMMAND ----------

//Función para guardar ficheros
def saveFile(df : DataFrame,ruta:String) : Unit = {
  
  val carpeta = "dbfs:///FileStore/proyectoFinal/" + ruta
  
  //Guardamos el dataframe en la carpeta
  print("Guardando fichero...")
  df.write.mode("overwrite").format("parquet").save(carpeta)
  print("Fichero guardado!")
}

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Exploracion de los datos

// COMMAND ----------

display(df1.describe())

// COMMAND ----------

display(df2.describe())

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Renombre de columnas no consistentes, y consolidar los DFs.

// COMMAND ----------

// Quitamos los espacios en los nombres de columnas, y lo transformamos en minúscula.
val df1renamed = df1.toDF(df1.columns.map(_.trim.toLowerCase.replaceAll(" ", "_")):_*)
val df2renamed = df2.toDF(df2.columns.map(_.trim.toLowerCase.replaceAll(" ", "_")):_*)

// COMMAND ----------

//Las columnas de un fichero no son consistentes con el otro, y hacemos un par de renames, añadimos 
val df2clean = df2renamed.withColumnRenamed("life_ladder","ladder_score")
.withColumnRenamed("log_gdp_per_capita","logged_gdp_per_capita")
.withColumnRenamed("healthy_life_expectancy_at_birth","healthy_life_expectancy")
.select("country_name","year","ladder_score","logged_gdp_per_capita","healthy_life_expectancy")

// COMMAND ----------

// Hacemos la reunion de df1clean con df2clean para uso en el futuro
val df1clean = df1renamed.withColumn("year",lit("2021")).select("country_name","year","ladder_score","logged_gdp_per_capita","healthy_life_expectancy")

val dfUnion = df1clean.union(df2clean)

// COMMAND ----------

// Usar cache para usar los dataframes 
cache(df1clean) 
cache(df2clean)
cache(dfUnion)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 1. ¿Cuál es el país más “feliz” del 2021 según la data? (considerar la columna “Ladder score”)

// COMMAND ----------

val happiestCountry2021 = df1clean.select("country_name","ladder_score").orderBy(desc("ladder_score")).limit(1)

// COMMAND ----------

println( "El país más “feliz” del 2021 según la data es : " + happiestCountry2021.select("country_name").as[String].collect().head)

// COMMAND ----------

// guardamos el resultado en un fichero parquet:
val ruta1 = "Ejercicio1"
saveFile(happiestCountry2021,ruta1)

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ##  2. ¿Cuál es el país más “feliz” del 2021 por continente según la data?

// COMMAND ----------

//Aqui usamos df1renamed en vez de df1clean por tener la columna regional_indicator
val df1_region = df1renamed.select("regional_indicator","country_name","ladder_score")

// COMMAND ----------

// MAGIC %md 
// MAGIC El continente no viene dado por el dato, entonces intentamos ver qué paises hay en cada continente

// COMMAND ----------

df1_region.filter(col("regional_indicator")==="Middle East and North Africa").show

// COMMAND ----------

df1_region.select("regional_indicator").distinct.show(false)

// COMMAND ----------

    def continentCase(country: String,region: String): String = {
        val listANZ = List("Australia", "New Zealand")
        val listAfrica = List("Lybia", "Morocco", "Algeria", "Tunisia", "Palestinian Territories", "Egypt")

        country match {
            case a if listANZ.contains(country) => "Oceania"
            case a if (region.contains("Middle East") && !listAfrica.contains(country)) => "Asia"
            case b if region.contains("Africa") => "Africa"
            case b if region.contains("America") => "America"
            case b if region.contains("Asia") => "Asia"
            case b if region.contains("Europe") => "Europe"
            case c if region.contains("Commonwealth") =>"Unknown"
            case _ => "Not classified"
        }
    }
    var udfDetContinent = udf((region: String, country: String) => continentCase(region, country))

// COMMAND ----------

val df1_continent = df1_region.withColumn("continent", udfDetContinent(col("country_name"),col("regional_indicator")))
show(df1_continent)

// COMMAND ----------

//Usar función window rank para hacer withColumn
val partitionByContinent = Window.partitionBy("continent").orderBy(desc("ladder_score"))
val happiestCountryByContinent =  df1_continent.select("continent","country_name","ladder_score","regional_indicator") 
.withColumn("happy_by_continent_rank", rank().over(partitionByContinent)) 
.filter(col("happy_by_continent_rank") === 1)



// COMMAND ----------

// Muestra el dataframe del resultado de pregunta 2
show(happiestCountryByContinent)

// COMMAND ----------

// Escribimos los dataframes resultados en una archivo
val ruta2 = "Ejercicio2"
saveFile(happiestCountryByContinent,ruta2)

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ## 3. ¿Cuál es el país que más veces ocupó el primer lugar en todos los años?

// COMMAND ----------

//Tratamos con el dataframe union
val partitionByYear = Window.partitionBy("year").orderBy(desc("ladder_score"))
val winnerCountry = dfUnion
.select("year","ladder_score","country_name")
.withColumn("year_rank", rank().over(partitionByYear))
.filter(col("year_rank")===1)
.groupBy("country_name")
.count().orderBy(desc("count"))
.limit(1)

// COMMAND ----------

show(winnerCountry)

// COMMAND ----------

//Guardamos el resultado en un parquet
val ruta3 = "Ejercicio3"
saveFile(winnerCountry,ruta3)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 4.¿Qué puesto de Felicidad tiene el país con mayor GDP del 2020?

// COMMAND ----------

//Obtenemos el país con mayor GDP del año 2020
val topGDPCountryHappyRank = df2clean.filter($"year" === 2020)
.withColumn("happiness_Ranking", rank().over(partitionByYear))
.orderBy($"logged_gdp_per_capita".desc)
.limit(1)

// COMMAND ----------

show(topGDPCountryHappyRank)

// COMMAND ----------

val ruta4 = "Ejercicio4"
saveFile(topGDPCountryHappyRank,ruta4)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.¿En qué porcentaje ha variado a nivel mundial el GDP promedio del 2020 respecto al 2021? ¿Aumentó o disminuyó?

// COMMAND ----------

val GDP2020 = df2clean.filter(col("year") === 2020).agg(avg("logged_gdp_per_capita")).as[Double].first()
val GDP2021 =  df1clean.agg(avg("logged_gdp_per_capita")).as[Double].first()

val percentage = (GDP2021-GDP2020)/GDP2020*100

// COMMAND ----------

println(s"El porcentaje variado a nivel mundial de gdp ha sido: ${percentage.toFloat}%")

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ## 6.¿Cuál es el país con mayor expectativa de vida (“Healthy life expectancy at birth”)? Y ¿Cuánto tenía en ese indicador en el 2019?

// COMMAND ----------

//Preparando el dataframe
val dfwithHealth = dfUnion.select("country_name","year","healthy_life_expectancy")
show(dfwithHealth)

// COMMAND ----------

// Cogemos el valor medio de los últimos 5 años de GDP
val dfTopLifeExpectancyAvg = dfwithHealth.filter(col("year") > 2016).groupBy("country_name").agg(avg("healthy_life_expectancy"),count($"country_name") as "num_registers").orderBy(desc("avg(healthy_life_expectancy)")).limit(1)


// COMMAND ----------

show(dfTopLifeExpectancyAvg)

// COMMAND ----------

//Guardamos el resultado en fichero parquet
val ruta6_1 = "Ejercicio6_1"
saveFile(dfTopLifeExpectancyAvg,ruta6_1)

// COMMAND ----------

val expectancy2019 = dfwithHealth.filter((col("year") ===2019) && (col("country_name") === dfTopLifeExpectancyAvg.select("country_name").as[String].first()))
show(expectancy2019)

// COMMAND ----------

//Guardamos el resultado
val ruta6_2 = "Ejercicio6_2"
saveFile(expectancy2019,ruta6_2)

// COMMAND ----------

//Liberar cache
liberarTodoElCache(spark)
