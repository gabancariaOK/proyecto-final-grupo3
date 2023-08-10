# Databricks notebook source
# MAGIC %md
# MAGIC ## Preparación, Importación, funciones utilitarias

# COMMAND ----------

from pyspark.sql.session import *

spark = SparkSession.builder.appName("Proyecto Final").config("spark.driver.memory", "1g").config("spark.dynamicAllocation.maxExecutors", "20").config("spark.executor.cores", "2").config("spark.executor.memory", "5g").config("spark.executor.memoryOverhead", "500m").config("spark.default.parallelism", "100").getOrCreate()


# COMMAND ----------

#Librerias
from pyspark.sql.functions import *
from pyspark.sql.window import *


# COMMAND ----------

#obtenemos el número de cores para repartición.

numCores = spark.sparkContext.defaultParallelism

# COMMAND ----------

#Lectura de los datos

df1 = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("dbfs:/FileStore/proyectoFinal/world_happiness_report_2021.csv").repartition(numCores)
df2 = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("dbfs:/FileStore/proyectoFinal/world_happiness_report.csv").repartition(numCores)

# COMMAND ----------

#función para cacher y librar de cache

def cache(df : DataFrame): 
  print("Almacenando en cache...")
  df.cache()
  print(", almacenado en cache!")


def liberarTodoElCache(spark : SparkSession):
  print("Liberando todo el cache...")
  spark.catalog.clearCache()
  print(", todo el cache liberado!")


# COMMAND ----------

# usamos un parametro para cambiar el show.

PARAM_SHOW_HABILITADO = True

#Definimos la función show 
def show(df : DataFrame):
  if(PARAM_SHOW_HABILITADO == True):
    df.show()
  


# COMMAND ----------

#funcion para guardar ficheros
def saveFile(df, ruta):
  carpeta = "dbfs:///FileStore/proyectoFinal/" + ruta  
  #Guardamos el dataframe en la carpeta
  print("Guardando fichero...")
  df.write.mode("overwrite").format("parquet").save(carpeta)
  print("Fichero guardado!")
  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploracion de los datos

# COMMAND ----------

display(df1.describe())


# COMMAND ----------

display(df2.describe())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Renombre de columnas no consistentes, y consolidar los DFs.

# COMMAND ----------

#quitamos los espacios en los nombres de columnas, y hacemos en lower case.

df1renamed = df1.toDF(*[c.lower().replace(" ","_") for c in df1.columns])
df2renamed = df2.toDF(*[c.lower().replace(" ","_") for c in df2.columns])

# COMMAND ----------

#Las columnas de un fichero no son consistentes con el otro, y hacemos un par de renames, añadimos 

df2clean = df2renamed.withColumnRenamed("life_ladder","ladder_score").withColumnRenamed("log_gdp_per_capita","logged_gdp_per_capita").withColumnRenamed("healthy_life_expectancy_at_birth","healthy_life_expectancy").select("country_name","year","ladder_score","logged_gdp_per_capita","healthy_life_expectancy")

# COMMAND ----------

#hacemos la reunion de df1clean con df2clean para uso en el futuro

df1clean = df1renamed.withColumn("year",lit("2021")).select("country_name","year","ladder_score","logged_gdp_per_capita","healthy_life_expectancy")

dfUnion = df1clean.union(df2clean)

# COMMAND ----------

#Usar cache para usar los dataframes 
cache(df1clean) 
cache(df2clean)
cache(dfUnion)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ¿Cuál es el país más “feliz” del 2021 según la data? (considerar la columna “Ladder score”)

# COMMAND ----------


happiestCountry2021 = df1clean.select("country_name","ladder_score").orderBy(desc("ladder_score")).limit(1)

# COMMAND ----------


print( "El país más “feliz” del 2021 según la data es : " + str(happiestCountry2021.select("country_name").first()[0]))

# COMMAND ----------

#guardamos el resultado en un fichero parquet:
ruta1 = "Ejercicio1"
saveFile(happiestCountry2021,ruta1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##  2. ¿Cuál es el país más “feliz” del 2021 por continente según la data?

# COMMAND ----------

#aqui usamos df1renamed en vez de df1clean por tener la columna regional_indicator

df1_region = df1renamed.select("regional_indicator","country_name","ladder_score")

# COMMAND ----------

# MAGIC %md 
# MAGIC El continente no viene dado por el dato, entonces intentamos ver qué paises hay en cada continente

# COMMAND ----------


df1_region.filter(col("regional_indicator")=="Middle East and North Africa").show

# COMMAND ----------


df1_region.select("regional_indicator").distinct().show()

# COMMAND ----------


def continentCase(country, region):
    listANZ = ["Australia", "New Zealand"]
    listAfrica = ["Lybia", "Morocco", "Algeria", "Tunisia", "Palestinian Territories", "Egypt"]

    if country in listANZ:
        return "Oceania"
    elif region in "Middle East" and country not in listAfrica:
        return "Asia"
    elif "Africa" in region:
        return "Africa"
    elif "America" in region:
        return "America"
    elif "Asia" in region:
        return "Asia"
    elif "Europe" in region:
        return "Europe"
    elif "Commonwealth"in region:
        return "Unknown"
    else:
        return "Not classified"


udfDetContinent = udf(lambda region, country: continentCase(region, country))

# COMMAND ----------

df1_continent = df1_region.withColumn("continent", udfDetContinent(col("country_name"),col("regional_indicator")))
show(df1_continent)

# COMMAND ----------

#Usar función window rank para hacer withColumn

windowSpec = Window.partitionBy("continent").orderBy(desc("ladder_score"))
happiestCountryByContinent =  df1_continent.select("continent","country_name","ladder_score","regional_indicator").withColumn("happy_by_continent_rank", rank().over(windowSpec)).filter(col("happy_by_continent_rank") == 1)



# COMMAND ----------

#Muestra el dataframe del resultado de pregunta 2
show(happiestCountryByContinent)

# COMMAND ----------

#Escribimos los dataframes resultados en una archivo

ruta2 = "Ejercicio2"
saveFile(happiestCountryByContinent,ruta2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. ¿Cuál es el país que más veces ocupó el primer lugar en todos los años?

# COMMAND ----------

#tratamos con el dataframe union

partitionByYear = Window.partitionBy("year").orderBy(desc("ladder_score"))
winnerCountry = dfUnion.select("year","ladder_score","country_name").withColumn("year_rank", rank().over(partitionByYear)).filter(col("year_rank")==1).groupBy("country_name").count().orderBy(desc("count")).limit(1)



# COMMAND ----------


show(winnerCountry)

# COMMAND ----------

#guardamos el resultado en un parquet

ruta3 = "Ejercicio3"
saveFile(winnerCountry,ruta3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.¿Qué puesto de Felicidad tiene el país con mayor GDP del 2020?

# COMMAND ----------

#obtenemos el país con mayor GDP del año 2020
topGDPCountryHappyRank = df2clean.filter(col("year") == 2020).withColumn("happiness_Ranking", rank().over(partitionByYear)).orderBy(col("logged_gdp_per_capita").desc()).limit(1)

# COMMAND ----------


show(topGDPCountryHappyRank)

# COMMAND ----------


ruta4 = "Ejercicio4"
saveFile(topGDPCountryHappyRank,ruta4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.¿En qué porcentaje ha variado a nivel mundial el GDP promedio del 2020 respecto al 2021? ¿Aumentó o disminuyó?

# COMMAND ----------


GDP2020 = float(df2clean.filter(col("year") == 2020).agg(avg("logged_gdp_per_capita")).first()[0])
GDP2021 =  float(df1clean.agg(avg("logged_gdp_per_capita")).first()[0])

percentage = (GDP2021-GDP2020)/GDP2020*100

# COMMAND ----------


print("El porcentaje variado a nivel mundial de gdp ha sido: " + str(percentage))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6.¿Cuál es el país con mayor expectativa de vida (“Healthy life expectancy at birth”)? Y ¿Cuánto tenía en ese indicador en el 2019?

# COMMAND ----------

#Preparando el dataframe

dfwithHealth = dfUnion.select("country_name","year","healthy_life_expectancy")
show(dfwithHealth)

# COMMAND ----------

#Cogemos el valor medio de los últimos 5 años de GDP

dfTopLifeExpectancyAvg = dfwithHealth.filter(col("year") > 2016).groupBy("country_name").agg(avg(col("healthy_life_expectancy")),count("country_name").alias("num_registers")).orderBy(col("avg(healthy_life_expectancy)").desc()).limit(1)

show(dfTopLifeExpectancyAvg)

# COMMAND ----------

#guardamos el resultado en fichero parquet

ruta6_1 = "Ejercicio6_1"
saveFile(dfTopLifeExpectancyAvg,ruta6_1)

# COMMAND ----------


expectancy2019 = dfwithHealth.filter((col("year") == 2019) & (col("country_name") == str(dfTopLifeExpectancyAvg.select("country_name").first()[0])))
show(expectancy2019)

# COMMAND ----------

#guardamos el resultado
ruta6_2 = "Ejercicio6_2"
saveFile(expectancy2019,ruta6_2)

# COMMAND ----------

#Liberar cache
liberarTodoElCache(spark)

# COMMAND ----------


