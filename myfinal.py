#!/usr/bin/env python
# coding: utf-8

# **Final Project: Spotify Top 2000's Statistics**
# 
# As shown below, the section titled "Final Demo: Main Menu" reflects my finished search functionalities, analytics, and exploration for grading.

# # **Import statements and data partitions**

# In[1]:


# import sys
# sys.path.append("/usr/local/lib/python3.11/dist-packages")


# In[2]:


get_ipython().system('pip install pandas')
import os
import urllib
import pandas as pd
import numpy as np
import sys
from scipy import stats, special
import seaborn as sns
from seaborn import pairplot, heatmap
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import *
import ipywidgets as widgets
import sqlalchemy as sa
import sqlite3 as sl
import seaborn as sns
from tabulate import tabulate 
from itertools import groupby
from sqlalchemy import create_engine, event
from sqlalchemy.engine.url import URL
from ipywidgets import GridspecLayout, Button, Layout, jslink, IntText, IntSlider
from ipywidgets import *
from IPython.display import display
from ipywidgets import interact, interact_manual

from sklearn import model_selection, metrics, linear_model, tree, datasets, feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# **Create partition tables and establish connection**
# 
# 

# In[ ]:


csvfile = open('/content/spotify_stats.csv', 'r').readlines()
file_name = 1
for i in range(len(csvfile)):
  if i % 1001 == 0:
    open("spotify_stats_" + str(file_name) + '.csv', 'w+').writelines(csvfile[i:i+1001])
    file_name += 1


# In[ ]:


df = pd.read_csv('./spotify_stats_1.csv')
df2 = pd.read_csv('./spotify_stats_2.csv')


# In[ ]:


conn = sl.connect('spotify.db')
print("Sucessfully connected to spotify_stats database")


# # **Read in partition tables and clean data**

# In[ ]:


print("Null values checked: ", df.isnull())
print("Duplicate values checked: ", df.duplicated())
print("Spotify dataframe shape: ", df.shape)
df1 = df.dropna()
df1 = df2.drop_duplicates()
print("Spotify dataframe shape after dropping values: ", df.shape)
print('\n')
print("Null values partition 2:", df2.isnull())
print("Duplicate values checked: ", df2.duplicated())
print("Spotify dataframe shape: ", df2.shape)
df2 = df2.dropna()
df2 = df2.drop_duplicates()
print("Shape after dropping values: ", df2.shape)


# In[ ]:


# Create first partition in SQL database
cursor = conn.cursor()
print("Checking if table exists already")
cursor.execute("DROP TABLE IF EXISTS Spotify_stats1;")
print("Creating table Spotify_stats1")
stmt1 = "CREATE TABLE Spotify_stats1 (\
title VARCHAR (41) NOT NULL\
,artist VARCHAR (41) NOT NULL\
,genre VARCHAR (41) NOT NULL\
,year INTEGER NOT NULL\
,bpm INTEGER NOT NULL\
,duration INTEGER NOT NULL\
,energy NUMERIC(6,4) NOT NULL\
,danceability NUMERIC(5,3) NOT NULL\
,loudness NUMERIC(7,3) NOT NULL\
,liveness NUMERIC(6,4) NOT NULL\
,valence NUMERIC(6,4) NOT NULL\
,acousticness NUMERIC(8,6) NOT NULL\
,speechiness NUMERIC(6,4) NOT NULL\
,popularity NUMERIC(7,3) NOT NULL)"
cursor.execute(stmt1)
print("Inserting values into table Spotify_stats1")
for row in df.itertuples():
  stmt2 = "INSERT INTO Spotify_stats1 (title,artist,genre,year,bpm,duration,energy,danceability,loudness,liveness,valence,acousticness,speechiness,popularity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
  val = (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14])
  cursor.execute(stmt2, val)
  conn.commit()
print("Table Spotify_stats1 created successfully")


# In[ ]:


pd.read_sql_query("select * from Spotify_stats1", con = conn)


# In[ ]:


# Create second partition in SQL database
cursor = conn.cursor()
print("Checking if table exists already")
cursor.execute("DROP TABLE IF EXISTS Spotify_stats2;")
print("Creating table Spotify_stats2")
stmt1 = "CREATE TABLE Spotify_stats2 (\
title VARCHAR (41) NOT NULL\
,artist VARCHAR (41) NOT NULL\
,genre VARCHAR (41) NOT NULL\
,year INTEGER NOT NULL\
,bpm INTEGER NOT NULL\
,duration INTEGER NOT NULL\
,energy NUMERIC(6,4) NOT NULL\
,danceability NUMERIC(5,3) NOT NULL\
,loudness NUMERIC(7,3) NOT NULL\
,liveness NUMERIC(6,4) NOT NULL\
,valence NUMERIC(6,4) NOT NULL\
,acousticness NUMERIC(8,6) NOT NULL\
,speechiness NUMERIC(6,4) NOT NULL\
,popularity NUMERIC(7,3) NOT NULL)"
cursor.execute(stmt1)
print("Inserting values into table Spotify_stats2")
for row in df2.itertuples():
  stmt2 = "INSERT INTO Spotify_stats2 (title,artist,genre,year,bpm,duration,energy,danceability,loudness,liveness,valence,acousticness,speechiness,popularity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
  val = (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14])
  cursor.execute(stmt2, val)
  conn.commit()
print("Table Spotify_stats2 created successfully")


# In[ ]:


pd.read_sql_query("select * from Spotify_stats2", con = conn)


# # **Mappartition and reduce functions**

# In[ ]:


def display():
  stmt = "SELECT * FROM Spotify_stats1;"
  cursor.execute(stmt)
  result = cursor.fetchall()
  print("Partition 1:", result)

  stmt2 = "SELECT * FROM Spotify_stats2;"
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  print("Partition 2:", result2)


# In[ ]:


#Explanation facility
def showExplain(stmt, stmt2, result, result2, final_result2, bool, bool1):
  print("Summary of partitions:")
  display()
  print()
  print("SQL statements for mapPartition():")
  print("Partition 1: " + stmt)
  print("Partition 2: " + stmt2)
  print()
  print("Result from Partition 1:", result)
  print("Result from Partition 1:", result2)
  print()
  if bool:
    print("Reduce function:")
    if bool1:
      print("Combining lists")
      print("Using groupby")
      print("For each key, sum up their values and divide by 2 to calculate average: sum(v[1] for v in g)/2")
      print("Now producing the final result")
    else:
      print("Combining two list by list add...")
      print("Using groupby from itertools, for each key: groupby(sorted(final_result), key = lambda x: x[0])")
      print("For each key, sum up their values to calculate sum: sum(v[1] for v in g)")
      print("Now producing the final result")
  else:
    print("Combining two list by list add...")
  print("Final result after reduce function:", final_result2)
  print()


# **Partition-based map and reduce functions for the average and sum**

# In[ ]:


def averageReduce(result, result2):
  final_result = result + result2
  final_result2 = []
  for i, g in groupby(sorted(final_result), key=lambda x: x[0]):
    final_result2.append([i, sum(v[1] for v in g)/2])
  return final_result2


# In[ ]:


def sumReduce(result, result2):
  #Reduce
  final_result = result + result2
  final_result2 = []
  #Sum of average with group by and list append
  for i, g in groupby(sorted(final_result), key = lambda x: x[0]):
    final_result2.append([i, sum(v[1] for v in g)])
  return final_result2


# # **Summarize and Visualize Data by Graphs**

# 
# Average of bpm/duration(ms)/energy/danceability/loudness/liveness/valence/acousticness/speechiness/popularity by year

# In[ ]:


def averageGraph(cat):
  #MapPartition from partitioned data
  stmt = "SELECT year, AVG(" + cat + ") FROM Spotify_stats1 GROUP BY year;"
  stmt2 = "SELECT year, AVG(" + cat + ") FROM Spotify_stats2 GROUP BY year;"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  #Reduce function
  final_result2 = averageReduce(result, result2)

  year = []
  average = []
  for i in final_result2:
    year.append(i[0])
  for i in final_result2:
    average.append(i[1])
  year, average = zip(*sorted(zip(year, average)))

  showExplain(stmt, stmt2, result, result2, final_result2, True, True)
  print("Display plot graph of your choice:", cat)

  fig, ax = plt.subplots(figsize=(20, 10))
  ax.set_xticks(range(1956, 2019, 1))
  ax.set_xlabel('Year')
  ax.set_ylabel(cat)
  ax.set_title("The average value of " + cat + " over the years")
  ax.plot(year, average)
  plt.grid()
  plt.show()


# In[ ]:


def runProgram():
  averageGraphAnswer()

runProgram()


# In[ ]:


#Categories
def catePop(cat):
  #mapPartition and reduce from partition
  stmt = "SELECT " + cat + ", popularity FROM Spotify_stats1 GROUP BY " + cat + ";"
  stmt2 = "SELECT " + cat + ", popularity FROM Spotify_stats2 GROUP BY " + cat + ";"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  final_result2 = result + result2

  cat_data = []
  popularity = []
  for i in final_result2:
    cat_data.append(i[0])
  for i in final_result2:
    popularity.append(i[1])
  
  showExplain(stmt, stmt2, result, result2, final_result2, False, False)
  print("Displaying the scatter graph of your choice:", cat)
  plt.figure(figsize = (20,10))
  plt.scatter(cat_data, popularity, c ="lightblue")
  plt.xlabel(cat)
  plt.ylabel("popularity")
  plt.title("Scatter graph of popularity and " + cat)
  plt.show()



# In[ ]:


def runProgram():
  comparePopGraph()

runProgram()


# Top 50 artists and their songs' sum popularity.

# In[ ]:


def artistSumPop():
  stmt = "SELECT artist, SUM(popularity) AS SUM_Popularity from Spotify_stats1 GROUP BY artist ORDER BY popularity DESC LIMIT 50;"
  stmt2 = "SELECT artist, SUM(popularity) AS SUM_Popularity from Spotify_stats2 GROUP BY artist ORDER BY popularity DESC LIMIT 50;"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  
  #Reduce func from partitions
  final_result2 = sumReduce(result, result2)

  SUM_score = []
  for i in final_result2:
    SUM_score.append(i[1])
  artist = []
  for y in final_result2:
    artist.append(y[0])
  SUM_score, artist = zip(*sorted(zip(SUM_score, artist)))

  showExplain(stmt, stmt2, result, result2, final_result2, True, False)
  print("Displaying horizontal bar graph: Sum Score of Popularity based on artist")
  plt.figure(figsize=(8,20))
  plt.barh(artist, SUM_score)
  plt.xlabel("Popularity score")
  plt.ylabel("artist")
  plt.title("Sum score of popularity based on artist")
  plt.grid()
  plt.show()


# In[ ]:


artistSumPop()


# Top 50 artists and their songs' sum popularity.

# In[ ]:


def artistAvgPop():
  stmt = "SELECT artist, AVG(popularity) AS AVG_Popularity from Spotify_stats1 GROUP BY artist ORDER BY popularity DESC LIMIT 50;"
  stmt2 = "SELECT artist, AVG(popularity) AS AVG_Popularity from Spotify_stats2 GROUP BY artist ORDER BY popularity DESC LIMIT 50;"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  
  #Reduce func from partitions
  final_result2 = averageReduce(result, result2)

  avg_score = []
  for i in final_result2:
    avg_score.append(i[1])
  artist = []
  for y in final_result2:
    artist.append(y[0])
  avg_score, artist = zip(*sorted(zip(avg_score, artist)))

  showExplain(stmt, stmt2, result, result2, final_result2, True, True)
  print("Displaying horizontal bar graph: Average Score of Popularity based on artist")
  plt.figure(figsize=(8,20))
  plt.barh(artist, avg_score)
  plt.xlabel("Popularity score")
  plt.ylabel("artist")
  plt.title("Average score of popularity based on artist")
  plt.grid()
  plt.show()


# In[ ]:


artistAvgPop()


# Average popularity of each genre

# In[ ]:


def genrePop():
  stmt = "SELECT genre, AVG(popularity) AS AVG_Popularity from Spotify_stats1 GROUP BY genre ORDER BY popularity DESC LIMIT 50;"
  stmt2 = "SELECT genre, AVG(popularity) AS AVG_Popularity from Spotify_stats2 GROUP BY genre ORDER BY popularity DESC LIMIT 50;"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  
  #Reduce func from partitions
  final_result2 = averageReduce(result, result2)

  avg_score = []
  for i in final_result2:
    avg_score.append(i[1])
  genre = []
  for y in final_result2:
    genre.append(y[0])
  avg_score, genre = zip(*sorted(zip(avg_score, genre)))

  showExplain(stmt, stmt2, result, result2, final_result2, True, True)
  print("Displaying horizontal bar graph: Average Score of Popularity based on genre")
  plt.figure(figsize=(8,16))
  plt.barh(genre, avg_score)
  plt.xlabel("Popularity score")
  plt.ylabel("genre")
  plt.title("Average score of popularity based on genre")
  plt.grid()
  plt.show()


# In[ ]:


genrePop()


# **# of songs and categories**

# In[ ]:


def popularity(cat):
  stmt = "SELECT popularity, " + cat + " FROM Spotify_stats1 GROUP BY " + cat + ";"
  stmt2 = "SELECT popularity, " + cat + " FROM Spotify_stats2 GROUP BY " + cat + ";"
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt)
  result = cursor.fetchall()
  cursor.execute(stmt2)
  result2 = cursor.fetchall()
  final_result = result + result2
  
  #Reduce func from partitions
  final_result2 = sumReduce(result, result2)

  score = []
  for i in final_result2:
    score.append(round(i[1]))
  num_song = []
  for y in final_result2:
    num_song.append(y[0])
  

  showExplain(stmt, stmt2, result, result2, final_result2, True, False)
  print("Displaying bar graph based on choice:", cat)
  score, num_song = zip(*sorted(zip(score, num_song)))
  plt.figure(figsize=(8,16))
  plt.hist([score, num_song])
  plt.xlabel(cat)
  plt.ylabel("Number of Song titles")
  plt.title(cat + " and number of Song titles")
  plt.grid()
  plt.show()


# In[ ]:


subMenuOne()
pop_input = int(input())
if pop_input == 1:
  genrePop()
elif pop_input == 2:
  print("""Please pick one of following: (Enter a number)
  1. Sum popularity
  2. Average popularity""")
  art_input = int(input())
  if art_input == 1:
    artistSumPop()
  else:
    artistAvgPop()
elif pop_input == 3:
  popularity("bpm")
elif pop_input == 4:
  popularity("duration")
elif pop_input == 5:
  popularity("energy")
elif pop_input == 6:
  popularity("danceability")
elif pop_input == 7:
  popularity("loudness")
elif pop_input == 8:
  popularity("liveness")
elif pop_input == 9:
  popularity("valence")
elif pop_input == 10:
  popularity("acousticness")
elif pop_input == 11:
  popularity("speechiness")
elif pop_input == 12:
  popularity("popularity")


# # **Final Demo: Main menu**
# 

# In[ ]:


def mainMenu():
  print("""Please choose from the following categories: (Enter a number)
  1. Explore graphs
  2. Exit""")

def menuTwo():
  print("""Let's explore different graphs based on the Spotify dataset,
  Please choose from the following categories: (Enter a number)
  1. Explore average values of different categories over the years
  2. Explore popularities and one of the other categories
  3. Explore the scatter plot graph of popularity and other categories of your choice
  4. Exit to main menu""")

def comparePopGraph():
  subMenuThree()
  plot_input = int(input())
  if plot_input == 1:
    catePop("bpm")
  elif plot_input == 2:
    catePop("duration")
  elif plot_input == 3:
    catePop("energy")
  elif plot_input == 4:
    catePop("danceability")
  elif plot_input == 5:
    catePop("loudness")
  elif plot_input == 6:
    catePop("liveness")
  elif plot_input == 7:
    catePop("valence")
  elif plot_input == 8:
    catePop("acousticness")
  elif plot_input == 9:
    catePop("speechiness")

def averageGraphAnswer():
  subMenuTwo()
  average_input = int(input())
  if average_input == 1:
    averageGraph("bpm")
  elif average_input == 2:
    averageGraph("duration")
  elif average_input == 3:
    averageGraph("energy")
  elif average_input == 4:
    averageGraph("danceability")
  elif average_input == 5:
    averageGraph("loudness")
  elif average_input == 6:
    averageGraph("liveness")
  elif average_input == 7:
    averageGraph("valence")
  elif average_input == 8:
    averageGraph("acousticness")
  elif average_input == 9:
    averageGraph("speechiness")
  elif average_input == 10:
    averageGraph("popularity")

def subMenuOne():
  print("""Please choose from the following categories: (Enter a number)
  1. genre
  2. artist
  3. bpm
  4. duration
  5. energy
  6. danceability
  7. loudness
  8. liveness
  9. valence
  10. acousticness
  11. speechiness
  12. popularity""")

def subMenuTwo():
  print("""Please choose from the following categories: (Enter a number)
  1. bpm
  2. duration
  3. energy
  4. danceability
  5. loudness
  6. liveness
  7. valence
  8. acousticness
  9. speechiness
  10. popularity""")

def subMenuThree():
  print("""Please choose from the following categories: (Enter a number)
  1. bpm
  2. duration
  3. energy
  4. danceability
  5. loudness
  6. liveness
  7. valence
  8. acousticness
  9. speechiness""")

def userChoiceMenu():
  print("""What do you want to search? Please select from the following categories: (Enter a number):
  1. artist name
  2. song title
  3. year
  4. popularity
  5. genre
  6. exit to main menu""")



def userMainChoice():
  cont = 1
  user_input = 1
  while cont == 1:
    mainMenu()
    user_input = int(input())
    if user_input == 1:
      userGraphChoice()
    elif user_input == 2:
      print("Thanks for using my final demonstration!")
      cont = 0

def userGraphChoice():
  cont = 1
  sub_input = 1
  while cont == 1:
    menuTwo()
    sub_input = int(input())
    if sub_input == 1:
      averageGraphAnswer()
    elif sub_input == 2:
      subMenuOne()
      pop_input = int(input())
      if pop_input == 1:
        genrePop()
      elif pop_input == 2:
        print("""Please pick one of the following: (Enter a number))
        1. Sum popularity
        2. Average popularity""")
        art_input = int(input())
        if art_input == 1:
          artistSumPop()
        else:
          artistAvgPop()
      elif pop_input == 3:
        popularity("bpm")
      elif pop_input == 4:
        popularity("duration")
      elif pop_input == 5:
        popularity("energy")
      elif pop_input == 6:
        popularity("danceability")
      elif pop_input == 7:
        popularity("loudness")
      elif pop_input == 8:
        popularity("liveness")
      elif pop_input == 9:
        popularity("valence")
      elif pop_input == 10:
        popularity("acousticness")
      elif pop_input == 11:
        popularity("speechiness")
    elif sub_input == 3:
      subMenuThree()
      plot_input = int(input())
      if plot_input == 1:
        catePop("bpm")
      elif plot_input == 2:
        catePop("duration")
      elif plot_input == 3:
        catePop("energy")
      elif plot_input == 4:
        catePop("danceability")
      elif plot_input == 5:
        catePop("loudness")
      elif plot_input == 6:
        catePop("liveness")
      elif plot_input == 7:
        catePop("valence")
      elif plot_input == 8:
        catePop("acousticness")
      elif plot_input == 9:
        catePop("speechiness")
    elif sub_input == 4:
      cont = 0

def main():
  userMainChoice()

main()
        

