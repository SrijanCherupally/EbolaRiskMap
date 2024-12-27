# EbolaRiskMap
This project is based on creating a risk map for Ebola. Ebola is an extremely dangerous virus and knowing where it resides prominently is extremely important. This data is gathered from the locations of bats infected with Ebola, since there is a direct correlation between bats and Ebola outbreak zones. 

# How this data was sourced
The Ebola infected bat data was sourced from https://database.ebo-sursy.eu/public-interface/. 

# How to use this project
This project is intended to be a research project serving to warn civilians, and medical care about potential Ebola outbreak zones. This project is coded in python. Selecting the month from the dropdown menu will launch a website locally hosted which will display the Ebola Risk Map. 

# How this code works
This code requires the pandas, numpy, random, torch, sklearn, and plotly libraries to run. Pandas is used to read and load the csv file. Pytorch and Sklearn is required to run the data clustering and unsupervised learning algorithms to properly model the Ebola Risk data. Plotly is used to plot the data and run it on a web browser. 
