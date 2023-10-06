#!/usr/bin/env python
# coding: utf-8

#set up libraries and packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from scipy.stats import norm
import pandas as pd
import scipy.stats as stats
import random
import pylab
from scipy.stats import binom
from scipy.stats import beta
import matplotlib.pyplot as plt
import os

#read in the data file
#call data to show table
data = pd.read_csv("simple_RheaData.csv")
data = data.drop(columns = ['Sadness', 'BigSad', 'Pain', 'What even'])
#Diameter and depth are those aspects of crater. d/D is depth/diameter. CWS is Crater Wall Slope

#Making table of relevant info
#call data_table (not print) to see a nicer table
stats_dict = {"Min":np.min(data.Diameter),"1st Quart":np.quantile(data.Diameter,0.25),"Median":np.median(data.Diameter),"Mean":np.mean(data.Diameter),"3rd Quart":np.quantile(data.Diameter,0.75),"Max":np.max(data.Diameter)}
data_table = pd.DataFrame(data=stats_dict, index=[0])
print(data_table)

#boxplot of diameter
plt.boxplot(data.Diameter)
plt.title("Boxplot of Diameter")
plt.xlabel("Diameter")
plt.ylabel("Km")
plt.show()

#Histogram with kde (blue) and normal distribution (red)
ax = sns.histplot(data.Diameter,bins=30,binrange=[4,18.3],kde=True,stat="density")
x_pdf = np.linspace(4, 18.3, 100)
y_pdf = stats.norm.pdf(x_pdf,stats_dict["Mean"],np.sqrt(np.var(data.Diameter)))
ax.plot(x_pdf, y_pdf, 'r', lw=2, label='Normal')
ax.legend()
plt.show(ax)

#QQ plot
stats.probplot(data.Diameter,dist="norm",plot=pylab)
pylab.title("QQ Plot for Diameter")
pylab.show()
