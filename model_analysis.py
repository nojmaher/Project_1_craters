#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[6]:


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


# In[7]:


#simple to complex crater range is from 4.5 to 15 km which likely accounts for double peak at around 16km
#(1) craters with highly disrupted rims
#(e.g., rims superposed by at least one other crater, such as
#doublet craters), (2) craters partially out of our image range,
#and (3) ambiguous crater morphologies.
#what is difference between sample and parameter space in our case
#every measured crater
diameter is represented by a Gaussian distribution with a mean
value of D and a standard deviation given as 0.25D.


# In[29]:


#Prior
x = np.linspace(0,25,1000)
a = 7.5 #prior mean, middle of our data range, not robust. 
#I chose 7.5 because based on prior knowledge, the largest known simple craters studied are usually around 18 while the smallest are 3, so 7.5 falls right in the mean.
b = 4.5 #prior variance, not robust. Just a variance from 7.5 to 13 or 7.5 to 3
theta = stats.norm.pdf(x,a,b)
prior = plt.plot(x, theta, 'black', lw=2, label='Normal')
plt.show(prior)


# In[30]:


#Likelihood
xbar = stats_dict["Mean"]
v = 0.25*xbar
print("Variance: "+str(v))
n = 248
print("Number of Observations: "+str(n))


# In[31]:


#Posterior
mu_p = 1./(n/v+1/b)*(n*xbar/v)
var_p = 1./(n/v+1/b)

print('Posterior Mean = ',mu_p)
print('Posterior Variance =',var_p)


# In[ ]:




