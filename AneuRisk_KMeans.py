#!/usr/bin/env python
# coding: utf-8

# # K-Means
# 
# In this notebook we apply K-Means clustering technique to divide patients in homogeneous groups according to their geometric characteristics. Then, we perform descriptive analysis on clusters to understand in more details if there could be some common trend.

# In[1]:


# import libratries
import os
import pandas as pd
import scipy.interpolate
import numpy as np
from localreg import *
import matplotlib.pyplot as plt
import matplotlib.lines
from statsmodels.stats import proportion
import scipy.stats
from numpy import diff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


# In[2]:


# import the AneuRisk class
from aneurisk import Aneurisk


# In[3]:


# create an instance of Aneurisk class
aneurisk = Aneurisk()


# ## Build the dataset:

# #### Notation:
# * ICA = internal carotid artery
# * ACA = anterior cerebral artery
# * MCA = middle cerebral artery
# * BSA = basilar artery

# In[4]:


# define the path
path = 'C:/Users/erica/Desktop/Thesis/AneuriskDatabase-master/dicom'
dirs = os.listdir(path)


# In[5]:


# read the csv file containing the data
l = []
for d in dirs:
    df = pd.read_csv(path + '/' + d + '/manifest.csv', header = 0)
    l.append(df)


# In[6]:


# create the final dataset
final_data = pd.concat(l)
final_data.index = final_data['id']
final_data = final_data.drop('id', axis = 1)


# In[7]:


# see the dataset
final_data


# In[8]:


# export the dataset
final_data.to_csv('C:/Users/erica/Desktop/Thesis/AneuriskDatabase-master/dataset.csv')


# In ruptureStatus column there is a value 'F' which seems to be an error, let's remove it:

# In[9]:


# understand which is that patient to later remove it also from other dataframes
final_data[final_data.ruptureStatus == 'F'].index


# In[10]:


# reset the index to get unique value for each row
final_data = final_data.reset_index()
# drop the row assocaited to ruptureStatus = F
final_data = final_data.drop(final_data[final_data.ruptureStatus == 'F'].index, axis = 0)
# re-associate the index to ids
final_data.index = final_data['id']
# drop unuseless columns
final_data = final_data.drop(['id'], axis = 1)
final_data


# ## Preliminary checks:

# In[11]:


# check the items
final_data.count()


# In[12]:


# proportion of male and female
male = final_data[final_data.sex == 'M']['sex'].count()
female = final_data[final_data.sex == 'F']['sex'].count()
sex_df = pd.DataFrame(data = {'Male': [male, (male/final_data.shape[0])*100],
                              'Female': [female, (female/final_data.shape[0])*100]},
                            index = ['Counts', 'Proportion'], 
                            columns = ['Male', 'Female'])

# check if difference is significant
stat, pval = proportion.proportions_ztest(male, final_data.shape[0], value = 0.5)
pval


# *Taking as threshold alpha 0.05, it seens to be significant, and so the proportion can not be considered balanced.*

# In[13]:


# check age
plt.hist(final_data['age'],bins=20, color='lightsteelblue', edgecolor='slategrey')
# set axis labels
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
# set the grid
plt.grid(True, linestyle='--', alpha=0.5)
# set the background color
plt.gca().set_facecolor('whitesmoke')


# In[14]:


# shapiro wilk test
scipy.stats.shapiro(final_data['age'])


# *Age variable can be considered as normal distributed.*

# ## Build centerline df:

# In[15]:


# define the centerline path
path2 = 'C:/Users/erica/Desktop/Thesis/AneuriskDatabase-master/models'
dirs2 = os.listdir(path2)


# In[16]:


# create the dataframe
c = []
for d in dirs2:
        for i in range(20):
            try:
                # read the file with centerlines
                df = pd.read_csv(path2 + '/' + d + f'/morphology/centerlines_{i}.csv')
                # create the dataframe
                df['ID_patient'] = d
                # append all dataframes to the 'c' list
                c.append(df)
            except:
                break


# In[17]:


# contatenate all the dataframes
df_cent = pd.concat(c)


# In[18]:


# rename ID for centerlines
df_cent = df_cent.rename(columns = {'ID':'ID_centerline'})


# In[19]:


# create the index
df_cent.index = np.arange(df_cent.shape[0])


# In[20]:


# check the dataframe
df_cent


# In[21]:


# remove the C0095 patient
df_cent = df_cent.drop(df_cent[df_cent.ID_patient == 'C0095'].index, axis = 0)
df_cent


# #### Plot the X(s), Y(s) and Z(s) centerlines and their derivatives.

# In[22]:


# plot the X(s)
aneurisk.plot_splines(coordinates = 'x', df = df_cent)


# In[23]:


# plot X'(s) to avoid amplitude variability
aneurisk.plot_derivatives(coordinates = 'x', df = df_cent)


# In[24]:


# plot the Y(s)
aneurisk.plot_splines(coordinates = 'y', df = df_cent)


# In[25]:


# plot Y'(s) to avoid amplitude variability
aneurisk.plot_derivatives(coordinates = 'y', df = df_cent)


# In[26]:


# plot the Z(s)
aneurisk.plot_splines(coordinates = 'z', df = df_cent)


# In[27]:


# plot Z'(s) to avoid amplitude variability
aneurisk.plot_derivatives(coordinates = 'z', df = df_cent)


# ## Interpolation:

# In[28]:


# set the index equal to the patient ID
df_cent.index = df_cent['ID_patient']


# Firstly, try with only one patient:

# In[29]:


# example on only one patient

# select the interested columns
t = df_cent[['x','s','ID_patient', 'ID_centerline']]
# select the patient
t1 = t[t.ID_patient == 'C0001']
# select the centerline
t2 = t1[t1.ID_centerline == 0]
# select the x
x = t2['x']
# select the s
s = t2['s']
# use local regression to build the spline
spline_prova = localreg(np.array(s), np.array(x), kernel = rbf.gaussian, degree = 1, radius = 0.05)


# In[30]:


np.linalg.norm(x- spline_prova)


# In[31]:


# plot the spline
plt.figure(figsize = (20,6))
plt.plot(np.array(s), np.array(spline_prova))
plt.plot(np.array(s), np.array(x))
plt.xlabel('s')
plt.ylabel('Xspline')
plt.title('Patient C0001 - centerline 0')


# #### Now, create the spline for the whole ICA referred to each patient.

# Create the spline for X(s):

# In[32]:


# fit one 4th degree polynomial for the whole ICA wrt X(s)
l = aneurisk.localreg_and_plot(coordinates = 'x', df = df_cent)


# In[33]:


# create the dataset containing the XSpline coordinates
Xspline_df = pd.concat(l)
Xspline_df.index = np.arange(0, Xspline_df.shape[0])


# In[34]:


Xspline_df


# Create the spline for Y(s).

# In[35]:


# fit one 4th degree polynomial for the whole ICA wrt Y(s)
l = aneurisk.localreg_and_plot(coordinates = 'y', df = df_cent)


# In[36]:


# create the dataset containing the YSpline coordinates
Yspline_df = pd.concat(l)
Yspline_df.index = np.arange(0, Yspline_df.shape[0])


# In[37]:


Yspline_df


# Create the spline for Z(s):

# In[38]:


# fit one 4th degree polynomial for the whole ICA wrt Z(s)
l = aneurisk.localreg_and_plot(coordinates = 'z', df = df_cent)


# In[39]:


# create the dataset containing the ZSpline coordinates
Zspline_df = pd.concat(l)
Zspline_df.index = np.arange(0, Yspline_df.shape[0])


# In[40]:


Zspline_df


# #### Merge al the X, Y, Z splines dataframe:

# In[41]:


# create a unique dataframe
spline_df_temp = pd.merge(Xspline_df, Yspline_df, how = 'outer', left_index = True, right_index = True)
spline_df = pd.merge(spline_df_temp, Zspline_df, how = 'outer', left_index = True, right_index = True)


# In[42]:


# after checking that the merge was done correctly and that all the patients and centerlines' IDs coincide 
# remove the unuseful columns
spline_df.index = spline_df['ID_patient']
spline_df = spline_df.drop(['ID_patient_x', 'ID_patient_y', 'ID_centerline_x', 'ID_centerline_y', 
                            's_x', 's_y', 'ID_patient'], axis = 1)
spline_df['Radius'] = df_cent['r']


# In[43]:


spline_df


# In[44]:


# export the dataframe
spline_df.reset_index(inplace = True)
spline_df.to_feather('C:/Users/erica/Desktop/Thesis/spline_df.feather')


# ## Cluster Analysis

# The splines have not the same length. So, it is necessary to perform **padding** to get all the splines with the same length as the longest one.

# To extract the maximum length, I'll consider all the centerlines for each patient and I'll extract the maximum length for the n-th spline for the i-th patient. Then, I'll compare all the maximum patient's spline length and extract the final maximum length.

# In[45]:


# upload the dataset
spline_df = pd.read_feather('C:/Users/erica/Desktop/Thesis/spline_df.feather')
spline_df.index = spline_df['ID_patient']
spline_df = spline_df.drop('ID_patient', axis = 1)


# In[46]:


# create an empty list to fill with the max length for each of the 94 patients
ids = spline_df.index.unique()
max_len_temp = [0] * len(ids)

for i in range(len(ids)):
    max_len_temp[i] = spline_df.loc[ids[i], ['ID_centerline', 's']].groupby('ID_centerline').count().max()[0]
    
    
# extract the final maximum length
max_len = max(max_len_temp)
max_len


# In[47]:


# check the dataset
spline_df


# In[48]:


# remove the two outliers
spline_df = spline_df.drop(['C0009', 'C0038'], axis = 0)


# ### Padding:
# Here I'm creating a Dataframe with all the padded splines. In this way I got:
# * one row for each spline datapoints
# * 16050 points for each patient
# In this way the cluster should be done on each patients datapoint.

# In[49]:


spline_pad = aneurisk.df_opt1(df = spline_df, max_len = max_len)


# In[50]:


spline_pad


# In[51]:


# spline_pad.reset_index(inplace = True)
# spline_pad.to_feather('C:/Users/erica/Desktop/Thesis/pad_spline.feather')


# --------------------------------------------------------------------------------------------------------------------------------
# ### Option 1:
# 
# Apply the K-Means clustering considering a vector of (x, y, z) coordinates for each pointwise centerline for each patient.

# #### Cluster Analysis:

# In[52]:


spline = spline_pad
np.random.seed(10)
# create cluster label list
cluster_labels = [0]
# create empty list
splines_list = []

for i in range(1,4):
    
    # standardize the splines
    scaler = StandardScaler()
    splines_scaled = scaler.fit_transform(spline.iloc[:, [i]])
    splines_list.append(splines_scaled)
    
spline['Xscaled'] = splines_list[0]
spline['Yscaled'] = splines_list[1]
spline['Zscaled'] = splines_list[2]

# perform K-means clustering
num_clusters = 2
# create the pipeline which firstly standardize data and then compute the kmean cluster
# KMeans is run 10 times with different centroid seeds 
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=num_clusters, n_init = 42, random_state = 1))
# fit the pipeline
kmeans.fit(spline.iloc[:, 6:9])
    
# associate each spline with one cluster
cluster_labels = kmeans.predict(spline.iloc[:, 6:9])


# In[53]:


# create the final spline df associating the cluster labels
sp_df = spline
sp_df['ClusterLabel'] = cluster_labels


# In[54]:


sp_df


# #### Associate 1 cluster to each centerline:

# In[55]:


# see the dataframe
spline_df


# In[56]:


# obtain 1 cluster for each centerline spline using the maximum value criteria

ids = spline_df.index.unique()
label_list = []
not_working = []
for i in range(len(ids)):
    select = sp_df.loc[ids[i], ['ID_centerline', 'ClusterLabel']]
    max_cent = select['ID_centerline'].max()
    
    for j in range(max_cent + 1):
        
        # group by cluster
        data = select[select.ID_centerline == j][['ClusterLabel','ID_centerline']].groupby('ClusterLabel').count()
        max_value = data['ID_centerline'].max()
        
        # ensure that the maximum value is unique
        check = data[data.ID_centerline == max_value]
        
        if check.shape[0] > 1:
            print(f'There is a problem with patient {ids[i]}, centerline {j}.\nThe number of observation for each cluster is the same.')
            continue
            
        else:
            label = check.index[0]
            
            label_list.append(label)


# In[57]:


# crete the final label list
label_list = np.repeat(label_list, 1625)


# In[58]:


# associate the label to the dataframe column
sp_df['Label'] = label_list


# In[59]:


# check the new dataset
sp_df = sp_df.drop(['ClusterLabel', 'Xscaled', 'Yscaled', 'Zscaled'], axis = 1)
sp_df


# <!-- #### Associate 1 cluster to each centerline. -->

# <!-- Consider each centerline and associate to it the cluster that is more common within the 3 splines (X, Y, Z): -->

# ________________________________________________________________________________________________________________________________
# ### Option 2:
# 
# Now, instead, try to associate to each patient the array of splines. 
# In this way you can apply the cluster on the array.
# The dataframe will be composed of:
# * the 3 splines variable (X, Y, Z)
# * to each patient and each centerline is associated a vector of 1625 elements
# 
# In addition, check also to which cluster could be associated the radius measure.

# In[60]:


# see the padded dataframe
spline_pad


# In[61]:


# create the dataset
sp_df2 = aneurisk.df_opt2(df = spline_pad)


# In[62]:


# see the final dataset
sp_df2


# #### Perform KMeans clustering:

# In[63]:


# create the names list
names = ['Xspline', 'Yspline', 'Zspline', 'Radius']
# define the random seed
np.random.seed(5105752)


for i in range(len(names)):
    data = np.array(sp_df2[f'{names[i]}'].tolist())
    # standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    num_clusters = 2

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init = 10, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    # add the cluster labels
    sp_df2[f'{names[i]}_label'] = cluster_labels


# In[64]:


# see the dataset with the associated labels
sp_df2


# #### Associate 1 cluster to each centerline.

# In[65]:


# create the empty lists
label2 = []
problem2 = []
for i in sp_df2.index.unique():
    select = sp_df2.loc[i]
    max_cent = select['ID_centerline'].max()
    
    for j in range(max_cent + 1):
        data = select[select.ID_centerline == j]
        
        # check which label is the most common one
        
        if data['Xspline_label'][0] == data['Yspline_label'][0]:
            label2.append(data['Xspline_label'][0])
            
        elif data['Zspline_label'][0] == data['Xspline_label'][0]:
            label2.append(data['Xspline_label'][0])
            
        elif data['Yspline_label'][0] == data['Zspline_label'][0]:
            label2.append(data['Yspline_label'][0])
            
        else:
            problem2.append([i, j])
            print(f'There is a problem with patient {i} and centerline {j}.\nThe three splines are associated to different clusters.')


# In[66]:


# check the probelms
print(problem2)
len(problem2)


# ### Remove the ambiguous elements:
# 
# Remove those patients which had not a unique label associated to its centerlines using the 'problem' or 'problem2' list.

# ### Option 1:

# In[67]:


# sp_df = sp_df.reset_index()
# to_drop = [sp_df[(sp_df.ID_patient == i[0]) & (sp_df.ID_centerline == i[1])].index.values for i in problem]
# to_drop = np.concatenate(to_drop)
# to_drop


# In[68]:


# sp_df = sp_df.drop([to_drop[i] for i in range(len(to_drop))], axis = 0)
# sp_df


# In[69]:


# # create the dataframe with 1 label per centerline - option 1
# labels = np.repeat(label, 1625)
# sp_df = sp_df.reset_index()
# sp_df['Label'] = labels
# sp_df = sp_df.drop(['X_label', 'Y_label', 'Z_label', 'index'], axis = 1)
# sp_df


# ### Option 2:

# In[70]:


# sp_df2 = sp_df2.reset_index()
# to_drop2 = [sp_df2[(sp_df2.ID_patient == i[0]) & (sp_df2.ID_centerline == i[1])].index.values for i in problem2]
# to_drop2 = np.concatenate(to_drop2)
# to_drop2


# In[71]:


# sp_df2 = sp_df2.drop([to_drop2[i] for i in range(len(to_drop2))], axis = 0)
# sp_df2


# In[72]:


sp_df2 = sp_df2.reset_index()
sp_df2['Label'] = label2
sp_df2 = sp_df2.drop(['Xspline_label', 'Yspline_label', 'Zspline_label'], axis = 1)
sp_df2


# In[73]:


# check in how many cases the radius and the splines are associated to different clusters
count = 0
for i in range(sp_df2.shape[0]):
    if sp_df2['Radius_label'][i] == sp_df2['Label'][i]:
        count = count + 1
        
print(count)


# There are more than 2 hundred patients associated to different clusters. It's better to not consider it.

# #### Finally, associate the final cluster to each patient for both the dataset:

# ### Option 1:

# In[74]:


# make final associations
ids = sp_df.index.unique()
final_label, review, to_drop = aneurisk.final_labels(df = sp_df, ids = ids)


# In[75]:


# check the not added elements
review


# In[76]:


# drop the patients 
sp_df.index
sp_df = sp_df.drop(to_drop, axis = 0)
sp_df


# In[77]:


# associate 1 label to each patient
final_1 = pd.DataFrame({'ID_patient': sp_df.index.unique(),
                       'Label': np.array(final_label).flatten()})
final_1


# ### Option 2

# In[78]:


sp_df2.index = sp_df2['ID_patient']
sp_df2 = sp_df2.drop('ID_patient', axis = 1)


# In[79]:


# make final association
ids = sp_df2.index.unique()
final_label2, review2, to_drop2 = aneurisk.final_labels(df = sp_df2, ids = ids)


# In[80]:


# check the not added elements
review2


# In[81]:


# drop the patients before
sp_df2 = sp_df2.drop(to_drop2, axis = 0)
sp_df2


# In[82]:


# associate 1 label to each patient
final_2 = pd.DataFrame({'ID_patient': sp_df2.index.unique(),
                       'Label': np.array(final_label2).flatten()})
final_2


# ## Descriptive Analysis on Clusters

# Firstly, create the two **final** datasets:

# In[83]:


# see the descriptive dataset
final_data


# In[84]:


finals = pd.read_feather('C:/Users/erica/Desktop/Thesis/final_dataset.feather')
finals = finals.drop(['Mean_Top', 'Curvature', 'Max_Top', 'index'], axis = 1)
finals.index = finals['id']
finals = finals.drop(['id'], axis = 1)
finals


# In[85]:


# create the final dataframe for the first option

# create an empty list
to_drop = []
for i in finals.index.unique().tolist(): 
    # if the index in final_data is not present in final_1, drop it
    if i not in final_1['ID_patient'].unique().tolist():
        to_drop.append(i)
        
# drop the selected ids
final_data1 = finals.drop(to_drop, axis = 0)
# associate the cluster label
final_data1['Label'] = np.array(final_label).flatten()
final_data1


# In[86]:


# create the final dataframe for the second option

# create an empty list
to_drop = []
for i in finals.index.unique().tolist(): 
    # if the index in final_data is not present in final_2, drop it
    if i not in final_2['ID_patient'].unique().tolist():
        to_drop.append(i)
        
# drop the selected ids
final_data2 = finals.drop(to_drop, axis = 0)
# associate the cluster label
final_data2['Label'] = np.array(final_label2).flatten()
final_data2


# In[87]:


# create the dataframes for the first option
cl0_1 = final_data1[final_data1.Label == 0]
cl1_1 = final_data1[final_data1.Label == 1]
# cl2_1 = final_data1[final_data1.Label == 2]

# create the dataframes for the second option
cl0_2 = final_data2[final_data2.Label == 0]
cl1_2 = final_data2[final_data2.Label == 1]
# cl2_2 = final_data2[final_data2.Label == 2]


# Check **proportion of observations**:

# In[88]:


# proportion of obs
pd.DataFrame({'Cluster 0 - 1': [cl0_1.shape[0], f'{(cl0_1.shape[0]/final_1.shape[0])*100:.2f}%'],
              'Cluster 1 - 1': [cl1_1.shape[0], f'{(cl1_1.shape[0]/final_1.shape[0])*100:.2f}%'],
#               'Cluster 2 - 1': [cl2_1.shape[0], f'{(cl2_1.shape[0]/final_1.shape[0])*100:.2f}%'],
              'Cluster 0 - 2': [cl0_2.shape[0], f'{(cl0_2.shape[0]/final_2.shape[0])*100:.2f}%'],
              'Cluster 1 - 2': [cl1_2.shape[0], f'{(cl1_2.shape[0]/final_2.shape[0])*100:.2f}%']},
#               'Cluster 2 - 2': [cl2_2.shape[0], f'{(cl2_2.shape[0]/final_2.shape[0])*100:.2f}%']},
              index = ['Number', 'Proportion'])


# *Elements are balanced for option 2, and a bit unbalanced for option 1.*

# _______________________________________

# Check **male and female** proportion:
# 
# *Proportion is computed dividing the numer of male or female belonging to one specific cluster, by the overall number of male or female present in the dataset.*

# In[89]:


# MALE AND FEMALE

# option 1
tot_f1 = final_data1.groupby('sex')['sex'].count()[0]
tot_m1 = final_data1.groupby('sex')['sex'].count()[1]

nf_0_1 = cl0_1.groupby('sex')['sex'].count()[0]
nf_1_1 = cl1_1.groupby('sex')['sex'].count()[0]
# nf_2_1 = cl2_1.groupby('sex')['sex'].count()[0]

nm_0_1 = cl0_1.groupby('sex')['sex'].count()[1]
nm_1_1 = cl1_1.groupby('sex')['sex'].count()[1]
# nm_2_1 = cl2_1.groupby('sex')['sex'].count()[0]


# option 2
tot_f2 = final_data2.groupby('sex')['sex'].count()[0]
tot_m2 = final_data2.groupby('sex')['sex'].count()[1]

nf_0_2 = cl0_2.groupby('sex')['sex'].count()[0]
nf_1_2 = cl1_2.groupby('sex')['sex'].count()[0]
# nf_2_2 = cl2_2.groupby('sex')['sex'].count()[0]

nm_0_2 = cl0_2.groupby('sex')['sex'].count()[1]
nm_1_2 = cl1_2.groupby('sex')['sex'].count()[1]
# nm_2_2 = cl2_2.groupby('sex')['sex'].count()[0]

pd.DataFrame({'Cluster 0 - 1': [nf_0_1, f'{(nf_0_1/tot_f1)*100:.2f}%', nm_0_1, f'{(nm_0_1/tot_m1)*100:.2f}%'],
              
              'Cluster 1 - 1': [nf_1_1, f'{(nf_1_1/tot_f1)*100:.2f}%', nm_1_1, f'{(nm_1_1/tot_m1)*100:.2f}%'],
              
#               'Cluster 2 - 1': [nf_2_1, f'{(nf_2_1/tot_f1)*100:.2f}%', nm_2_1, f'{(nm_2_1/tot_m1)*100:.2f}%'],
              
              'Cluster 0 - 2': [nf_0_2, f'{(nf_0_2/tot_f2)*100:.2f}%', nm_0_2, f'{(nm_0_2/tot_m2)*100:.2f}%'],
             
              'Cluster 1 - 2': [nf_1_2, f'{(nf_1_2/tot_f2)*100:.2f}%', nm_1_2, f'{(nm_1_2/tot_m2)*100:.2f}%']},
              
#               'Cluster 2 - 2': [nf_2_2, f'{(nf_2_2/tot_f2)*100:.2f}%', nm_2_2, f'{(nm_2_2/tot_m2)*100:.2f}%']},
             
             index = ['F', 'F proportion ', 'M', 'M proportion'])


# *Considering the unbalanced proportion of male and females in the dataset, it seems that the distribution is quite similar for both the options. Specifically, the two clusters seem to be inverted for the two options.*

# **X-Square Test:**

# Check if the there is a relation between cluster and gender or they can be considered as independent:

# In[90]:


# option 1
contingency_table1 = pd.crosstab(final_data1['Label'], final_data1['sex'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table1)
print(p_value, chi2)


# In[91]:


# option 2
contingency_table2 = pd.crosstab(final_data2['Label'], final_data2['sex'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table2)
print(p_value, chi2)


# *Being both the p-value above 0.05, we can consider the proportion division as equal.*

# ********************************************************************************************************************************

# Check the **age** distribution:

# In[92]:


# AGE

#### compute the mean age for each cluster
# option 1
mean_cl0_1 = cl0_1['age'].mean()
mean_cl1_1 = cl1_1['age'].mean()
# mean_cl2_1 = cl2_1['age'].mean()

# option 2
mean_cl0_2 = cl0_2['age'].mean()
mean_cl1_2 = cl1_2['age'].mean()
# mean_cl2_2 = cl2_2['age'].mean()

# overall mean 1
overall_mean1 = final_data1['age'].mean()

# ovaralle mean 2
overall_mean2 = final_data2['age'].mean()

#### comput the mode age for each cluster
# option 1
mode_cl0_1 = cl0_1['age'].mode().to_list()
mode_cl1_1 = cl1_1['age'].mode().to_list()
# mode_cl2_1 = cl2_1['age'].mode().to_list()

# option 2
mode_cl0_2 = cl0_2['age'].mode().to_list()
mode_cl1_2 = cl1_2['age'].mode().to_list()
# mode_cl2_2 = cl2_2['age'].mode().to_list()

# ovarall mode 1
overall_mode1 = final_data1['age'].mode().to_list()

# ovarall mode 2
overall_mode2 = final_data2['age'].mode().to_list()

pd.DataFrame({'Cluster 0 - 1': [f'{mean_cl0_1:.4}', mode_cl0_1],
              
              'Cluster 1 - 1': [f'{mean_cl1_1:.4}', mode_cl1_1],
              
#               'Cluster 2 - 1': [f'{mean_cl2_1:.4}', mode_cl2_1],
              
              'Overall - 1': [f'{overall_mean1:.4}', overall_mode1],
              
              'Cluster 0 - 2': [f'{mean_cl0_2:.4}', mode_cl0_2],
              
              'Cluster 1 - 2': [f'{mean_cl1_2:.4}', mode_cl1_2],
              
#               'Cluster 2 - 2': [f'{mean_cl2_2:.4}', mode_cl2_2],
             
              'Overall - 2': [f'{overall_mean2:.4}', overall_mode2]},
             
              index = ['Mean', 'Mode'])


# **Mode:**
# - *the overall modes are 74 and 42. Most of the patient with age 74 are in cluster 0 for option 1; while in cluster 1 for option 2*.
# - *Younger patients seem to be associated to cluster 1 for opton 1 and in cluster 0 for option 2.*
# 
# **Mean:**
# - *The two overall mean are the same.* 
# - *Considering the cluster 0, in option 1 it has a mean age slightly greater than the overall mean.*
# - *Moving to cluster 1, in option 1 the age is smaller than the overall one.*
# - *Considering option 2, the mean of ages is smaller for cluster 0 and bigger for cluster 1.*
# 

# Check if age could be considered as different between clusters.
# 
# Check **ANOVA** assumptions.

# In[93]:


# option 1
# check normality
print('Option 1:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_1['age']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_1['age'])) 
#       '\ncl2: ', scipy.stats.shapiro(cl2_1['age']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):\n'
      'cl0: ', cl0_1['age'].std(), 
      '\ncl1: ', cl1_1['age'].std())
#       '\ncl2: ', cl2_1['age'].std())


# Assumption for ANOVA are respected, we can perform ANOVA one-way test:

# In[94]:


f_oneway(cl0_1['age'], cl1_1['age'])#, cl2_1['age'])


# In[95]:


# option 2
# check normality
print('Option 2/:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_2['age']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_2['age']))
      #'\ncl2: ', scipy.stats.shapiro(cl2_2['age']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_2['age'].std(), 
      '\ncl1: ', cl1_2['age'].std()) 
#       '\ncl2: ', cl2_2['age'].std())


# Assumption for ANOVA are respected, we can perform ANOVA one-way test:

# In[96]:


# option 2
f_oneway(cl0_2['age'], cl1_2['age'])#, cl2_2['age'])


# In both cases the age can not be considered a discriminant. Indeed p-value is above 0.05. 

# ___________

# Check **aneurysm type**:

# In[97]:


# aneurysm type

# count the total number of patient per aneurysm type 
# option 1
tot_lat1 = final_data1.groupby('aneurysmType')['aneurysmType'].count()[0]
tot_ter1 = final_data1.groupby('aneurysmType')['aneurysmType'].count()[1]

# option2
tot_lat2 = final_data2.groupby('aneurysmType')['aneurysmType'].count()[0]
tot_ter2 = final_data2.groupby('aneurysmType')['aneurysmType'].count()[1]

# count for each cluster
# option 1
lat_0_1 = cl0_1.groupby('aneurysmType')['aneurysmType'].count()[0]
lat_1_1 = cl1_1.groupby('aneurysmType')['aneurysmType'].count()[0]
# lat_2_1 = cl2_1.groupby('aneurysmType')['aneurysmType'].count()[0]

ter_0_1 = cl0_1.groupby('aneurysmType')['aneurysmType'].count()[1]
ter_1_1 = cl1_1.groupby('aneurysmType')['aneurysmType'].count()[1]
# ter_2_1 = cl2_1.groupby('aneurysmType')['aneurysmType'].count()[1]

# option 2
lat_0_2 = cl0_2.groupby('aneurysmType')['aneurysmType'].count()[0]
lat_1_2 = cl1_2.groupby('aneurysmType')['aneurysmType'].count()[0]
# lat_2_2 = cl2_2.groupby('aneurysmType')['aneurysmType'].count()[0]

ter_0_2 = cl0_2.groupby('aneurysmType')['aneurysmType'].count()[1]
ter_1_2 = cl1_2.groupby('aneurysmType')['aneurysmType'].count()[1]
# ter_2_2 = cl2_2.groupby('aneurysmType')['aneurysmType'].count()[1]


pd.DataFrame({'Cluster 0 - 1': [lat_0_1, f'{(lat_0_1/tot_lat1)*100:.2f}%', ter_0_1, f'{(ter_0_1/tot_ter1)*100:.2f}%'],
              
              'Cluster 1 - 1': [lat_1_1, f'{(lat_1_1/tot_lat1)*100:.2f}%', ter_1_1, f'{(ter_1_1/tot_ter1)*100:.2f}%'],
              
#               'Cluster 2 - 1': [lat_2_1, f'{(lat_2_1/tot_lat1)*100:.2f}%', ter_2_1, f'{(ter_2_1/tot_ter1)*100:.2f}%'],
              
              'Cluster 0 - 2': [lat_0_2, f'{(lat_0_2/tot_lat2)*100:.2f}%', ter_0_2, f'{(ter_0_2/tot_ter2)*100:.2f}%'],
              
              'Cluster 1 - 2': [lat_1_2, f'{(lat_1_2/tot_lat2)*100:.2f}%', ter_1_2, f'{(ter_1_2/tot_ter2)*100:.2f}%']},
               
#               'Cluster 2 - 2': [lat_2_2, f'{(lat_2_2/tot_lat2)*100:.2f}%', ter_2_2, f'{(ter_2_2/tot_ter2)*100:.2f}%']},
             
             index = ['Lateral', 'Lateral proportion', 'Terminal', 'Terminal proportion'])


# *From the above table we can deduct that:*
# 
# **Cluster 0:**
# - *In option 1 the proportion wrt to the total number of patient with lateral and terminal aneurysm is higher for terminal proportion. There is a significant prevalence of terminal proportion. While for option 2, looking at the pure numbers, there are more patients having an aneurysm on the terminal portion; moving instead to proportion of elements, there are more patients having an aneurysm on the lateral portion associated to this cluster.*
# 
# **Cluster 1:**
# - *In option 1, there is an equal number of patients wrt to the type of aneurysm. Considering instead the proportion of assigned elements, there is a prevalence of assigned laterals. Turning to option 2, the situation is inverted. Looking at both the numbers and the proportions, there is a prevalence of terminal aneurysms.*

# **X-Square Test:**
# 
# Check if there is a relation between cluster and aneurysm type or they can be considered as independent:

# In[98]:


# option 1
contingency_table1 = pd.crosstab(final_data1['Label'], final_data1['aneurysmType'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table1)
print(p_value, chi2)


# In[99]:


# option 2
contingency_table2 = pd.crosstab(final_data2['Label'], final_data2['aneurysmType'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table2)
print(p_value, chi2)


# The difference in proportion is not significant. Variables can be considered as independently distributed.

# _____________

# Check **Rupture Status**:

# In[100]:


# rupture status

# compute the total number of patients according to the status of the aneurysm
# option 1
tot_R_1 = final_data1.groupby('ruptureStatus')['ruptureStatus'].count()[0]
tot_U_1 = final_data1.groupby('ruptureStatus')['ruptureStatus'].count()[1]

# option 2
tot_R_2 = final_data2.groupby('ruptureStatus')['ruptureStatus'].count()[0]
tot_U_2 = final_data2.groupby('ruptureStatus')['ruptureStatus'].count()[1]

# counts the rupture status' number of patients for each cluster
# option 1
R_0_1 = cl0_1.groupby('ruptureStatus')['ruptureStatus'].count()[0]
U_0_1 = cl0_1.groupby('ruptureStatus')['ruptureStatus'].count()[1]

R_1_1 = cl1_1.groupby('ruptureStatus')['ruptureStatus'].count()[0]
U_1_1 = cl1_1.groupby('ruptureStatus')['ruptureStatus'].count()[1]


# option 2
R_0_2 = cl0_2.groupby('ruptureStatus')['ruptureStatus'].count()[0]
U_0_2 = cl0_2.groupby('ruptureStatus')['ruptureStatus'].count()[1]

R_1_2 = cl1_2.groupby('ruptureStatus')['ruptureStatus'].count()[0]
U_1_2 = cl1_2.groupby('ruptureStatus')['ruptureStatus'].count()[1]


pd.DataFrame({'Cluster 0 - 1': [R_0_1, f'{(R_0_1/tot_R_1)*100:.2f}%', 
                                U_0_1, f'{(U_0_1/tot_U_1)*100:.2f}%'],
            
              'Cluster 1 - 1': [R_1_1, f'{(R_1_1/tot_R_1)*100:.2f}%',
                                U_1_1, f'{(U_1_1/tot_U_1)*100:.2f}%'],
           
              'Cluster 0 - 2': [R_0_2, f'{(R_0_2/tot_R_2)*100:.2f}%',
                                U_0_2, f'{(U_0_2/tot_U_2)*100:.2f}%'],
             
              'Cluster 1 - 2': [R_1_2, f'{(R_1_2/tot_R_2)*100:.2f}%',
                                U_1_2, f'{(U_1_2/tot_U_2)*100:.2f}%']},
          
             index = ['Ruptured', 'Ruptured proportion', 'Unruptured', 'Unruptured proportion'])


# *For all the clusters the distribution seems to be quite balanced. Let's check it statistically.*

# **X-Square Test:**

# In[101]:


# option 1
contingency_table1 = pd.crosstab(final_data1['Label'], final_data1['ruptureStatus'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table1)
print(p_value, chi2)


# In[102]:


# option 2
contingency_table2 = pd.crosstab(final_data2['Label'], final_data2['ruptureStatus'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table2)
print(p_value, chi2)


# P-values confirm what assumed above.

# _______________________________________________________________________________________________________________________

# **Radius**:

# In[103]:


# radius

max_01 = cl0_1['Radius'].max()
max_11 = cl1_1['Radius'].max()

max_02 = cl0_2['Radius'].max()
max_12 = cl1_2['Radius'].max()

mean_01 = cl0_1['Radius'].mean()
mean_11 = cl1_1['Radius'].mean()


mean_02 = cl0_2['Radius'].mean()
mean_12 = cl1_2['Radius'].mean()

pd.DataFrame({'Cluster 0 - opt1': [max_01, mean_01],

              'Cluster 1 - opt1': [max_02, mean_12],

             'Cluster 0 - opt2': [max_01, mean_01],

             'Cluster 1 - opt2': [max_02, mean_12]},

             index = ['Max Radius', 'Mean Radius'])


# **ANOVA** test.
# 
# Check ANOVA Assumptions:

# In[104]:


# option 1
# check normality
print('Option 2/:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_1['Radius']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_1['Radius']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_1['Radius'].std(), 
      '\ncl1: ', cl1_1['Radius'].std())


# In[105]:


# option 1
f_oneway(cl0_1['Radius'], cl1_1['Radius'])


# *They result as statistically different.*

# In[106]:


# plot boxplots of radius distribution

import seaborn as sns
data = final_data1[['Radius', 'Label']]
sns.boxplot(x='Label', y='Radius', data=data, palette="Blues")
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.grid(True, linestyle='-', alpha=0.3)
plt.gca().set_facecolor('whitesmoke')
plt.title('Radius Box-Plot by Clusters')

plt.xticks([0, 1], ['1', '2'])

plt.show()


# In[107]:


# option 2
# check normality
print('Option 2/:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_2['Radius']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_2['Radius']))
      #'\ncl2: ', scipy.stats.shapiro(cl2_2['age']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_2['Radius'].std(), 
      '\ncl1: ', cl1_2['Radius'].std()) 
#       '\ncl2: ', cl2_2['age'].std())


# In[108]:


# option 2
f_oneway(cl0_2['Radius'], cl1_2['Radius'])#, cl2_2['age'])


# *They do not result as statistically different.*

# **Aneurysm Location:**

# In[109]:


# aneurysm location

# count the total number of patient per aneurysm location
tot_aca = finals.groupby('aneurysmLocation')['aneurysmLocation'].count()[0]
tot_bas = finals.groupby('aneurysmLocation')['aneurysmLocation'].count()[1]
tot_ica = finals.groupby('aneurysmLocation')['aneurysmLocation'].count()[2]
tot_mca = finals.groupby('aneurysmLocation')['aneurysmLocation'].count()[3]

# OPTION 1
# count for each cluster
aca_01 = cl0_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[0]
aca_11 = cl1_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[0]

bas_01 = cl0_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[1]
bas_11 = cl1_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[1]

ica_01 = cl0_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[2]
ica_11 = cl1_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[2]

mca_01 = cl0_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[3]
mca_11 = cl1_1.groupby('aneurysmLocation')['aneurysmLocation'].count()[3]



# OPTION 2
# count for each cluster
aca_02 = cl0_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[0]
aca_12 = cl1_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[0]

bas_02 = cl0_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[1]
bas_12 = cl1_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[1]

ica_02 = cl0_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[2]
ica_12 = cl1_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[2]

mca_02 = cl0_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[3]
mca_12 = cl1_2.groupby('aneurysmLocation')['aneurysmLocation'].count()[3]


pd.DataFrame({'Cluster 0 - opt1': [aca_01, f'{(aca_01/tot_aca)*100:.2f}%', bas_01, f'{(bas_01/tot_bas)*100:.2f}%', ica_01, f'{(ica_01/tot_ica)*100:.2f}%', mca_01, f'{(mca_01/tot_mca)*100:.2f}%'],

              'Cluster 1 - opt1': [aca_11, f'{(aca_11/tot_aca)*100:.2f}%', bas_11, f'{(bas_11/tot_bas)*100:.2f}%', ica_11, f'{(ica_11/tot_ica)*100:.2f}%', mca_11, f'{(mca_11/tot_mca)*100:.2f}%'],

             'Cluster 0 - opt2': [aca_02, f'{(aca_02/tot_aca)*100:.2f}%', bas_02, f'{(bas_02/tot_bas)*100:.2f}%', ica_02, f'{(ica_02/tot_ica)*100:.2f}%', mca_02, f'{(mca_02/tot_mca)*100:.2f}%'],

             'Cluster 1 - opt2': [aca_12, f'{(aca_12/tot_aca)*100:.2f}%', bas_12, f'{(bas_12/tot_bas)*100:.2f}%', ica_12, f'{(ica_12/tot_ica)*100:.2f}%', mca_12, f'{(mca_12/tot_mca)*100:.2f}%']},

             index = ['ACA', 'ACA Proportion', 'BAS', 'BAS Proportion', 'ICA', 'ICA Proportion', 'MCA', 'MCA Proportion'])


# **X-Square Test:**

# In[110]:


# option 1
contingency_table1 = pd.crosstab(final_data1['Label'], final_data1['aneurysmLocation'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table1)
print(p_value, chi2)


# In[111]:


# option 2
contingency_table2 = pd.crosstab(final_data2['Label'], final_data2['aneurysmLocation'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table2)
print(p_value, chi2)


# *Variables can be considered as not related, but as independent for both the options.*

# ___________________

# #### Check the **curvature**:

# In[112]:


spline = pd.read_feather('C:/Users/erica/Desktop/Thesis/pad_spline.feather')


# In[116]:


spline


# In[114]:


# analysis on curvature - c(s) = (x(s), y(s), z(s))

c = np.array([spline['Xspline'].values, spline['Yspline'].values, spline['Zspline'].values])
c = np.transpose(c)
c


# In[117]:


# create the curvature vectors
spline = spline.reset_index()
spline['C'] = np.split(c.flatten(), spline.shape[0])
spline


# In[118]:


# compute the DERIVATIVES

# create a list of ids
ids = spline['ID_patient'].unique().tolist()
# create two empty arrays
# c_prime = np.array([])
c = pd.DataFrame(columns = ['ID_patient', 'ID_centerline','FirstDerivative', 'SecondDerivative'])
# c_second = np.array([])
for i in ids:
    # select ith patient
    select = spline[spline.ID_patient == i]
    # find numer of centerlines for each patient
    max_cent = select['ID_centerline'].unique()
    
    for j in max_cent:
        s = np.vstack(select[select.ID_centerline == j]['s'])
        
        # compute the c derivative wrt s
        first_der = diff(np.vstack(select[select.ID_centerline == j]['C']), axis = 0) / diff(s, axis = 0)

        # compute second derivative
        sec_der = diff(first_der, axis = 0) / diff(s[:-1], axis = 0)

        # remove NaN
        # given that padding was performed copying the first and last element half times at the beginning and at the end
        # NaN are given by this reason -> having equal number, the diff function has 0 as result
        first_der = np.nan_to_num(first_der, nan = 0)
        sec_der = np.nan_to_num(sec_der, nan = 0)
    
        c = c.append({'ID_patient': i, 'ID_centerline': j, 
                      'FirstDerivative': first_der[:-1], 
                      'SecondDerivative': sec_der}, 
                       ignore_index = True)


# In[119]:


c


# In[120]:


# compute the CURVATURE 

curvature = []
k = 0
for i in c['ID_patient'].unique():
    # select the ith patient
    select = c[c.ID_patient == i]
    # create a list with all the centerline values
    cent = select['ID_centerline'].unique()
    
    for j in cent:
        # compute first derivative
        first_derivative = select[select.ID_centerline == j]['FirstDerivative'][k]
        # compute second derivative
        second_derivative = select[select.ID_centerline == j]['SecondDerivative'][k]
        # update the rows counter
        k += 1

        # compute the cross product of the two vectors
        prod = np.cross(first_derivative, second_derivative)
        # compute the norm of the product
        curv = np.linalg.norm(prod, axis = 1)
        # compute denominator (norm of first derivative)
        first_der_3 = np.linalg.norm(first_derivative, axis = 1)
        # compute the final curvature vector
        final_curvature = curv / (first_der_3)**3
        
        # create the final list
        curvature.append(final_curvature)
        
        
# avoid NaN
for i in range(len(curvature)):
    curvature[i] = np.nan_to_num(curvature[i], nan=0)


# In[121]:


c['ID_patient'].unique()


# In[122]:


# associate to each patient and centerline the curvature array
c['Curvature'] = curvature
c


# In[123]:


# create a new column wich has for each patient and centerline the maximum curvature value
c['Max_Curvature'] = [c.loc[i, 'Curvature'].max() for i in range(c.shape[0])]
c


# **Option 1:**

# In[124]:


# create a new dataframe with each patient and the maxium and the mean curvature value
max_list = []
mean_list = []
for i in final_data1.index:
    # select the ith patient
    select = c[c.ID_patient == i]
    # compute the maximum curvature within the centerlines
    max_curv = select['Max_Curvature'].max()
    # compute the mean of maximum curvature within the centerlines
    mean_curv = select['Max_Curvature'].mean()
    
    # create the lists
    max_list.append(max_curv)
    mean_list.append(mean_curv)
    
    
# create the final new dataframe
final_data1['Max_Curvature'] = max_list
final_data1['Mean_Curvature'] = mean_list

final_data1


# **Option 2:**

# In[125]:


# create a new dataframe with each patient and the maxium and the mean curvature value
max_list = []
mean_list = []
for i in final_data2.index:
    # select the ith patient
    select = c[c.ID_patient == i]
    # compute the maximum curvature within the centerlines
    max_curv = select['Max_Curvature'].max()
    # compute the mean of maximum curvature within the centerlines
    mean_curv = select['Max_Curvature'].mean()
    
    # create the lists
    max_list.append(max_curv)
    mean_list.append(mean_curv)
    
    
# create the final new dataframe
final_data2['Max_Curvature'] = max_list
final_data2['Mean_Curvature'] = mean_list

final_data2


# **Descriptive Analysis on Curvature**:

# In[126]:


# create the dataframes for the first option
cl0_1 = final_data1[final_data1.Label == 0]
cl1_1 = final_data1[final_data1.Label == 1]

# create the dataframes for the second option
cl0_2 = final_data2[final_data2.Label == 0]
cl1_2 = final_data2[final_data2.Label == 1]


# In[127]:


max_mean0_1 = cl0_1['Max_Curvature'].mean()
max_mean1_1 = cl1_1['Max_Curvature'].mean()

mean_mean0_1 = cl0_1['Mean_Curvature'].mean()
mean_mean1_1 = cl1_1['Mean_Curvature'].mean()

max_max0_1 = cl0_1['Max_Curvature'].max()
max_max1_1 = cl1_1['Max_Curvature'].max()

max_mean0_2 = cl0_2['Max_Curvature'].mean()
max_mean1_2 = cl1_2['Max_Curvature'].mean()

mean_mean0_2 = cl0_2['Mean_Curvature'].mean()
mean_mean1_2 = cl1_2['Mean_Curvature'].mean()

max_max0_2 = cl0_2['Max_Curvature'].max()
max_max1_2 = cl1_2['Max_Curvature'].max()

pd.DataFrame({'Cluster 0 - 1': [max_mean0_1, mean_mean0_1, max_max0_1],
            
              'Cluster 1 - 1': [max_mean1_1, mean_mean1_1, max_max1_1],
              
              'Cluster 0 - 2': [max_mean0_2, mean_mean0_2, max_max0_2],
            
              'Cluster 1 - 2': [max_mean1_2, mean_mean1_2, max_max1_2]},
              
              index = ['Mean of Maximum', 'Mean of Mean', 'Max of Maximum'])
          


# Check if curvature can be considered a statistically discriminant factor between clusters.
# 
# **ANOVA** Test assumptions:

# Maximum curvature check:

# In[128]:


# option 1 - max curvature
# check normality
print('Option 1:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_1['Max_Curvature']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_1['Max_Curvature'])) 

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_1['Max_Curvature'].std(), 
      '\ncl1: ', cl1_1['Max_Curvature'].std()) 


# In[129]:


# option 2 - max curvature
# check normality
print('Option 2:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_2['Max_Curvature']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_2['Max_Curvature'])) 
#       '\ncl2: ', scipy.stats.shapiro(cl2_2['Max_Curvature']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_2['Max_Curvature'].std(), 
      '\ncl1: ', cl1_2['Max_Curvature'].std()) 
#       '\ncl2: ', cl2_2['Max_Curvature'].std())


# In both cases ANOVA assumptions are **not** respected. 

# Mean Curvature Check:

# In[130]:


# option 1 - mean curvature
# check normality
print('Option 1:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_1['Mean_Curvature']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_1['Mean_Curvature'])) 
#       '\ncl2: ', scipy.stats.shapiro(cl2_1['Mean_Curvature']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_1['Mean_Curvature'].std(), 
      '\ncl1: ', cl1_1['Mean_Curvature'].std())
#       '\ncl2: ', cl2_1['Mean_Curvature'].std())


# In[131]:


# option 2 - mean curvature
# check normality
print('Option 2:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_2['Mean_Curvature']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_2['Mean_Curvature'])) 
#       '\ncl2: ', scipy.stats.shapiro(cl2_2['Max_Curvature']))

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_2['Mean_Curvature'].std(), 
      '\ncl1: ', cl1_2['Mean_Curvature'].std()) 
#       '\ncl2: ', cl2_2['Mean_Curvature'].std())


# Normality assumptions and homogeneity of variances are **not** satisfied. So, ANOVA can not be applied. 

# Let's apply the **Kruskal-Wallis H-test** for independent samples. It does not assum that data come from a specific distribution.

# In[132]:


# maximum value:
# option 1
scipy.stats.kruskal(cl0_1['Max_Curvature'], cl1_1['Max_Curvature'])


# In[133]:


# maximum value:
# option 2
scipy.stats.kruskal(cl0_2['Max_Curvature'], cl1_2['Max_Curvature'])


# In[134]:


# mean value:
# option 1
scipy.stats.kruskal(cl0_1['Mean_Curvature'], cl1_1['Mean_Curvature'])


# In[135]:


# mean value:
# option 2
scipy.stats.kruskal(cl0_2['Mean_Curvature'], cl1_2['Mean_Curvature'])


# *Results suggest amìny statistical difference between curvature values distribution*.

# # PCA
# 
# To summarize better the curvature vectors, try to perform Principal Component Analysis on them.

# In[136]:


# upload the libraries
from sklearn.decomposition import PCA


# In[137]:


# stack vertially curvature values
curv_pca = np.vstack(c['Curvature'].values).T


# In[138]:


# check the shape
curv_pca.shape


# In[139]:


# scale arrays
scaler = StandardScaler()
curv_scaled = scaler.fit_transform(curv_pca)


# In[140]:


# perform PCA
pca = PCA(n_components=10)
pca.fit_transform(curv_scaled)


# In[141]:


# see the explained variance
print(pca.explained_variance_ratio_)


# In[142]:


# see graphically the variance explaines
color = sns.color_palette("Blues")[3]
plt.plot(np.arange(1, 11), pca.explained_variance_ratio_, '-o', color = color)
plt.title('PCA explained variance')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.grid(True, linestyle='-', alpha=0.3)
plt.gca().set_facecolor('whitesmoke')


# Considering the distribution of the explained variance and the scope of this analysis, consider only **one Principal Component** value.

# In[143]:


# perform again PCA
pca = PCA(n_components=1)
pca.fit_transform(curv_scaled)


# In[144]:


# select the pc values
curv_pca = pca.components_.flatten().tolist()
len(curv_pca)


# In[145]:


# assocaite the values to te dataframe
c['Curv_PCA'] = curv_pca


# In[146]:


# create a new dataframe with each patient and the maxium and the mean curvature value
pca_list = []

for i in finals.index.unique():
    # select the ith patient
    select = c[c.ID_patient == i]
    # compute the maximum curvature within the centerlines
    pca_curv = select['Curv_PCA'].max()

    pca_list.append(pca_curv)


# create the final new dataframe
finals['Curv_PCA'] = pca_list

finals


# #### Option 1:

# In[147]:


# create the final dataframe for the first option

# create an empty list
to_drop = []
for i in finals.index.unique().tolist(): 
    # if the index in final_data is not present in final_1, drop it
    if i not in final_1['ID_patient'].unique().tolist():
        to_drop.append(i)
        
# drop the selected ids
final_data1 = finals.drop(to_drop, axis = 0)
# associate the cluster label
final_data1['Label'] = np.array(final_label).flatten()
final_data1


# #### Option 2:

# In[148]:


# create the final dataframe for the second option

# create an empty list
to_drop = []
for i in finals.index.unique().tolist(): 
    # if the index in final_data is not present in final_2, drop it
    if i not in final_2['ID_patient'].unique().tolist():
        to_drop.append(i)
        
# drop the selected ids
final_data2 = finals.drop(to_drop, axis = 0)
# associate the cluster label
final_data2['Label'] = np.array(final_label2).flatten()
final_data2


# In[149]:


# create the dataframes for the first option
cl0_1 = final_data1[final_data1.Label == 0]
cl1_1 = final_data1[final_data1.Label == 1]

# create the dataframes for the second option
cl0_2 = final_data2[final_data2.Label == 0]
cl1_2 = final_data2[final_data2.Label == 1]


# In[150]:


max_mean01 = cl0_1['Curv_PCA'].mean()
max_mean11 = cl1_1['Curv_PCA'].mean()
max_mean02 = cl0_2['Curv_PCA'].mean()
max_mean12 = cl1_2['Curv_PCA'].mean()

mean_mean01 = cl0_1['Curv_PCA'].mean()
mean_mean11 = cl1_1['Curv_PCA'].mean()
mean_mean02 = cl0_2['Curv_PCA'].mean()
mean_mean12 = cl1_2['Curv_PCA'].mean()

max_max01 = cl0_1['Curv_PCA'].max()
max_max11 = cl1_1['Curv_PCA'].max()
max_max02 = cl0_2['Curv_PCA'].max()
max_max12 = cl1_2['Curv_PCA'].max()

pd.DataFrame({'Cluster 0 - opt1': [max_mean01, mean_mean01, max_max01],

              'Cluster 1 - opt1': [max_mean11, mean_mean11, max_max11],

              'Cluster 0 - opt2': [max_mean02, mean_mean02, max_max02],

              'Cluster 1 - opt2': [max_mean12, mean_mean12, max_max12]},

              index = ['Mean of Maximum - pca', 'Mean of Mean - pca', 'Max of Maximum - pca'])


# Perform **ANOVA test**. 
# 
# Check ANOVA assumptions:

# In[151]:


# option 1 
# check normality
print('Option 1:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_1['Curv_PCA']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_1['Curv_PCA'])) 

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_1['Curv_PCA'].std(), 
      '\ncl1: ', cl1_1['Curv_PCA'].std())


# In[152]:


# option 2
# check normality
print('Option 1:\n\n', 'Shapiro Test:\n',
      'cl0: ', scipy.stats.shapiro(cl0_2['Curv_PCA']), 
      '\ncl1: ', scipy.stats.shapiro(cl1_2['Curv_PCA'])) 

# check homogeneity of variance
print('\n\nHomogeneity of Variance (std):',
      '\ncl0: ', cl0_2['Curv_PCA'].std(), 
      '\ncl1: ', cl1_2['Curv_PCA'].std())


# Not all the ANOVA assumptions are respected, check if with **Kruskal-Wallis H Test.**

# In[153]:


scipy.stats.kruskal(cl0_1['Curv_PCA'], cl1_1['Curv_PCA'])


# In[154]:


scipy.stats.kruskal(cl0_2['Curv_PCA'], cl1_2['Curv_PCA'])


# *With the last test, curvature PCA values are different also under a statistical point of view. However, considering the results obtained before, it is not possible to derive any conclusion about specific and peculiar features emerging fron the cluster subdivision.*

# In[ ]:




