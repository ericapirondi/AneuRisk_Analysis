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

class Aneurisk:

    def plot_splines(self, coordinates, df):
        # plot the X(s) or Y(s) or Z(s)
        for i in df['ID_patient'].unique():
            # select the i-ih patient
            select = df[df.ID_patient == i][['ID_centerline', coordinates, 's']]
            # find the numer of centerlines
            maximum = select['ID_centerline'].max()
    
            for j in range(maximum+1):
                # plot each centerline in the same graph for all the patients
                plt.plot(select[select.ID_centerline == j]['s'], select[select.ID_centerline == j][coordinates], label = f'{j} cent.')
            
            plt.xlabel('S')
            plt.ylabel(f'{coordinates.capitalize()}(s)')
            plt.title(f'{i} Patient Centerlines')
            plt.legend()
            plt.show()
        


    def plot_derivatives(self, coordinates, df):
        # Take first derivative wrt X(s) to avoid amplitude variability
        for i in df['ID_patient'].unique():
            # select the i-th patient 
            select = df[df.ID_patient == i][['ID_centerline', coordinates, 's']]
            # find the number of centerlines for that patient
            maximum = select['ID_centerline'].max()
            
            plt.figure(figsize=(20, 6))
            
            for j in range(maximum+1):
                # select those datapoints where the derivative of the 's' variable is not zero
                to_plot = np.where(diff(select[select.ID_centerline == j]['s'])!=0)[0]
                # take the derivative of X wrt S
                der_x = diff(select[select.ID_centerline == j][coordinates])[to_plot] / diff(select[select.ID_centerline == j]['s'])[to_plot]
                # plot the derivatives
                plt.plot(select[select.ID_centerline == j]['s'].to_numpy()[to_plot],der_x, label = f'{j} cent.')
            
    
            plt.xlabel('S')
            plt.ylabel(f"{coordinates.capitalize()}'(s)")
            plt.title(f'Patient Centerlines - {i}')
            plt.show()


    def localreg_and_plot(self, coordinates, df):
        # fit one 4th degree polynomial for the whole ICA wrt X(s) or Y(s) or Z(s)
        l = []
        for j in df['ID_patient'].unique().tolist():
            # create a temporary dataframe containing the centerline for each patient one at time
            temp = df[df.ID_patient == j]
            maximum_cent = temp['ID_centerline'].max()
            
            # define a scale of colors for the first graph
            cmap = plt.get_cmap('winter')
            colors1 = [cmap(i / maximum_cent) for i in range(maximum_cent + 1)]
            
            # define a scale of colors for the second graph
            cmap = plt.get_cmap('autumn')
            colors2 = [cmap(i / maximum_cent) for i in range(maximum_cent + 1)]

            fig, ax = plt.subplots(figsize = (20,6))
            
            for i in range(maximum_cent+1):
                coord = temp[temp.ID_centerline == i][coordinates]
                s = temp[temp.ID_centerline == i]['s']
                    
                spline = localreg(np.array(s), np.array(coord), kernel = rbf.gaussian, degree = 4, radius = 0.05)
            
                
                ax.plot(s, spline, alpha = 0.4, linewidth = 5, label = f'{i} spline', color = colors1[i])
                ax.plot(s, coord, linestyle = '--', linewidth = 1.7, label = f'{i} centerline', color = colors2[i])
                
                data = pd.DataFrame({f'{coordinates.capitalize()}spline': spline, coordinates: coord, 's': s})
                data['ID_patient'] = j
                data['ID_centerline'] = i
                l.append(data)
                        
            plt.xlabel('S')
            plt.title(f'Spline vs Centerline - {j} patient')
            plt.legend(fontsize = 8)
            plt.show()

        return l


    def df_opt1(self, df, max_len):
        # create the empty dataframe
        ids = df.index.unique()
        spline_pad = pd.DataFrame(columns = ['ID_patient', 'ID_centerline', 'Xspline', 'Yspline', 'Zspline', 's', 'Radius'])

        summ = 0
        # create a dataframe with padded centerlines
        for i in range(len(ids)):
            select = df.loc[ids[i], ['ID_centerline', 'Xspline', 'Yspline', 'Zspline', 's', 'Radius']]
            max_cent = select['ID_centerline'].max()
            
            # check the number of rows the dataset should have
            summ = summ + ((max_cent+1)*1625)
            
            for j in range(max_cent+1):
                data = select[select.ID_centerline == j]
                
                pad_length_before = (max_len - data['Xspline'].shape[0])//2
                pad_length_after = max_len - data['Xspline'].shape[0] - pad_length_before
                
                # pad at the beginning with the first value and at the end with the alst value
                pad_x = np.pad(data['Xspline'], (pad_length_before, pad_length_after), 
                            'constant', constant_values=(data.iloc[0,1], data.iloc[-1,1]))
                
                pad_y = np.pad(data['Yspline'], (pad_length_before, pad_length_after), 
                            'constant', constant_values=(data.iloc[0,2], data.iloc[-1,2]))
                
                pad_z = np.pad(data['Zspline'], (pad_length_before, pad_length_after), 
                            'constant', constant_values=(data.iloc[0,3], data.iloc[-1,3]))
                
                pad_s = np.pad(data['s'], (pad_length_before, pad_length_after), 
                            'constant', constant_values=(data.iloc[0,4], data.iloc[-1,4]))
                
                pad_r = np.pad(data['Radius'], (pad_length_before, pad_length_after), 
                            'constant', constant_values=(data.iloc[0,5], data.iloc[-1,5]))
                
                pad_IDpatient = [ids[i]] * (max_len)
                pad_IDcent = [j] * max_len
                
                spline_pad = spline_pad.append(pd.DataFrame({'ID_patient': pad_IDpatient,
                                                'ID_centerline': pad_IDcent, 
                                                'Xspline': pad_x, 
                                                'Yspline': pad_y, 
                                                'Zspline': pad_z, 
                                                's': pad_s,
                                                'Radius': pad_r}), ignore_index=True)
                
        #         sp_df.loc[len(sp_df)] = [ids[i], j, pad_x, pad_y, pad_z, pad_s]
        spline_pad.index = spline_pad['ID_patient']
        spline_pad = spline_pad.drop('ID_patient', axis = 1)

        # check the number of rows to be sure that everything is correct
        if summ == spline_pad.shape[0]:
            print(True)

        return spline_pad


    def df_opt2(self, df):
        # create an empty dataframe 
        ids = df.index.unique()
        sp_df2 = pd.DataFrame(columns = ['ID_patient', 'ID_centerline', 'Xspline', 'Yspline', 'Zspline', 's', 'Radius'])

        for i in range(len(ids)):
            select = df.loc[ids[i],['ID_centerline', 'Xspline', 'Yspline', 'Zspline', 's', 'Radius']]
            max_cent = select['ID_centerline'].max()
            
            for j in range(max_cent+1):
                # create the arrays from the elements of sp_df dataframe for each rows
                x = np.array(select[select.ID_centerline == j]['Xspline'])
                y = np.array(select[select.ID_centerline == j]['Yspline'])
                z = np.array(select[select.ID_centerline == j]['Zspline'])
                s = np.array(select[select.ID_centerline == j]['s'])
                r = np.array(select[select.ID_centerline == j]['Radius'])
                
                sp_df2 = sp_df2.append(pd.DataFrame({'ID_patient': ids[i],
                                                    'ID_centerline': j,
                                                    'Xspline': [x],
                                                    'Yspline': [y],
                                                    'Zspline': [z],
                                                    's': [s],
                                                    'Radius': [r]}))
                
        sp_df2.index = sp_df2['ID_patient']
        sp_df2 = sp_df2.drop('ID_patient', axis = 1)

        return sp_df2
            

    def final_labels(self, df, ids):
        final_label = []
        review = []
        to_drop = []
        for i in ids:
            # select ith patient
            select = df[df.index == i]
            # create a list with the associated label's centerlines
            cent_list = select['Label'].tolist()
            # count all the occurences of the labels
            occurences = dict([(k, cent_list.count(k)) for k in cent_list])
            # determine the label which is mostly present for that patient
            max_value = max(occurences.values())
            # return the keys associated to the maximum value
            max_keys = [key for key, value in occurences.items() if value == max_value]
            
            # if the maximum is unique, append it to the final list
            if len(max_keys) == 1:
                final_label.append(max_keys)
            
            # else, raise an error
            else:
                review.append(occurences)
                to_drop.append(i)
                print(f'For patient {i} there is not a unique associated label.')
        
        return final_label, review, to_drop
