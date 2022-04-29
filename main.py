import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA



df = pd.read_csv('VideoGamesSales.csv')
df = df.dropna()
print(df.columns)

features = ['Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

x_val = df.loc[:, features].values


x_val = StandardScaler().fit_transform(x_val)


pca = PCA(n_components=7)
principalComponents = pca.fit_transform(x_val)

#print(principalComponents)
#-------------------------------------------------
#tworzenie kolka xd
x=np.linspace(start=-1,stop=1,num=500)
y_positive=lambda x: np.sqrt(1-x**2) 
y_negative=lambda x: -np.sqrt(1-x**2)
plt.plot(x,list(map(y_positive, x)), color='maroon')
plt.plot(x,list(map(y_negative, x)),color='maroon')

x=np.linspace(start=-1,stop=1,num=30)
plt.scatter(x,[0]*len(x), marker='_',color='maroon')
plt.scatter([0]*len(x), x, marker='|',color='maroon')
#plt.show()
#----------------------------------------------------


#---------------------------------------------------
#Tworzenie mapy PCA variable

pca_values = pca.components_
print(pca.components_)


colors = ['blue', 'red', 'green', 'black', 'purple', 'brown', 'pink']
if len(pca_values[0]) > 7:
    colors=colors*(int(len(pca_values[0])/7)+1)

columns = features

add_string=""
for i in range(len(pca_values[0])):
    xi=pca_values[0][i]
    yi=pca_values[1][i]
    plt.arrow(0,0, 
              dx=xi, dy=yi, 
              head_width=0.03, head_length=0.03, 
              color=colors[i], length_includes_head=True)
    add_string=f" ({round(xi,2)} {round(yi,2)})"
    plt.text(pca_values[0, i], 
             pca_values[1, i] , 
             s=columns[i] + add_string )


plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0]*100,2)}%)")
plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1]*100,2)}%)")
plt.title('Variable factor map (PCA)')
plt.show()
#------------------------------------------------------------------------
