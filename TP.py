import datetime
import numpy as np
import pandas as pd
import warnings
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

data = pd.read_csv("instance_Ghouari.csv", sep=';')
print("Number of datapoints:", len(data))
#data.info()
data_ = data #backup

for column in data.columns:
    nan_counter = data[column].isna().sum()
    print(column,'=',nan_counter)    
data = data.dropna()

print("after cleanup we got:", len(data))
# print("============================ NAN CLEARED? =============================")
#for column in data.columns:
#    nan_counter = data[column].isna().sum()
#    print(column,'=',nan_counter)

data["Dt_Customer"]=pd.to_datetime(data["Dt_Customer"], dayfirst=True)
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)
print('first order: ',min(dates))
print('last order: ',max(dates))

days = []
dmax = max(dates)
for i in dates:
    delta = dmax - i
    days.append(delta)
data["Customer_xp"] = days
data["Customer_xp"] = pd.to_numeric(data["Customer_xp"],errors="coerce")

print('marital ',data["Marital_Status"].value_counts(),"\n")
# histo_ms = data["Marital_Status"].value_counts()
# histo_ms.plot(kind='bar')
print('education ',data["Education"].value_counts(),"\n")
# histo_ms = data["Education"].value_counts()
# histo_ms.plot(kind='bar', color='red')

#-------------------------------------------------------
data["Year_Birth"] = pd.to_datetime(data["Year_Birth"], format='mixed')
data["YB"] = data["Year_Birth"].dt.year
# data['YB'].plot(kind='hist', bins=30)
data["Age"] = 2025-data['YB']
#-------------------------------------------------------
data["MntElectroProds"] = data["MntElectroProds"]*100
mnt_cols = [col for col in data.columns if col.startswith("Mnt")]
data["Spent"] = data[mnt_cols].sum(axis=1)
#-------------------------------------------------------
data["Living_With"] = data["Marital_Status"].replace({"Married":"Partner","Together":"Partner","Single":"Alone","Divorced":"Alone","Yolo":"Alone","Absurd":"Partner","Widow":"Alone"})
#------------------------------------------------------- 
data["Children"] = data["kidshome"]+data["Teenshome"]
#------------------------------------------------------- 
data["FS"] = data["Living_With"].replace({"Alone":1,"Partner":2})
data["Family_Size"] = data["FS"]+data["Children"]

# Family Size correct it? based on real study of Widow Yolo ect
#------------------------------------------------------- 
data["Is_Parent"] = np.where(data.Children>0,1,0)
#------------------------------------------------------- 
data["Education"] = data["Education"].replace({"Basic":"Undergraduate","2nd Cycle":"Undergraduate","Graduation":"Graduate","Master":"Postgraduate","PhD":"Postgraduate","Phd":"Postgraduate"})
data["Educ"] = data["Education"].replace({"Undergraduate":1,"Graduate":2,"Postgraduate":3})
#------------------------------------------------------- 
data = data.rename(columns={"MntDrinks":"Drinks","MntFruits":"Fruits","MntVegetables":"Vegetables","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntElectroProds":"Electro"})
#------------------------------------------------------- 
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID", "Education", "YB", "Living_With"]
dataf = data.drop(to_drop, axis=1)
#------------------------------------------------------- 
#dataf.describe()
#dataf.describe(include='all')
#-------------------------------------------------------

sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"FFF9ED"})
pallete=["682F2F","9E726F","D6B2B1","B9C0C9","9F8A78","F3AB60"]
cmap = colors.ListedColormap(["682F2F","9E726F","D6B2B1","B9C0C9","9F8A78","F3AB60"])
To_Plot = ["Income","Recency","Customer_xp","Age","Spent","Is_Parent"]
print("Reletive Plot of Some Select Features: a Data Subset")
plt.figure()
sns.pairplot(dataf[To_Plot],hue="Is_Parent",palette=["#682F2F","#F3AB60"])
plt.show()
#-------------------------------------------------------

#IQR method to remove outliers
#1 Calculating Q1 (25th percentile) and Q3 (75th percentile)
#2 Calculating Q1 (25th percentile) and Q3 (75th percentile)
#3 Computing IQR = Q3 - Q1
#4 Defining outliers as values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using IQR method
    multiplier: typically 1.5 for outliers, 3.0 for extreme outliers
    """
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filter out outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
    return df_clean

# Columns to check for outliers
numeric_cols = ["Income", "Recency", "Customer_xp", "Age", "Spent"]

print(f"Before removing outliers: {len(dataf)} rows")

# Remove outliers
dataf_clean = remove_outliers_iqr(dataf, numeric_cols, multiplier=1.5)

print(f"After removing outliers: {len(dataf_clean)} rows")
print(f"Removed {len(dataf) - len(dataf_clean)} outlier rows")

# Now plot with cleaned data
To_Plot = ["Income", "Recency", "Customer_xp", "Age", "Spent", "Is_Parent"]
print("\nRelative Plot of Some Select Features: a Data Subset (Outliers Removed)")
plt.figure()
sns.pairplot(dataf_clean[To_Plot], hue="Is_Parent", palette=["#682F2F", "#F3AB60"])
plt.show()

#-------------------------------------------------------

# Normalization / Scaling
ds = dataf_clean.copy()

# Only drop columns that actually exist
cols_del = ['AcceptedCmp3', 'AcceptedCmp5.1', 'AcceptedCmp5', 'AcceptedCmp4']
cols_to_drop = [col for col in cols_del if col in ds.columns]
if cols_to_drop:
    ds = ds.drop(cols_to_drop, axis=1)
    print(f"Dropped columns: {cols_to_drop}")
else:
    print("No campaign columns found to drop")

scaler = StandardScaler()

# The standard score of a sample x is calculated as:
# z = (x - u) / s
# where u is the mean of the training samples
# s = std.fit(ds)

scaled_ds = pd.DataFrame(scaler.fit_transform(ds), columns=ds.columns)

# PCA
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])
PCA_ds.describe().T

x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]

# To plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()