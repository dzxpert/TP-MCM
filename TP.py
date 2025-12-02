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
cols_del = ['AcceptedCmp3', 'AcceptedCmp5.1', 'AcceptedCmp5', 'AcceptedCmp4','Complain','Response']
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

scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.fit_transform(ds), columns=ds.columns)
#-------------------------------------------------------
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

#-------------------------------------------------------
# Agglomerative Clustering
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
dataf_clean["Clusters"] = yhat_AC

# Define colormap with proper hex codes
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

# 3D Plot of the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()

# Cluster distribution countplot
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
pl = sns.countplot(x=dataf_clean["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.show()



# Detailed Cluster Analysis
print("="*60)
print("CLUSTER ANALYSIS SUMMARY")
print("="*60)

# Count of customers in each cluster
print("\n1. Cluster Distribution:")
print(dataf_clean["Clusters"].value_counts().sort_index())

# Statistical summary for each cluster
print("\n2. Cluster Characteristics:")
for cluster in sorted(dataf_clean["Clusters"].unique()):
    print(f"\n--- Cluster {cluster} ---")
    cluster_data = dataf_clean[dataf_clean["Clusters"] == cluster]
    
    print(f"Number of customers: {len(cluster_data)}")
    print(f"Average Income: ${cluster_data['Income'].mean():.2f}")
    print(f"Average Age: {cluster_data['Age'].mean():.1f} years")
    print(f"Average Spent: ${cluster_data['Spent'].mean():.2f}")
    print(f"Average Recency: {cluster_data['Recency'].mean():.1f} days")
    print(f"% Parents: {(cluster_data['Is_Parent'].sum() / len(cluster_data) * 100):.1f}%")
    print(f"Average Family Size: {cluster_data['Family_Size'].mean():.2f}")

# Create comparison visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Cluster Comparison Across Key Metrics', fontsize=16, fontweight='bold')

pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]

# Income by Cluster
sns.boxplot(data=dataf_clean, x="Clusters", y="Income", palette=pal, ax=axes[0, 0])
axes[0, 0].set_title('Income Distribution by Cluster')

# Age by Cluster
sns.boxplot(data=dataf_clean, x="Clusters", y="Age", palette=pal, ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Cluster')

# Spent by Cluster
sns.boxplot(data=dataf_clean, x="Clusters", y="Spent", palette=pal, ax=axes[0, 2])
axes[0, 2].set_title('Total Spending by Cluster')

# Recency by Cluster
sns.boxplot(data=dataf_clean, x="Clusters", y="Recency", palette=pal, ax=axes[1, 0])
axes[1, 0].set_title('Recency by Cluster')

# Family Size by Cluster
sns.boxplot(data=dataf_clean, x="Clusters", y="Family_Size", palette=pal, ax=axes[1, 1])
axes[1, 1].set_title('Family Size by Cluster')

# Is_Parent distribution
cluster_parent = dataf_clean.groupby(['Clusters', 'Is_Parent']).size().unstack(fill_value=0)
cluster_parent.plot(kind='bar', stacked=True, ax=axes[1, 2], color=['#B9C0C9', '#682F2F'])
axes[1, 2].set_title('Parent Status by Cluster')
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Count')
axes[1, 2].legend(['Not Parent', 'Parent'])

plt.tight_layout()
plt.show()

# Product preferences by cluster
print("\n3. Product Category Spending by Cluster:")
product_cols = ['Drinks', 'Fruits', 'Vegetables', 'Meat', 'Fish', 'Sweets', 'Electro']
cluster_products = dataf_clean.groupby('Clusters')[product_cols].mean()
print(cluster_products.round(2))

# Visualize product preferences
fig, ax = plt.subplots(figsize=(12, 6))
cluster_products.T.plot(kind='bar', ax=ax, color=pal)
ax.set_title('Average Spending on Product Categories by Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('Average Spending ($)')
ax.legend(title='Cluster', labels=[f'Cluster {i}' for i in range(4)])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)