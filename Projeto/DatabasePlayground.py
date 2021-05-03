import matplotlib.pyplot as plt
import seaborn as sns

from StarsDataset import DatasetPanda

dataset = DatasetPanda("")

features = ["Temperature","L","R","A_M"]

corr_spearman = dataset.original[features].corr(method="spearman")
corr_pearson = dataset.original[features].corr(method="pearson")

figure = plt.figure(figsize=(10,8))

sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.title("Pearson Correlation")
plt.show()

sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.title("Spearman Correlation")
plt.show()


# figure = plt.figure(figsize=(20,8))
# sns.boxplot(x="Temperature", y="Type", data=dataset.data)
# plt.show()

# figure = plt.figure(figsize=(20,8))
# sns.boxplot(x="L" ,y="Type",data=dataset.data)
# plt.show()

# figure = plt.figure(figsize=(20,8))
# sns.boxplot(x="R", y="Type",data=dataset.data)
# plt.show()

# figure = plt.figure(figsize=(20,8))
# sns.boxplot(x="A_M", y="Type",data=dataset.data)
# plt.show()

# figure = plt.figure(figsize=(20,8))
# sns.boxplot(x="Color", y="Type",data=dataset.uncategorized)
# plt.show()

fig, ax = plt.subplots(3,2, figsize=(20, 16))
ax = ax.flatten()
i = 0
for col in dataset.uncategorized.columns[:-1]:
    sns.boxplot(x="Type", y=col, ax = ax[i], data=dataset.uncategorized)
    ax[i].legend([col], loc='best')
    i += 1
plt.tight_layout()
plt.show()

# sns.pairplot(dataset.data, 
#              kind='reg',
#              plot_kws={
#                  'line_kws':{'color':'red'}, 
#                  'scatter_kws': {'alpha': 0.3}})
# plt.show()

# col = "Color"
# plt.figure(figsize=(10,4))
# dataset.data[col].value_counts().plot(kind='bar')
# plt.title(col)
# plt.grid()
# plt.show()
# plt.figure(figsize=(10,4))
# dataset.original[col].value_counts().plot(kind='bar')
# plt.title(col)
# plt.grid()
# plt.show()

encode = LabelEncoder()
print(dataset.data["Color"].value_counts())
print("----"*30)
print(dataset.original["Color"].value_counts())
