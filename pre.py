data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
data.describe()
data.drop(['name','id','host_name','last_review'], axis=1, inplace=True)
data['reviews_per_month'].fillna(0, inplace=True)
data.isnull().sum()
map_nyc = np.array(Image.open('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png'))
plt.figure(figsize = (10, 10))
plt.imshow(map_nyc)
plt.axis('off')
plt.ioff()
plt.show()
title = 'Neighbourhood Group Locations'
plt.figure(figsize = (10, 8))
sns.scatterplot(data.longitude, data.latitude, hue = data.neighbourhood_group).set_title(title)
plt.ioff()
plt.figure(figsize = (20, 10))
title = 'Correlation matrix of numerical variables'
sns.heatmap(data.corr(), square = True, cmap = 'Spectral')
plt.title(title)
plt.ioff()
fig = px.scatter_mapbox(data, lat = "latitude", lon = "longitude", color = "neighbourhood", size = "price", size_max = 30, opacity = .70, zoom = 12)
fig.layout.mapbox.style = 'carto-positron'
fig.update_layout(title_text = 'NYC Neighbourhood', height = 750)
fig.show()
data2 = data.loc[(data['price'] < 300) & (data['reviews_per_month'] < 10)]
title = 'Price relation to number of review per month for Properties under $300'
f, ax = plt.subplots(figsize = (8, 6))
sns.scatterplot(x = 'reviews_per_month', y = 'price', data = data2, color = 'red')
plt.title(title)
plt.ioff()

plt.figure(figsize = (18, 8))
sns.barplot(x = data.number_of_reviews[:50], y = data.neighbourhood[:50])
plt.xticks(rotation = 90)
plt.show()
colors = ['orange', 'turquoise', 'silver']
labels = data.room_type.value_counts().index
values = data.room_type.value_counts().values
plt.figure(0, figsize = (7, 7))
plt.pie(values, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title('Airbnb According to Neighbourhood Group')
plt.show()
data3 = data.dropna(subset = ["price"]).groupby("neighbourhood")[["neighbourhood", "price"]].agg("mean").sort_values(by = "price",
                                ascending = False).rename(index = str, columns = {"price": "Average price/night"}).head(10)

data3.plot(kind = 'bar', color = 'gold')
plt.show()

data3 = data.dropna(subset = ["price"]).groupby("neighbourhood")[["neighbourhood", "price"]].agg("mean").sort_values(by = "price",
                                ascending = False).rename(index = str, columns = {"price": "Average price/night"}).tail(10)

data3.plot(kind = 'bar', color = 'purple')
plt.show()
arget_columns = ['neighbourhood_group','room_type','price','minimum_nights','calculated_host_listings_count','availability_365']
data_1 = data[target_columns]

#Below we encode values of the forst column since they are strings and cannot be converted to float for linear regression
data_1['room_type'] = data_1['room_type'].factorize()[0]
data_1['neighbourhood_group'] = data_1['neighbourhood_group'].factorize()[0]
data_1.head()
Y = data_1['price']
X = data_1.drop(['price'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1, random_state = 42)
