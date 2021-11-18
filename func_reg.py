def regression_models(X_train, Y_train, X_test, Y_test):
#Linear Regression
lr = LinearRegression()
lr.fit(X_train,Y_train)
y_pred = (lr.predict(X_test))
print("LINEAR REGRESSION\nR-squared train score: ", lr.score(X_train, Y_train))
print("R-squared test score: ", lr.score(X_test, Y_test))
#Lasso Regression
las = Lasso(alpha = 0.0001)
las.fit(X_train, Y_train)
print("\nLASSO REGRESSION\nR-squared train score: ", las.score(X_train, Y_train))
print("R-squared test score: ", las.score(X_test, Y_test))
#Decision Tree
dec_tree = dtr(min_samples_leaf = 25)
dec_tree.fit(X_train, Y_train)
print("\nDECISION TREE\nR-squared train score: ", dec_tree.score(X_train, Y_train))
print("R-squared test score: ", dec_tree.score(X_test, Y_test))
#Random Forest Regressor
rfr = rfc()
rfr.fit(X_train, Y_train)
print("\nRANDOM FOREST REGRESSOR\nR-squared train score: ", rfr.score(X_train, Y_train))
print("R-squared test score: ", rfr.score(X_test, Y_test))
regression_models(X_train, Y_train, X_test, Y_test)
