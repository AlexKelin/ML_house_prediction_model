import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file = 'melb_data.csv'
file_open = pd.read_csv(file)
pd.set_option('display.max_columns', 100)
# print a summary of the data in Melbourne data
# print(file_open.describe())

# check the column names
# print(file_open.columns)
print(file_open.head())
# find if you have any information missing from data set
# print(file_open.isnull().sum())

# NEXT

# dropna drops missing values (think of na as "not available")
file_open = file_open.dropna(axis=0)

# Now we view the parameters that are in the area of interest.
# We cannot include non-numerical parameterrs to make predictions, like "type" and "date"
features = ['Rooms', 'Landsize', 'YearBuilt']
X = file_open[features]
# What we are predicting is price (usually called y), we can call the catiables by dot notation
y = file_open.Price

print(X.head())
print('*' * 20)
print(X.describe())
print('*' * 20)

# You can pull out a variable with dot-notation
# print(file_open.Price)

# NEXT

# Now we need to build the model itself with scikit learn

# from sklearn.tree import DecisionTreeRegressor

# Define model and Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# predictions for the first 5 houses; because the default for head is 5
predictions = melbourne_model.predict(X.head())
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(predictions)
print("*" * 30)
# predictions for the first 10 houses
predictions = melbourne_model.predict(X.head(10))
print("Making predictions for the following 10 houses:")
print(X.head(10))
print("The predictions are")
print(predictions)
print("*" * 30)
# predictions for houses 10 to 20
predictions = melbourne_model.predict(X.iloc[10:20])
print("Making predictions for houses 10 to 20:")
print(X.iloc[10:20])
print("The predictions are")
print(predictions)

# NEXT

# If you want to select rows based on their labels, you should use .loc[] instead of .iloc[]

predictions = melbourne_model.predict(X.loc[10:20])
print("Making predictions for houses 10 to 20 by number in csv:")
print(X.loc[10:20])
print("The predictions are")
print(predictions)

# Now we need to split the data into pieces for training and validating.
# The random_state argument guarantees we get the same split every time we
# Use the train_test_split function to split up your data.the random_state value run this script.
print("*" * 30)
# from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)

# Fit iowa_model with the training data.lets take first 7
train_predictions = melbourne_model.predict(train_X)
print('training observations')
print(train_predictions[0:7])
# Predict with all validation observations
val_predictions = melbourne_model.predict(val_X)
print('Validation observations')
print(val_predictions[0:7])

# print(mean_absolute_error(val_y, val_predictions))

# from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(f'Mean absolute Error: {val_mae}')
print("*" * 30)


# NEXT, we can rewrite this into a function to fine-tune it quickly. Mean Expected Error.
# max _leaf_nodes is how deep the decision tree may be.
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


thelist = {}
# Now lets call out funciton with different amount of leaf nodes.
# Just stored the results to check where we get the best result
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    thelist[max_leaf_nodes] = my_mae
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
# now we are looking for the MINIMUM value of the MAE error
print(thelist)
min_mea = min(thelist.values())
print(f'MEA {min_mea} and the number of nodes is {[k for k, v in thelist.items() if v == min_mea]}')

# NEXT lets fit the model
print("*" * 30)
# Assuming 'optimal_leaf_nodes' is the number of nodes that resulted in the lowest MAE
optimal_leaf_nodes = [k for k, v in thelist.items() if v == min(thelist.values())][0]

# Create the final model with the optimal number of leaf nodes
final_model = DecisionTreeRegressor(max_leaf_nodes=optimal_leaf_nodes, random_state=1)

# Fit the final model using the entire dataset
final_model.fit(X, y)


# predictions for the first 20 houses
new_predictions = final_model.predict(X.iloc[0:20])
print("The new price predictions are")
print(new_predictions)

#Random Forests
# can enhance model accuracy beyond single Decision Trees,
# which may overfit or underfit depending on their depth. Random Forests average predictions across
# multiple trees for improved accuracy and generally perform well with default settings.
# Further modeling advancements can yield better results but often require precise parameter tuning.

# from sklearn.ensemble import RandomForestRegressor
print("*" * 30)
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
rf_val_mae = forest_model.predict(val_X)
print(rf_val_mae)
#And this is the MAE of all RF values
print(mean_absolute_error(val_y, rf_val_mae))

#we got the result of 336,456$(thousands)
#The Mean Absolute Error (MAE) of 336,456 from your.
#Random Forest model is moderately high in the context of your dataset's house prices.
#While it indicates that the model's predictions are off by this amount on average,
# the acceptability of this error depends on factors like the average price range,
# comparison with other models (like your Decision Tree model which has a slightly lower MAE),
# and the intended use of the model.

#These are the predicted prices for houses in your validation dataset (val_X).
# Each number is the price predicted by the RandomForestRegressor model for a corresponding house.
#[ 763204.33333333  973177.78        871832.61363636 ... 2346532.
#  757515.         1512600.        ]


#This is the Mean Absolute Error for the predictions made by the RandomForestRegressor.
# It's the average absolute difference between the predicted
# values and the actual values (val_y) in your validation set.
# This number gives you an indication of how much, on average,
# the predictions deviate from the actual house prices.
#336456.6564916975