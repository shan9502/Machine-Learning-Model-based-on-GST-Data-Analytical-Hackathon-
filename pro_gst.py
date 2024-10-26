import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import  SMOTE, ADASYN
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve,mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

x_train = pd.read_csv('/Train_data/X_Train_Data_Input.csv').drop(['Column5','Column9','Column14'],axis=1)
y_train = pd.read_csv('/Train_data/Y_Train_Data_Target.csv')
x_test = pd.read_csv('/Test_data/X_Test_Data_Input.csv').drop(['Column5','Column9','Column14'],axis=1)
y_test = pd.read_csv('/Test_data/Y_Test_Data_Target.csv')

train_df = pd.merge(x_train, y_train, on='ID').drop('ID',axis=1)

x_train.drop('ID',axis=1, inplace=True)
y_train.drop('ID',axis=1,inplace=True)
x_test.drop('ID',axis=1, inplace=True)
y_test.drop('ID',axis=1,inplace=True)

train_df['Column6'] = train_df['Column6'].fillna(train_df['Column6'].interpolate())
train_df['Column8'] = train_df['Column8'].fillna(train_df['Column8'].interpolate())
train_df['Column15'] = train_df['Column15'].fillna(train_df['Column15'].interpolate())
# train_df['Column3'] = train_df['Column3'].fillna(train_df['Column3'].interpolate())
# train_df['Column4'] = train_df['Column4'].fillna(train_df['Column4'].interpolate())
train_df['Column0'] = train_df['Column0'].fillna(train_df['Column0'].interpolate())
xtrain = train_df.drop('target',axis=1)
ytrain = train_df['target']

train_df.isna().sum()

def custom_imput(df,target_column,algorithm='lr'):
  # Separate samples with missing target values
  df_missing = df[df[target_column].isnull()]
  df_complete = df[df[target_column].notnull()]
  if True in df_complete.isna().any().to_list():
    print("Found Nan")
    print(df_complete.isna().sum())
    return print("Remove Nan from the training dataset")
    # df_complete = df_complete.interpolate()

  # Prepare features and target
  feature_X = df_complete.drop(columns=[target_column])
  feature_y = df_complete[target_column]

  # Split the data
  feature_X_train, feature_X_test, feature_y_train, feature_y_test = train_test_split(feature_X, feature_y, test_size=0.2, random_state=42)

  # Prepare the data with missing target values
  X_missing = df_missing.drop(columns=[target_column])
  X_missing = X_missing.interpolate()

  if algorithm == 'lr':
      feature_model = LinearRegression()
  elif algorithm == 'rf':
      feature_model = RandomForestRegressor()
  elif algorithm == 'knn':
      feature_model = KNeighborsRegressor()
  else:
      raise ValueError("Invalid algorithm specified")

  feature_model.fit(feature_X_train, feature_y_train)
  feature_y_pred = feature_model.predict(feature_X_test)
  performance = mean_squared_error(feature_y_test, feature_y_pred, squared=False)
  print(f"Model RMSE: {performance}")

  # Predict missing values
  missing_pred = feature_model.predict(X_missing)

  # Update the original dataframe with predicted values
  df.loc[df[target_column].isnull(), target_column] = missing_pred

  return df

test_df = train_df.copy()
# test_df.drop(['Column3'], axis=1, inplace= True)
test_df.drop(['target'], axis=1, inplace= True)

col3 = custom_imput(test_df.drop('Column4',axis=1),'Column3','lr')['Column3']
test_df['Column3'] = col3
col4 = custom_imput(test_df,'Column4','lr')['Column4']
test_df['Column4'] = col4
test_df['Column3'] = train_df['Column3']
test_df['Column3'] = custom_imput(test_df,'Column3','lr')['Column3']

# # test_df['Column3'] = col3
imputed_df = test_df

def display_class_distribution(y):
    """
    Display the distribution of classes in the target variable.

    Parameters:
    y (array-like): The target variable
    """
    print("Class distribution:")
    print(pd.Series(y).value_counts(normalize=True))

def random_oversampling(X, y):
    """
    Perform random oversampling on the minority class.

    Parameters:
    X (array-like): The feature matrix
    y (array-like): The target variable

    Returns:
    tuple: X_resampled, y_resampled
    """
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)

    # Separate majority and minority classes
    df_majority = X_df[y_df == y_df.value_counts().index[0]]
    df_minority = X_df[y_df == y_df.value_counts().index[-1]]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled.iloc[:, :-1].values, df_upsampled.iloc[:, -1].values

def random_undersampling(X, y):
    """
    Perform random undersampling on the majority class.

    Parameters:
    X (array-like): The feature matrix
    y (array-like): The target variable

    Returns:
    tuple: X_resampled, y_resampled
    """
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X, y)
    return X_under, y_under

def smote_oversampling(X, y):
    """
    Perform SMOTE oversampling.

    Parameters:
    X (array-like): The feature matrix
    y (array-like): The target variable

    Returns:
    tuple: X_resampled, y_resampled
    """
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    X_smote = pd.DataFrame(X_smote, columns=X.columns)
    y_smote = pd.Series(y_smote, name=y.name)
    return X_smote, y_smote

def adasyn_oversampling(X, y):
    """
    Perform ADASYN oversampling.

    Parameters:
    X (array-like): The feature matrix
    y (array-like): The target variable

    Returns:
    tuple: X_resampled, y_resampled
    """
    adasyn = ADASYN()
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    return X_adasyn, y_adasyn

def smote_tomek_sampling(X, y):
    """
    Perform combined over- and under-sampling using SMOTE and Tomek links.

    Parameters:
    X (array-like): The feature matrix
    y (array-like): The target variable

    Returns:
    tuple: X_resampled, y_resampled
    """
    smt = SMOTETomek()
    X_smt, y_smt = smt.fit_resample(X, y)
    return X_smt, y_smt

xtrain = imputed_df
ytrain = train_df['target']
# imputed_df['target'] = ytrain
# df = imputed_df

# Example usage:
# X, y = load_your_data()
display_class_distribution(ytrain)
#
# # Choose one of the following methods:
# X_resampled, y_resampled = random_oversampling(xtrain, ytrain)
# # OR
# X_resampled, y_resampled = random_undersampling(xtrain, ytrain)
# # OR
X_resampled, y_resampled = smote_oversampling(xtrain, ytrain)
# # OR
# X_resampled, y_resampled = adasyn_oversampling(xtrain, ytrain)
# # OR
# X_resampled, y_resampled = smote_tomek_sampling(xtrain, ytrain)
#
print("Original dataset shape:", xtrain.shape)
print("Resampled dataset shape:", X_resampled.shape)
display_class_distribution(y_resampled)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False
)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(x_test)
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred)
print(f"\nAUC-ROC: {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))