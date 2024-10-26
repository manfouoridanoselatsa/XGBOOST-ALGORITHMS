import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import category_encoders as ce
from rich import print
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

dataWIDS = pd.read_csv('Datasets/TrainingWiDS2021.csv')


#(100*(dataWIDS.isnull().sum())/(dataWIDS.shape[0]))
#print(dataWIDS.isna().sum())
max_missing_percent = 30
#missing_percent = dataWIDS.isna().mean() * 100


def drop_columns_with_nan(df, threshold):
    nan_percent = (100*(dataWIDS.isnull().sum())/(dataWIDS.shape[0]))
    columns_to_drop = nan_percent[nan_percent >= threshold].index
    return df.drop(columns=columns_to_drop)
  

def analyze_label_distribution_percent(df, label_column):
    # Calculate the distribution in percentages
    label_percentages = df[label_column].value_counts(normalize=True) * 100

    # Display numerical results
    print("Label Distribution (Percentage):")
    for label, percentage in label_percentages.items():
        print(f"{label}: {percentage:.2f}%")

    # Visualize the distribution
    #plt.figure(figsize=(10, 6))
    #sns.barplot(x=label_percentages.index, y=label_percentages.values)
    #plt.title("Label Distribution (Percentage)")
    #plt.xlabel("Labels")
    #plt.ylabel("Percentage")
    #plt.ylim(0, 100)  # Set y-axis limit to 100%
    #plt.xticks(rotation=45)
    #for i, v in enumerate(label_percentages.values):
    #    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    #plt.tight_layout()
    #plt.show()

    # Pie chart
    #plt.figure(figsize=(8, 8))
    #plt.pie(label_percentages.values, labels=label_percentages.index, autopct='%1.1f%%', startangle=90)
    #plt.title("Label Distribution (Pie Chart)")
    #plt.axis('equal')
    #plt.tight_layout()
    #plt.show()

def impute_nan_values(df):
    # Make a copy of the dataframe to avoid modifying the original
    df_imputed = df.copy()
    
    for column in df_imputed.columns:
        if df_imputed[column].dtype in ['int64', 'float64']:
            # For numerical columns, impute with median
            median_value = df_imputed[column].median()
            df_imputed[column].fillna(median_value, inplace=True)
        else:
            # For categorical columns, impute with mode
            mode_value = df_imputed[column].mode()[0]  # [0] because mode can return multiple values
            df_imputed[column].fillna(mode_value, inplace=True)
    
    return df_imputed


def standardize_dataset(df, label_column):
    # Create a new DataFrame to store the standardized data
    df_standardized = df.copy()

    # Identify numeric columns, excluding the label column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop(label_column) if label_column in numeric_columns else numeric_columns

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the numeric columns
    df_standardized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df_standardized

def oversample_data(X, y, random_state=42):
    # Initialize SMOTE
    smote = SMOTE(random_state=random_state)

    # Fit and apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def process_and_oversample(df, label_column, test_size=0.2, random_state=42):
    # Separate features and label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Apply oversampling to the training data
    X_train_resampled, y_train_resampled = oversample_data(X_train,y_train, random_state=random_state)

    # Print class distribution before and after oversampling
    print("Original class distribution:", Counter(y_train))
    print("Resampled class distribution:", Counter(y_train_resampled))
    X_train_resampled = pd.concat([X_train_resampled, X_test], axis=0)
    y_train_resampled = pd.concat([y_train_resampled, y_test], axis=0)
    
    return X_train_resampled, y_train_resampled

dataWIDS = drop_columns_with_nan(dataWIDS, max_missing_percent)
analyze_label_distribution_percent(dataWIDS, 'diabetes_mellitus')
dataWIDS = impute_nan_values(dataWIDS)

#on supprime les colonnes
dataWIDS.drop(dataWIDS.columns[0:3], axis=1, inplace=True)
dataWIDS.drop(columns=['icu_id', 'icu_type'], inplace=True)

#dataWIDS = standardize_dataset(dataWIDS, 'diabetes_mellitus')
encoder = ce.TargetEncoder(cols=['gender','ethnicity', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type'])
encoder.fit(dataWIDS, dataWIDS['diabetes_mellitus'])
dataWIDS = encoder.transform(dataWIDS)

#print(dataWIDS.isna().sum())
print(dataWIDS)
#  Save Not Over Sampled Data
# X = dataWIDS.drop(columns=['diabetes_mellitus'])
# y = dataWIDS['diabetes_mellitus']

# X.to_csv("Train_data_noOversampling_c_100k.csv", index=False, header=False)
# y.to_csv("Train_label_noOversampling_c_100k.csv", index=False, header=False)

# Ici on fait de l'oversampling
X_train, y_train = process_and_oversample(dataWIDS,'diabetes_mellitus')

X_train.to_csv("Train_data_Oversampled_c_100k.csv", index=False, header=False)
y_train.to_csv("Train_label_Oversampled_c_100k.csv", index=False, header=False)
print(X_train)
# Ici on met en csv
#train_data = X_train
#test_data = X_test.iloc[:2000].copy()
#y_train = y_train
#y_test = y_test.iloc[:2000].copy()
#train_data_df = pd.concat([y_train, train_data], axis=1)
#test_data_df = pd.concat([y_test, test_data], axis=1)
#train_data.to_csv("Train_data_c_100k.csv", index=False, header=False)
#test_data.to_csv("Test_data_c_50k.csv", index=False, header=False)
#y_train.to_csv("Train_labels_c_100k.csv", index=False, header=False)
#y_test.to_csv("Test_labels_c_50k.csv", index=False, header=False)
#test_data['diabetes_mellitus'] = y_test
#test_data.to_csv("Test_data.csv", index=False)






#df_cleaned = df_cleaned.drop(df_cleaned.columns[99:106], axis=1)
#df_cleaned = df_cleaned.drop(columns=['icu_id', 'icu_type'])
#df_cleaned = df_cleaned.drop(df_cleaned.columns[0:3], axis=1)
#dataWIDSCopy = dataWIDS.copy()

#max_missing_percent = 40

# Calculez le pourcentage de valeurs manquantes ou NaN pour chaque colonne
#missing_percent = df_Copy.isna().mean() * 100

# Supprimez les colonnes ayant trop de valeurs manquantes ou NaN
#df_cleaned = df_Copy.loc[:, missing_percent < max_missing_percent]
#df_cleaned= df_cleaned.dropna(axis=0, how='any')
#df_cleaned = df_cleaned.drop(df_cleaned.columns[99:106], axis=1)
#df_cleaned = df_cleaned.drop(columns=['icu_id', 'icu_type'])
#df_cleaned = df_cleaned.drop(df_cleaned.columns[0:3], axis=1)
#encoder = OneHotEncoder(sparse_output=False)
#df_cleaned['gender'] = encoder.fit_transform(df_cleaned[['gender']])

#encoder = ce.TargetEncoder(cols=['ethnicity', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type'])

# Fit the encoder on the entire DataFrame
#encoder.fit(df_cleaned, df_cleaned['diabetes_mellitus'])
#df_encoded = encoder.transform(df_cleaned)


#df_cleaned['ethnicity'] = df_encoded['ethnicity']
#df_cleaned['hospital_admit_source'] = df_encoded['hospital_admit_source']
#df_cleaned['icu_admit_source'] = df_encoded['icu_admit_source']
#df_cleaned['icu_stay_type'] = df_encoded['icu_stay_type']

#dataWIDS.to_csv('../Datasets/TrainingWiDS2021.csv',index=False)
