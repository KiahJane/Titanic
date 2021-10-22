import pandas as pd
import numpy as np
import os

def read_data():
    # set the path to raw data
    train_path = os.path.join(os.path.pardir, 'data', 'raw', 'train.csv')
    test_path = os.path.join(os.path.pardir, 'data', 'raw', 'test.csv')
    # read data w/ all default parameters
    train_df = pd.read_csv(train_path, index_col = 'PassengerId')
    test_df = pd.read_csv(test_path, index_col = 'PassengerId')
    # concate test_df to train_df
    test_df['Survived'] = -888 
    df = pd.concat((train_df, test_df), axis=0)
    
    return df

def process_data(df):
    
    return (df
            # create title feature
            .assign(Title = lambda x: x.Name.map(GetTitle))
            # fill in missing values
            .pipe(fill_missing_values)
            # create fare bin feature
            .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))
            # create age state feature
            .assign(AgeState = lambda x: np.where(x['Age'] >= 18, 'Adult', 'Child'))
            # create family size feature
            .assign(FamilySize = lambda x: x.Parch + x.SibSp + 1)
            # create mother feature
            .assign(IsMother = lambda x: np.where(((x.Sex == 'female') & (x.Age > 18) & (x.Parch > 0) & (x.Title != 'Miss')), 1, 0))
            # create cabin feature
            .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
            # create deck feature
            .assign(Deck = lambda x: x.Cabin.map(get_deck))
            # feature encoding
            .assign(IsMale = lambda x: np.where(x.Sex == 'male', 1, 0))
            .pipe(pd.get_dummies, columns = ['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
            # drop unnecessary columns
            .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
            # reorder columns
            .pipe(reorder_columns)
            )


def GetTitle(name):
    title_group = {'mr':'Mr',            
                   'mrs':'Mrs',
                   'miss':'Miss',
                   'master':'Master',
                   'don':'Sir',
                   'rev':'Sir',
                   'dr':'Officer',
                   'ms':'Mrs',
                   'mme':'Mrs',
                   'major':'Officer',
                   'lady':'Lady',
                   'sir':'Sir',
                   'mlle':'Miss',
                   'col':'Officer',
                   'capt':'Officer',
                   'the countess':'Lady',
                   'jonkheer':'Sir',
                   'dona':'Lady'}
    first_name_with_title = name.split(',')[1]      
    title = first_name_with_title.split('.')[0]     
    title = title.strip().lower()                   
    return(title_group[title])

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def fill_missing_values(df):
    # embarked feature
    df.Embarked.fillna('C', inplace = True)
    # fare feature
    median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median()
    df.Fare.fillna(median_fare, inplace = True)
    # age feature
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace = True)
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def write_data(df):
    # where to save
    train_path = os.path.join(os.path.pardir, 'data', 'raw', 'train.csv')
    test_path = os.path.join(os.path.pardir, 'data', 'raw', 'test.csv')
    # train data
    df.loc[df.Survived != -888].to_csv(write_train_path)
    # test data
    columns = [column for column in df.columns if column != 'Survived']  
    df.loc[df.Survived == -888, columns].to_csv(write_test_path)

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
