import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
  """
  A function for loading two CSV files and merge them on 'id' column.
  Inputs:
    messages_filepath: String, filepath for the first CSV file
    categories_filepath: String, filepath for the second CSV file
  Output:
    df: Pandas Dataframe containing the merged content from both CSV inputs.
  """
  messages = pd.read_csv(messages_filepath)

  categories = pd.read_csv(categories_filepath)

  df = messages.merge(categories, on = 'id')

  return df


def clean_data(df):
  """
  A function for quick data cleaning. It:
  - Expands the categories into multiple columns
  - Renames the columns
  - Transforms columns values into numeric representation
  - Concatenates it and drops duplicates
  Inputs:
    df: Pandas Dataframe, contains merged 'Messages' and 'Categories' files
  Output:
    df: Pandas Dataframe, containing cleaned df.
  """
  categories = pd.DataFrame(df.categories.str.split(";", expand=True))

  # select the first row of the categories dataframe
  row = categories.iloc[:1]

  # use this row to extract a list of new column names for categories
  category_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis =0)

  # rename the columns of `categories`
  categories.columns = category_colnames

  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])

  # drop the original categories column from `df`
  df.drop(columns='categories', inplace=True)

  # reseting indexes
  categories.reset_index(drop=True, inplace=True)
  df.reset_index(drop=True, inplace=True)

  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)

  # drop duplicates
  df.drop_duplicates(inplace = True)

  return df



def drop_table(table_name, database_filepath):
  """
  A function to check if a table exists on a database. If it does, table is dropped.
  Inputs:
    table_name: String, name of the table.
    database_filepath: String, path to the Database to be checked.
  Output:
    None
  """
  conn = sqlite3.connect(database_filepath)
  c = conn.cursor()

  # dropping table when it exists
  try:
    c.execute(f"DROP TABLE IF EXISTS {table_name}")
    print(f"    TABLE: 'MessagesCategories' dropped on {database_filepath}")
    conn.commit()
  except:
    None

  conn.close()
  return None


def save_data(df, database_filepath):
  """
  A function to save a Pandas Dataframe into a database.
  Inputs:
    df: Pandas Dataframe, cleaned and ready to be saved.
    database_filepath: String, path to the Database where df will be saved.
  Output:
    None
  """
  conn = sqlite3.connect(database_filepath)
  df.to_sql('MessagesCategories', conn, index=False)
  return None  


def main():
  """
  Function that loads the data, cleans it and saves it toa database.
  Input:
    None
  Output:
    None
  """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Attempting to drop table if it exists')
        drop_table('MessagesCategories', database_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()