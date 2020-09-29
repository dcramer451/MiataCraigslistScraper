#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:25:41 2020

@author: dcramer451
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import sqlite3
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor

#Gather City URLs from Craigslist

initial_scrape_source = 'https://www.craigslist.org/about/sites#US'

cities_code = []
code_get = requests.get(initial_scrape_source)
code_soup = BeautifulSoup(code_get.text, features = 'html.parser')
for i in code_soup.find_all('a'):
    cities_code.append(i.get('href'))
    
  
    
#Since the above loop gathers URLs for the entire world we need to limit our list to only the United States. 

us_city_links = cities_code[8:424]

#Next, we need to adjust the URLs for used car specific results. In Craigslist, these are formatted as https://(city).craigslist.org/search/cto(page listing in intervals of 120)

#The below code takes the city codes we generated above and formats them into the used car URL format for every listing in the United States.  The results is the Combined Links Code list. We will later use this list to scrape the every individual posting. 

page_codes = list(range(120, 3000, 120))

string_page_codes =['miata', 'mx-5', 'mx5']

pages = []
links = []
combined_link_codes = []

for link in us_city_links:
    new_list = link + 'search/cto?query='
    pages.append(new_list)

  
combined_list = [(p,c) for p in pages for c in string_page_codes]

list_df = pd.DataFrame(combined_list)

list_df['combined_column'] = list_df[0] + list_df[1]

combined_links_codes = list_df['combined_column'].tolist()



#Now we will scrape the aggregated listings pages and pull some information we are interested in. 
#From these pages we can gather the following: 1) The posting's title, 2) The Price, 3) When the posting occured, 4) The unique ID assigned by Craigslist, 5) The URL associated with the inidual posting.  We will use this later to gather more information. 
#Since errors sometimes occur, I have included some exceptions to keep the script going. This appears to be a metadata issue but skipping over them and labeling as 'Null' works. 

url_list = []
price = []
title = []
post_time = []
ID = []
features = []
title2list = []
post_text = []


for page in combined_links_codes:
    listings_p1 = requests.get(page)
    soup = BeautifulSoup(listings_p1.text, features = 'html.parser')
    listings = soup.find_all('h2')
    for i in listings:
        try:
            link_find = i.find('a', class_ = 'result-title hdrlnk')
            link_search = link_find['href']
        except TypeError:
            link_search = 'null'
        url_list.append(link_search)


print('Finished First Scan')
#After gathering our initial information, let's pair it down to only include Miatas since that is what we are studying. 
#To do this, I'm going to put the informaton into a dataframe and filter it using the 'in' function. 



df = pd.DataFrame(url_list, columns = ['URL'])
df.drop_duplicates(subset = ['URL'], inplace = True)








#We've now isolated every Miata for sales in the United States.  What comes next is pulling data from the individual postings we have isolated.  This can give us more in depth information such as title status, transmission type, etc. 
#To do this, I'm going to convert the filtered URLs to a list and loop those through similar to what I did above.  I'll then merge those back to the MiataData Dataframe by zipping the collected features to the URL.

miataurl = df['URL'].tolist()

print('Finished Link Processing')

#Loop to collect Features

for link in miataurl:
    try:
        listings_info = requests.get(link)
        soup = BeautifulSoup(listings_info.text, features = 'html.parser')
    except IndexError: 
        listing_info = 'null'  
    try: 
        feature_list = soup.find_all('p', class_ = 'attrgroup')[1]
        feature_list = feature_list.get_text()
    except IndexError:
        feature_list = 'null'
    features.append(feature_list)
    try: 
        title2 = soup.find_all('p', class_ = 'attrgroup')[0]
        title2 = title2.find('span')
        title2 = title2.get_text()
    except IndexError:
        feature_list = 'null'
    title2list.append(title2)
    try:
        posting = soup.find('section', id = 'postingbody')         
        posting = posting.get_text()
    except AttributeError:
        posting = 'null'
    post_text.append(posting)
    try :
        price_search = soup.find('span', class_ = 'price')
        price_search = price_search.get_text()
    except AttributeError:
        price_search = 'null'
    price.append(price_search)      
    try:
        time_find = soup.find('time', class_ = 'date timeago')
        time_search = time_find['datetime']
    except TypeError:
        time_search = 'null'
    post_time.append(time_search)
    try: 
        title_list = soup.find('span', id = 'titletextonly')
        title_list = title_list.get_text()
    except AttributeError:
        title_list = 'null'
    title.append(title_list)    
   
print('Finished Second Scan')        

zipparams= zip(miataurl, title2list, price, post_time, post_text, features)

MiataData = pd.DataFrame(zipparams, columns = (['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Features']))

MiataData.drop_duplicates(subset = 'URL', keep = 'first', inplace = True)

def find_miata(x):
    if 'miata' in x:
        return 1
    elif 'mx-5' in x:
        return 1
    elif 'mx5' in x:
        return 1
    else:
        return 0

MiataData['Is_Miata_Title'] = MiataData.Title.apply(lambda x: find_miata(x.lower()))
MiataData['Is_Miata_URL'] = MiataData.URL.apply(lambda x: find_miata(x.lower()))
MiataData['Sum_Test'] = MiataData['Is_Miata_Title'] + MiataData['Is_Miata_URL']

def add_two(x):
    if x >= 1:
        return True
    else:
        return False
    
MiataData['Is_Miata'] = MiataData['Sum_Test'].apply(lambda x: add_two(x))
print(MiataData)



MiataData = MiataData[MiataData['Is_Miata'] == True]
MiataData = MiataData[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Features']]

#Next, let's get the year from our Titles we extracted.  I'll start with the first Title and then try the second if it is not provided. 

MiataData['Year'] = MiataData.Title.str.extract('(^\d*)')
MiataData['Year'].replace('', np.nan, inplace=True)


MiataData.reset_index(inplace=True)

MiataData = MiataData.dropna(subset = ['Year'])
MiataData['Year']  = MiataData['Year'].astype(int)

def year_format(x):
    if 0 <= x <= 19:
        return '200' + str(x)
    elif 20 <= x <= 99:
        return '19' + str(x)
    else:
        return x

MiataData['Year'] = MiataData.Year.apply(lambda x: year_format(x))

#Now that the Year is taken care of let's find out where these Miatas are located. 
#I'm going to pull the data from the URL string and join it to a spreadsheet I found of all craigslist URL codes associated to their locations. 


city_lookup = pd.read_excel('/Users/dcramer451/Desktop/Script/citydecoder.xlsx')  
URL_split = MiataData['URL'].str.split('/', expand = True)
join_back = MiataData.merge(URL_split, how = 'left', left_index = True, right_index = True)
MiataData = join_back.merge(city_lookup, how = 'left', right_on = 'URL_string', left_on = 2)

MiataData = MiataData[[ 'URL', 'Title','Price','Post_Time', 'Post_Text', 'Features','Year', 'Location', 'State']]



#Let's split the time from the date next

date_time = MiataData['Post_Time'].str.split(" ", expand = True)
date_time = date_time.apply(lambda x: x.str.strip())
df = df.merge(date_time, left_index = True, right_index = True, how = 'left')       
df = df.rename(columns= {0 : 'Posted_date', 1 : 'Posted_time'})


#Here comes the tricky part.  Currently, the features are clumped together in one cell with a bunch of newlines. 
#They need to be separated and cleaned before this can happen. 
#To do this, I'm going to write a series of formulas that will split each into it's own column.
#Once I'm done cleaning, I'll rejoin this data back to the MiataData dataframe. 


#this splits the features based the newlines into separate columns. 

features_split = MiataData.Features.str.split('\n', expand=True)


#Here are the conditions I'll use the locate and label each feature. 

def find_condition(x):
    if 'condition' in x:
        return x
    else:
        return 'NA'

def find_fuel(x):
    if 'fuel' in x:
        return x
    else:
        return 'NA'
    
def find_cylinders(x):
    if 'cylinders' in x:
      return x
    else:
        return 'NA'

def find_title(x):
    if 'title' in x:
        return x
    else:
        return 'NA'    

def find_drive(x):
    if 'drive' in x:
        return x
    else:
        return 'NA'   
 
def find_transmission(x):
    if 'transmission' in x:
        return x
    else:
        return 'NA'
    
def find_odometer(x):
    if 'odometer' in x:
        return x
    else:
        return 'NA'

def find_color(x):
    if 'color' in x:
        return x
    else:
        return 'NA'

def find_size(x):
    if 'size' in x:
        return x
    else:
        return 'NA' 
 
def find_type(x):
    if 'type' in x:
        return x
    else:
        return 'NA'

def replace_none(x):
    if x is None:
        return '0' 
    else:
        return x      
    
#Only odd columns have values so I select those, I also convert Nones to NaNs since they are easier to work with.

features_split = features_split[[1,3,5,7,9,11,13,15,17,19,21]]

features_split = features_split.replace('', np.nan)

features_split = features_split.applymap(str)
    
split_list = [1,3,5,7,9,11,13,15,17,19,21]

condition_columns = []
cylinder_columns = []
fuel_columns = []
transmission_columns = []
title_columns = []
odometer_columns = []
paint_columns = []
size_columns = []
type_columns = []

#This itterates through each column and finds when a certain feature is listed using the above conditional formulas.
for i in split_list:
    features_split['condition' + str(i)] = features_split[i].apply(lambda x: find_condition(x))
    features_split['cylinders' + str(i)] = features_split[i].apply(lambda x: find_cylinders(x))
    features_split['fuel' + str(i)] = features_split[i].apply(lambda x: find_drive(x))
    features_split['transmission' + str(i)] = features_split[i].apply(lambda x: find_transmission(x))
    features_split['title' + str(i)] = features_split[i].apply(lambda x: find_title(x))
    features_split['odometer' + str(i)] = features_split[i].apply(lambda x: find_odometer(x))
    features_split['paint' + str(i)] = features_split[i].apply(lambda x: find_color(x))
    features_split['size' + str(i)] = features_split[i].apply(lambda x: find_size(x))
    features_split['type' + str(i)] = features_split[i].apply(lambda x: find_type(x))


#the columns are built as lists using the values pulled from the formulas and loop. 
for i in split_list:
    condition_columns.append('condition' + str(i))
    cylinder_columns.append('cylinders' + str(i))
    fuel_columns.append('fuel' + str(i))
    transmission_columns.append('transmission' + str(i))
    title_columns.append('title' + str(i))
    odometer_columns.append('odometer' + str(i))
    paint_columns.append('paint' + str(i))
    size_columns.append('size' + str(i))
    type_columns.append('type' + str(i))


#Additional cleanup
features_split = features_split.replace('NA',"")

condition = features_split[condition_columns]
cylinders = features_split[cylinder_columns]
fuel = features_split[fuel_columns]
transmission = features_split[transmission_columns]
title_status = features_split[title_columns]
odometer = features_split[odometer_columns]
paint = features_split[paint_columns]
size = features_split[size_columns]
vtype = features_split[type_columns]

condition['Condition'] = condition.sum(axis=1).astype(str)
cylinders['Cylinders'] = cylinders.sum(axis=1).astype(str)
fuel['Fuel'] = fuel.sum(axis=1).astype(str)
transmission['Transmission'] = transmission.sum(axis=1).astype(str)
title_status['Title_status'] = title_status.sum(axis=1).astype(str)
odometer['Odometer'] = odometer.sum(axis=1).astype(str)
paint['Paint'] = paint.sum(axis=1).astype(str)
size['Size'] = size.sum(axis=1).astype(str)
vtype['Type'] = vtype.sum(axis=1).astype(str)

condition = condition[['Condition']]
cylinders = cylinders[['Cylinders']]
fuel = fuel[['Fuel']]
transmission = transmission[['Transmission']]
title_status = title_status[['Title_status']]
odometer = odometer[['Odometer']]
paint = paint[['Paint']]
size = size[['Size']]
vtype = vtype[['Type']]

#Split values based on common ":" delimiter. 

condition = condition.Condition.str.split(":", expand = True)
cylinders = cylinders.Cylinders.str.split(":", expand = True)
fuel = fuel.Fuel.str.split(":", expand = True)
transmission = transmission.Transmission.str.split(":", expand = True)
title_status = title_status.Title_status.str.split(":", expand = True)
odometer = odometer.Odometer.str.split(":", expand = True)
paint = paint.Paint.str.split(":", expand = True)
size = size.Size.str.split(":", expand = True)
vtype = vtype.Type.str.split(":", expand = True)

condition = condition[[1]]
cylinders = cylinders[[1]]
fuel = fuel[[1]]
transmission = transmission[[1]]
title_status = title_status[[1]]
odometer = odometer[[1]]
paint = paint[[1]]
size = size[[1]]
vtype = vtype[[1]]


#Join them all together. 

features_merge = condition.merge(cylinders, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(fuel, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(transmission, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(title_status, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(odometer, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(paint, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(size, how = 'left', left_index = True, right_index = True)
features_merge = features_merge.merge(vtype, how = 'left', left_index = True, right_index = True)

features_merge.columns = ['Condition', 'Cylinders', 'Fuel', 'Transmission', 'Title_status', 'Odometer', 'Paint', 'Size', 'Vehical_type']

#Additional cleanup    

features_merge = features_merge.apply(lambda x: x.str.strip())
features_merge = features_merge.apply(lambda x : x.str.title())  

features_merge['Odometer'] = features_merge['Odometer'].apply(lambda x: replace_none(x))
features_merge['Odometer'] = features_merge['Odometer'].astype(int)

features_merge['Cylinders'] = features_merge.Cylinders.str.extract('(^\d*)')


#Join back to main dataset.

df = MiataData.merge(features_merge, how = 'left', left_index = True, right_index = True)


#For our purposes, a lot of these features are unneccesary and can be dropped. 

df = df[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year','Location', 'State', 'Condition', 'Transmission', 'Title_status', 'Odometer']]



df['Post_Text'] = df.Post_Text.replace('\n',"", regex = True)

df['Post_Text'] = df.Post_Text.replace('QR Code Link to This Post', "", regex = True)

#Now that we have the posting body by itself, we should try to see if we can flush any data out that might be missing from the features. 
#One thing that comes to mind is the odometer reading.  Using some regex and conditions, we should be able to fill in some missing odometers with this information. 

#Let's first try just extracting some numbers and then filtering them so they don't match model years. 


posting_extract =  df.Post_Text.str.extract('(\d\d\d\d\d)', expand = True)
posting_extract['2'] = df.Post_Text.str.extract('(\d\d\d\d\d\d)', expand = True)
posting_extract['3'] = df.Post_Text.str.extract('(\d\d\d\d)', expand = True)


def find_k(x):
    return re.findall('(\d+)k(i?)', x)

#That got us some results, but not all of the possibilities.  Many owners list this milage as \d\d\dk.  Let's use regex to find those as well. 

posting_extract['4'] = df.Post_Text.apply(find_k)
posting_extract['4'] = posting_extract['4'].str[0]
posting_extract['4'] = posting_extract['4'].str[0]
posting_extract['4'] = posting_extract['4'].fillna(0) 
posting_extract['4'] = posting_extract['4'].astype(int)

def kilo_filter(x):
    if x < 999:
        return x * 1000
    else:
        return 0
    
def kilo_filter2(x):
    if x < 999:
        return x * 1000
    else:
        return x    

posting_extract['4'] = posting_extract['4'].apply(lambda x: kilo_filter(x))    


posting_extract['3'] = posting_extract['3'].fillna(0)
posting_extract['3'] = posting_extract['3'].astype(int)

def extract_condition(x):
    if x > 2030:
        return x
    else:
        return 0

posting_extract['3'] = posting_extract['3'].apply(lambda x: extract_condition(x))

posting_extract['2'] = posting_extract['2'].fillna(posting_extract[0])

posting_extract['2'] = posting_extract['2'].fillna(posting_extract['3'])

posting_extract['2'] = posting_extract['2'].replace(0, posting_extract['4'])

posting_extract = posting_extract[['2']]

#Now merge them back together. 

df = df.merge(posting_extract, how = 'left', left_index = True, right_index = True)

#Fill 0's with the information we extracted. 

df['Odometer'] = df['Odometer'].replace(0, df['2'])



df = df[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_status', 'Odometer']]



df['Odometer'] = df['Odometer'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Odometer'] = df['Odometer'].apply(lambda x: kilo_filter2(x))

print(df.columns)

conn = sqlite3.connect('MiataDatabase.db')   
c = conn.cursor() 

df.to_sql('Prep_Data', conn, if_exists= 'append') 

c.execute(''' SELECT DISTINCT * from Prep_Data''')

df = pd.DataFrame(c.fetchall())

df.columns = ['Index' ,'URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_status', 'Odometer']

c.close()

df.reset_index(inplace = True)

df = df[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_status', 'Odometer']]


#Clean up Price & Year.

df = df[df['Price'] != 'null']

df['Price'] = df['Price'].str.replace('$', "")
df['Price'] = df['Price'].str.replace(',', "")
df['Price'] = df['Price'].astype(int)

df = df[df['Price'] > 1000]
df = df[df['Price'] < 100000]

df = df[df['Year'] > 1989]
df = df[df['Year'] < 2022]

# We still have some 0's in the odometer columns that need to be filled.

def odometer_impute_condition(x):
    if x == 0:
        return "Estimated Mileage"
    else:
        return "Actual Mileage"

df['Odometer_Imputation'] = df['Odometer'].apply(lambda x: odometer_impute_condition(x)) 



#Since year and mileage are usually related, I'm going to find the mean for all years and use that to estimate the mileage of the vehicals. 

odo_average = df[df['Odometer'] != 0]
odo_average = odo_average[['Year', 'Odometer']]
odo_average = odo_average.groupby('Year').mean()
odo_average.reset_index(inplace = True)
odo_average['Odometer'] = odo_average['Odometer'].astype(int)

df = df.merge(odo_average, how = 'left', left_on = 'Year', right_on = 'Year')
df['Odometer'] = df['Odometer_x'].replace(0, df['Odometer_y'])

#Let's break out the date and see if we can use it for something interesting later. 
df = df[df['Post_Time'] != 'null']
df['Post_Time'] = df['Post_Time'].str.split('T', expand = True)
df['Post_Time'] = df['Post_Time'].str[:10]
df['Day'] = pd.to_datetime(df['Post_Time'])

df['Day'] = df['Day'].dt.day_name()
df['Month'] = pd.to_datetime(df['Post_Time'])
df['Month'] = df['Month'].dt.month_name()

#Almost there!  Let's do some final cleanup.



#Other is almost always used when people confuse paddle shifters with manual.  This is a safe assumption. 
def fix_transmission(x):
    if x == 'Other':
        return "Automatic"
    else:
        return x
    
df['Transmission'] = df['Transmission'].apply(lambda x: fix_transmission(x))

#Now we can define the generation codes. 

def generations(x):
    if 1998 >= x >= 1989:
        return 'NA'
    elif 2004 >= x >= 1999:
        return 'NB'
    elif 2015 >= x >= 2005:
        return 'NC'
    elif 2021 >= x >= 2016:
        return 'ND'
    
df['Generation'] = df['Year'].apply(lambda x: generations(x))



df = df[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month','Generation']]



df.columns = ['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_Status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month','Generation']

def condition_standardize(x):
    if x == 'Salvage':
        return 'Fair'
    elif x == 'Like New':
        return 'Excellent'
    else:
        return x

def estimated_condition(x):
    if pd.isna(x):
        return "Estimated"
    else:
        return "Actual"

df['Condition'] = df['Condition'].apply(lambda x: condition_standardize(x))
df['Condition_Estimated'] = df["Condition"].apply(lambda x: estimated_condition(x))
df = df[df['Condition'] != 'New']




df = df.drop_duplicates(subset = 'URL')

condition_exists = df[df.Condition.notnull()]
condition_predict = df[df.Condition.isnull()]
condition_predict.reset_index(inplace = True)
predict_set = condition_predict['Post_Text']




REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

condition_exists['clean_post'] = condition_exists['Post_Text'].apply(clean_text)

X = condition_exists.clean_post
Y = condition_exists.Condition

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 42)

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SGDClassifier(loss = 'hinge',random_state = 42)),])

nb.fit(X_train, y_train)    

y_pred = nb.predict(predict_set)


predicted_condition = pd.DataFrame(y_pred, columns = ['cond_pred'])
condition_predict = condition_predict.merge(predicted_condition, left_index = True, right_index = True)
condition_predict = condition_predict.reset_index()
condition_predict = condition_predict[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_Status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month', 'Generation', 'Condition_Estimated', 'cond_pred']]


condition_predict['Condition'] = condition_predict['cond_pred']



condition_predict = condition_predict[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_Status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month', 'Generation', 'Condition_Estimated']]

df = pd.concat([condition_predict, condition_exists])

df = df[['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location','State', 'Condition', 'Transmission', 'Title_Status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month', 'Generation', 'Condition_Estimated']]


def hardtop(x):
    if 'hardtop' in x:
        return 'Yes'
    elif 'Hardtop' in x:
        return 'Yes'
    elif 'RF' in x:
        return 'Yes'
    elif 'rf' in x:
        return 'Yes'
    elif 'PHRT' in x:
        return 'Yes'
    elif 'phrt' in x:
        return x
    elif 'hard top' in x:
        return 'Yes'
    elif 'Hard top' in x:
        return 'Yes'
    elif 'Hard Top' in x:
        return 'Yes'
    else:
        return 'No'

df['Hardtop_Desc'] = df['Post_Text'].apply(lambda x: hardtop(x))
   
#Now we have to store our data so the dataset can continue to grow over time. 
conn = sqlite3.connect('MiataDatabase.db')   
c = conn.cursor() 

df.to_sql('Finished_Data', conn, if_exists= 'append', index = False) 
c.execute(''' SELECT DISTINCT * from Finished_Data''')

MiataDatabase = pd.DataFrame(c.fetchall())

c.close()

MiataDatabase.columns = ['URL', 'Title', 'Price', 'Post_Time', 'Post_Text', 'Year', 'Location', 'State', 'Condition', 'Transmission', 'Title_Status', 'Odometer_Imputation', 'Odometer', 'Day', 'Month', 'Generation', 'Condition_Estimated', 'Hardtop_Desc']

MiataDatabase = MiataDatabase.drop_duplicates(subset = 'URL')
#Now that our data is stored and able to be built over time we need to check if the URLs coming out of the database are currently active or if they have expired.


url_check = MiataDatabase.URL.tolist()
code_list = []
for i in url_check:
    response = requests.get(i)
    code_list.append(response.status_code)

def url_code(x):
    if x == 200:
        return "Active Link"
    elif x == 404:
        return "Inactive Link"

code_list = pd.DataFrame(code_list, columns = ['Active_Links'])

MiataDatabase = MiataDatabase.merge(code_list, how = 'left', left_index = True, right_index = True)


MiataDatabase['Active_Links'] = MiataDatabase['Active_Links'].apply(lambda x: url_code(x))


MiataDatabase.dropna(inplace = True)



#That that our data is prepared, let's do some exploratory analysis.  
#I'll do some more indepth work in Tableau so these are mostly for me to decide on what model to use for price prediction.





#Without doing any processing, we can see that we have a decently postive relationship between price and year and a negative relationship between price and mileage. 
#Both of these were intuitive and to be expected by still cool to visualize to drive a point home. 


MiataDatabaseAnalysis = MiataDatabase[['Price','Year', 'Odometer','Condition', 'Hardtop_Desc', 'Title_Status', 'State', 'Generation']]
dummies = pd.get_dummies(MiataDatabaseAnalysis[['Condition', 'Hardtop_Desc', 'Title_Status', 'State', 'Generation']])

MiataDatabaseAnalysis = MiataDatabaseAnalysis.merge(dummies, left_index = True, right_index = True, how = 'left')

column_list = dummies.columns
column_list = column_list.tolist()
column_list.extend(('Year' , 'Odometer'))


Y = MiataDatabaseAnalysis['Price']
X = MiataDatabaseAnalysis[column_list]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 42)

reg = RandomForestRegressor(n_estimators = 60, random_state = 42)
reg.fit(X_train, y_train)
print('Current Number of Listings: ' + str(len(Y)))
print('Model currently scoring at ' + str((reg.score(X_test, y_test))))

predicted_values = reg.predict(X)   

pred_values = pd.DataFrame(predicted_values, columns = ['Predicted_Price'])
    
df = MiataDatabase.merge(pred_values, left_index = True, right_index = True, how = 'left')
   
df.to_excel('outputfordashboard.xlsx')



    













 






        