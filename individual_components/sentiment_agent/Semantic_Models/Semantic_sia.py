import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Semantic_functions import merge_files, extract_target_sentences

TARGET = "AAPL"

data = pd.DataFrame(merge_files())

#Create a new column that has only the TARGET sentences in
data['Filtered_Text'] = data['Text'].apply(lambda x: extract_target_sentences(x, TARGET))
data = data[data['Filtered_Text'] != ""]

headlines = data['Filtered_Text'].dropna().tolist()


sia = SentimentIntensityAnalyzer()
results = []

for index, row in data.iterrows():
    #Calculate the VADER sentiment on the filtered text
    score = sia.polarity_scores(str(row['Filtered_Text'])) 
    
    #Store the results using the specific headline for this row
    results.append({
        'headline': row['Headline'],
        'compound': score['compound']
    })
#Convert to a DataFrame
df = pd.DataFrame(results)


print(df[['headline', 'compound']].head(10))
print(f"\nTotal Headlines Processed: {len(df)}")