import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from Semantic_functions import merge_files, extract_target_sentences

TARGET = "AAPL"

data = pd.DataFrame(merge_files())

#Create a new column that has only the Target sentences in
data['Filtered_Text'] = data['Text'].apply(lambda x: extract_target_sentences(x, TARGET))
data = data[data['Filtered_Text'] != ""]

headlines = data['Filtered_Text'].dropna().tolist()

#Load the FinBERT model
finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


results = []

for index, row in data.iterrows():
    #Truncate the text to ensure it fits the model limits
    output = nlp(str(row['Filtered_Text']), truncation=True)[0]
    
    
    #Store results
    results.append({
        'headline': row['Headline'],
        'label': output['label'],
        'score': output['score'],
    })

#Convert to a DataFrame
df_results = pd.DataFrame(results)

#Change scores signs based off of labels
df_results.loc[df_results['label'] == 'negative', 'score'] = -df_results['score']
df_results.loc[df_results['label'] == 'neutral', 'score'] = 0

print(df_results[['headline', 'label', 'score']].head(10))
print(f"\nTotal items processed: {len(df_results)}")