import pandas as pd
import re 
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from dateutil.parser import parse

class FinancialSentimentAnalyser:
    def __init__(self, headlines_csv='Headlines.csv', news_data_csv='News_Data.csv'):
        self.headlines_csv = headlines_csv
        self.news_data_csv = news_data_csv
        
        #Load the models once upon initialisation instead of at each time
        self.finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)

    def merge_files(self):
        #All of the datasets + drop blank values
        headlines_df = pd.read_csv(self.headlines_csv)
        news_df = pd.read_csv(self.news_data_csv, on_bad_lines='skip', low_memory=False)
        news_df = news_df.dropna(subset=['Text', 'Mark'])

        #This has all the URLs needed for joining
        valid_urls = set(headlines_df['URL'].dropna())

        #Find the URLs so that only when they are in both dataframes it stays
        filtered_news_df = news_df[news_df['Url'].isin(valid_urls)]

        #Drop duplicates from the Headlines side to prevent creating extra duplicate rows during the merge
        headlines_subset = headlines_df[['URL', 'Headline']].drop_duplicates(subset=['URL'])

        #Perform the Merge
        merged_df = pd.merge(filtered_news_df, headlines_subset, left_on='Url', right_on='URL', how='left')

        #Drop the extra URL column + unnamed columns that were made + no description text values
        merged_df = merged_df.drop(columns=['URL'])
        merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
        merged_df = merged_df[merged_df['Text'] != "0"]

        return merged_df

    @staticmethod
    def extract_target_sentences(text, phrase):
        #Handle empty or NaN values
        if pd.isna(text):
            return ""
        
        #Split text into sentences based on punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', str(text))
        #This does the exact phrase and nothing more e.g. BA doesn't have BATH
        pattern = r'\b' + re.escape(phrase) + r'\b'
        
        #Filter sentences that contain the target phrase 
        matched_sentences = []
        for sentence in sentences:
            if re.search(pattern, sentence):
                matched_sentences.append(sentence.strip())
        
        #Rejoin the matched sentences into a single string
        return " ".join(matched_sentences)

    @staticmethod
    def parse_custom_date(date_str):
        try:
            if pd.isna(date_str):
                return pd.NaT
                
            date_str2 = str(date_str)
            edt = 'EDT' in date_str2
            
            #Remove dash and timezone
            clean_str = date_str2.replace('—', '').replace('EST', '').replace('EDT', '').strip()
            parsed_dt = parse(clean_str)
            
            #Convert EDT to EST
            if edt:
                parsed_dt = parsed_dt - pd.Timedelta(hours=1)
                
            return parsed_dt
        except Exception:
            return pd.NaT

    def semantic_year(self, target, year):
        data = self.merge_files()

        #Filter data first not last so that its standardised for indexing
        data['Date'] = data['Date'].apply(self.parse_custom_date)
        data = data.dropna(subset=['Date'])
        
        data = data[(data['Date'] >= f"{year-1}-01-01") & (data['Date'] <= f"{year}-12-31")]

        #Create a new column that has only the Target sentences in
        data['Filtered_Text'] = data['Text'].apply(lambda x: self.extract_target_sentences(x, target))
        data = data[data['Filtered_Text'] != ""]

        if data.empty:
            return pd.DataFrame()

        #Use tolist() for scalability just in case more data is added
        texts_to_process = data['Filtered_Text'].astype(str).tolist()

        results = []

        #Run Sentiment Analysis using the preloaded pipeline
        for (index, row), output in zip(data.iterrows(), self.nlp(texts_to_process, truncation=True)):
            results.append({
                'headline': row['Headline'],
                'label': output['label'],
                'score': output['score'],
                'date': row['Date']
            })

        df_results = pd.DataFrame(results)

        #merge_asof requires the right kind of DataFrame to be sorted by the merge key
        df_results = df_results.sort_values('date')

        #DataFrame with every day of the target year
        df_year = pd.DataFrame({'target_date': pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')})

        #direction='backward' only looks at headlines less than or equal to the target date
        #tolerance=pd.Timedelta(days=365) replicates the original 365-day filter window for good measure
        final_df = pd.merge_asof(
            df_year, 
            df_results, 
            left_on='target_date', 
            right_on='date', 
            direction='backward',
            tolerance=pd.Timedelta(days=365)
        )

        #Easy look at date column to see NaT if there was no headline if just scrolling
        final_df['score'] = final_df['score'].fillna(0)
        final_df['label'] = final_df['label'].fillna('neutral')
        final_df['headline'] = final_df['headline'].fillna('No headline in range')

        return final_df

#analyzer = FinancialSentimentAnalyser()
#result_2020 = analyzer.semantic_year("BA", 2020)
#print(result_2020)