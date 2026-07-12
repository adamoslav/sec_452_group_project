The two data files are from these respective URL's

Headlines: https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/main/data_scraper/headline%20scraper/headlines/aa.csv
News_Data: https://github.com/Zdong104/FNSPID_Financial_News_Dataset/tree/main/data_processor/news_data_raw

Since they are both called "aa.csv", I have renamed them to their respective names and I have included the renamed csv files in the Gitlab for ease of use.

For the sentiment agent, there are multiple files and a separate folder "Semantic_Models". 

The folder contains all of the models we are trying and testing to see which is ther most accurate. We then choose the best model from the "Semantic_comparison" file. 

The file is called "Semantic_functions", which contains all of the functions to clean, standardise, and a semantic comparison for the data. Each function does a different thing:

merge_files(): This joins each csv file together based on the URL's. This is because there is a mismatch in lengths, as well as many disorganised rows in the News_Data file. 

extract_target_sentences(text, phrase): This finds the sentences containing the TARGET word, which will be the stock like "AAPL". It returns only the sentences with that word in. 

parse_custom_date(date_str): This standardises the dates into a usable format so that the function can be called for a specific date. It changes EDT times into EST through the dateutil.parser library, since the timings are different by 1 hour. 

semantic(TARGET, target_date=None): This is the final function that calls all of the other functions for the data. It finds the score for the data, then finds the closest date to the date inputted, before returning a final output of the score at the closest date within a span of a year to the target_date for a specific TARGET. 
