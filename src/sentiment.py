# Import packages
import re
import os
import operator import itemgetter
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text():
    # read the txt file
    file = open(os.path.join("data", "civil_war_diary_1864.txt"), 'r')
    file = file.read()

    # D A T E S
    # find all dates using regex. Dates are formatted like so "  FRIDAY, March 18, 1864."
    dates = re.compile(r'\s\w{1,6}DAY,\s\w{3,5}[.]?\s\d{1,2}[,|.]\s1864[.]\n')
    dates = dates.findall(file)

    # cleaning the dates
    dates_clean = []
    for i in range(len(dates)):
        current_date = dates[i]
        current_date = current_date.replace("\n", "")
        current_date = re.sub(r'\t\w+DAY(.|,)\s', '', current_date)
        current_date = re.sub(r'(.|,)\s', "", current_date)
        dates_clean.append(current_date)
    
    # E N T R I E S
    # find all dates using regex. Entries are preceded by two new line characters and followed by two new line characters.
    entries = re.compile(r'((?<=1864[.]\n{2})(.*\n)+?(?=\n{2}))')
    entries = entries.findall(file)

    # cleaning the dates by removing new line characters
    entries_clean = []
    for i in range(len(entries)):
        current_entry = entries[i][0]
        current_entry = current_entry.replace("\n", " ")
        entries_clean.append(current_entry)
    
    return dates_clean, entries_clean

def sentiment_analysis(entries_clean):
    # initialize the Transformers classifier pipeline
    classifier = pipeline(task="sentiment-analysis")

    sentiments = []
    for entry in entries_clean:
        # if entry is under 1500 characters peform analysis right away (pipeline has a character limit)
        if len(entry) < 1500:
            output = classifier(entry)
            sentiments.append(output)
        else:
            # if over 1500 characters devide entry into 1500 character chunks
            entry_chunks = re.findall('.{1,1500}', entry)
            # peform analysis on every chunk
            sentiment_chunks = []
            for chunk in entry_chunks:
                output = classifier(chunk)
                sentiment_chunks.append(output)
            # "sentiment_chunks" is a list of list of dicts. It must be flattened into a list of dict to create pandas dataframe.
            sentiment_chunks = [dict for sublist in sentiment_chunks for dict in sublist]
            # sort scores from highest to lowest
            sentiment_chunks_sorted = sorted(sentiment_chunks, key=itemgetter('score'), reverse=True)
            # return first entry. re-wrap in list to have consistent formating to the results from above.
            clearest_sentiment = [sentiment_chunks_sorted[0]]
            sentiments.append(clearest_sentiment)
    
    # variable "results" is a list of list of dicts. It must be flattened into a list of dict to create pandas dataframe.
    results = [dict for sublist in sentiments for dict in sublist]

    return results

def make_df(dates_clean, entries_clean, results):
    
    # making data frame from dates
    df1 = pd.DataFrame(dates_clean, columns=["date"])

    # making a list with all the months
    months = ["January"]*31 + ["February"]*29 + ["March"]*31 + ["April"]*30 + ["May"]*31 + ["June"]*30 + ["July"]*31 + ["August"]*31 + ["September"]*30 + ["October"]*31 + ["November"]*30 + ["December"]*31
    # appending as column
    df1['month'] = months

    # making data frame from entries
    df2 = pd.DataFrame(entries_clean, columns=["entry"])

    # making data frame from sentiment analysis results
    df3 = pd.DataFrame.from_records(results)

    # combining the data frames
    df = pd.concat([df1, df2, df3], axis=1)

    # saving data frame
    df.to_csv(os.path.join("out", "results.csv"))

    return df

def find_highest_scores(df):

    # L A B E L : N E G A T I V E 
    # selecting only rows with negative labels
    df_negative = df[df["label"]=="NEGATIVE"]

    # sorting by most certain
    df_negative = df_negative.sort_values(by="score", ascending=False)
    # selecting top 5
    df_negative = df_negative.head(5)
    # converting to list
    df_negative = df_negative["entry"].tolist()

    # saving negative results as txt file
    with open(os.path.join("out", "negative.txt"), 'w') as f:
    for entry in df_negative:
        f.write("%s\n" % entry)
    
    # L A B E L : P O S I T I V E
    df_positive = df[df["label"]=="POSITIVE"]
    df_positive = df_positive.sort_values(by="score", ascending=False)
    df_positive = df_positive.head(5)
    df_positive = df_positive["entry"].tolist()

    with open(os.path.join("out", "positive.txt"), 'w') as f:
    for entry in df_positive:
        f.write("%s\n" % entry)

def make_visualizations(df):
    # defing colour palette
    palette = {"NEGATIVE" : "#B2301B", "POSITIVE" : "#4B5BC1"}

    # making a plot showin the distribution of negatives and posives
    sns.displot(df, x="label", hue="label", palette=palette)
    plt.savefig(os.path.join("out", "dis_label.png"))

    # making a plot showing the distribution of scores
    plt.clf()
    sns.catplot(data=df, x="label", hue="label", y="score",  kind="swarm", palette=palette)
    plt.savefig(os.path.join("out", "score.png"))
    
    # making a plot showing the development of sentiment through the year
    plt.clf()
    sns.displot(df, x="month", hue="label", palette=palette)
    # rotating the labels on x-axis
    plt.xticks(rotation=45, 
                    horizontalalignment='right',
                    fontweight='light',
                    fontsize='small')
    plt.savefig(os.path.join("out", "dis_month.png"))

def main():
    dates_clean, entries_clean = clean_text()
    results = sentiment_analysis(entries_clean)
    df = make_df(dates_clean, entries_clean, results)
    find_highest_scores(df)
    make_visualizations(df)

if __name__=="__main__":
    main()
