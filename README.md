# sentiment-in-the-recollections-of-a-soldier

In this repository I peform sentiment analysis on the personal recollections of a young soldier who served under General Ulysses S. Grant during the American Civil War. The analysis will cover the year of 1864. The purpose is to explore how machine learning can aid in analysing and visualizing the personal experience of being on the front lines of war. 

Lemuel Abijah Abbott's diary can be accessed on Project Gutenburg via this [link](https://www.gutenberg.org/ebooks/47332)

The script in this repository does the following: 
1. Prepares the text for analysis. The diary has been downloaded as a plain text file and can be found in the ```data``` folder. By using regex the dates and entries will be seperated into two separate lists.
2. Peform sentiment analysis on all entries using a Transformers pipeline from the HuggingFace library.
3. Saves a csv file to the folder ```out``` with all the data including a new column showing the month.
4. Saves two text files to the folder ```out``` containing the five entries with the highest certainty score for negative sentiment and positive sentiment respectively. 
5. Creates and saves three visualizations using Seaborn. One showing the distribution between negative and positive labels. Another showing the distribution of certainty scores. And a last showing the distribution between labels over the twelve month period.

## User instructions

1. Install the necessary packages listed in the requirements file by using the setup shell script like so:

```bash setup.sh```

2. Run the code in the ```src``` folder by using the run shell script:

```bash run.sh```



