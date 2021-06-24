"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies

from IPython.display import display
import numpy as np 
import pandas as pd
import nltk
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.features import RadViz


# Noise removal helper libraries
import re
import string 
from stopwordsiso import stopwords as sw
from nltk.corpus import stopwords

# Text Preprocessing
from nltk.tokenize import TweetTokenizer
from nltk import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Feature Engineering and Data preparation for modelling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Model building and training
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#Model evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from PIL import Image



# Vectorizer
vectorizer = open("resources/CVvectorizer.pkl","rb")
tweet_cv = joblib.load(vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
train = pd.read_csv("train.csv")
test = pd.read_csv("test_with_no_labels.csv")
img_raw = Image.open('resources/imgs/raw.jpg')
img_clean = Image.open('resources/imgs/clean.jpg')
pos_freq = Image.open('resources/imgs/pos_freq.jpg')
neg_freq = Image.open('resources/imgs/neg_freq.jpg')
neu_freq = Image.open('resources/imgs/neu_freq.jpg')
news_freq = Image.open('resources/imgs/news_freq.jpg')
warming_1 = Image.open('resources/imgs/warming_1.jpeg')
warming_2 = Image.open('resources/imgs/warming_2.jpeg')

# The main function where we will build the actual app

def main():
        """Tweet Classifier App with Streamlit """

        # Creates a main title and subheader on your page -
        # these are static across all pages
        st.title("Climate Change Sentiment Analysis")
        st.write("Classifying tweets about man-made climate change.")
        
        #Creates a main title on the sidebar.
        st.sidebar.title("Climate Change Sentiment Analysis")
        st.sidebar.write("Classifying tweets about man-made climate change.")

        #Create a label to display group name
        st.sidebar.subheader("Classification_AM3")

        #Create checkbox to display team member names
        team = st.sidebar.checkbox("Collaborators")

        #List of group members
        #When team checkbox is ticked
        if team:
                
                st.subheader("**Classification_Am3**")
                st.image(warming_1)
                st.write("Collaborators")
                st.markdown("* Katleho Mputhi - Coordinator")
                st.markdown("* Hendri Kouter")
                st.markdown("* Siyamukela Hadebe")
                st.markdown("* Wisani Khosa")
                st.markdown("* Dineo Seakhela")

        #Introduction Page
        st.sidebar.subheader("Introduction")

        #Create a introduction section
        intro = st.sidebar.checkbox("Context")

        #When intro checkbox is ticked
        if intro:
                
                #Create Context label for the introduction section
                st.subheader("Context")

                st.image(warming_2)

                #Create introduction bullet points
                st.markdown("**EDSA Climate Change Belief Analysis 2021**")
                st.markdown("Over the years more and more companies have come to realise\
                            the benefits of paying attention to the impact their operations\
                            have on the environment as an increasing number of consumers\
                            become concerned the reduction of their and their brand of\
                            choice's carbon footprint reduction. Consumers who support the\
                            idea of a human driven change in climate tend to support companies\
                            that are more environmentally friendly, for instance those that\
                            have moved from the use of plastic to that paper and other alternatives.\
                            \
                            For this reason, it is worthwhile for a company to observe the opinions\
                            of existing and potential clients in order to establish public opinion\
                            and adjust their marketing practices accordingly.")

        #Create a label for the preprocessing section on the sidebar
        st.sidebar.subheader("Preprocessing")

        #Create a checkbox that shows training dataframe, raw and clean
        df = st.sidebar.checkbox("Tweets")

        #When the df check is ticked
        if df:
                #Create two columns to display the two dataframe
                train_set, clean_set = st.beta_columns(2)

                #Display the image showing the first ten rows of the raw training set

                #Create a label for the raw train set image
                st.subheader(" Raw Train Set")

                #Display the image
                st.image(img_raw)

                #Write the shape of the raw training set
                st.write("Training data shape:", train.shape)

                #Display the image showing the first ten rows of the clean training set

                #Create a label for the clean train set image
                st.subheader('Clean Train Set')

                #Display the image
                st.image(img_clean)

                #Write the shape of the clean training set
                st.write("Testing data shape:", test.shape)
        
        #cache text preprocessing function
        @st.cache(persist = True)

        #Create a function that clean the data in a dataframe
        def text_preprocessing (df):
                pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
                pattern_digits = r'\d+'
                
                df = df.copy()
                
                df['message'] = df['message'].replace(to_replace = pattern_url, value = '', regex = True)
                df['message'] = df['message'].replace(to_replace = pattern_digits, value = '', regex = True)
                
                low = lambda tweets: ''.join([tweet.lower() for tweet in tweets])
                df['message'] = df['message'].apply(low)
                
                punct = lambda tweets: ''.join([tweet for tweet in tweets if tweet not in string.punctuation])
                df['message'] = df['message'].apply(punct)
                
                df.message.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
                
                return df
        
        #Call the text preprocessing function on the train and test sets       
        clean_train = text_preprocessing(train)
        clean_test = text_preprocessing(test)

        #cache tokenization function
        @st.cache(persist = True)

        #Create a function that tokenizes the words in a dataframe
        def tokens(df, column_name):
                df = df.copy()
    
                tknzr = TweetTokenizer(reduce_len = True)
                df[column_name] = df[column_name].apply(tknzr.tokenize)
                return df
        
        #Call the tokens funciton on the train and test sets
        tokenized_train = tokens(clean_train, 'message')
        tokenized_test = tokens(clean_test, 'message')

        #Create a function that removes stopwords from a dataframe
        @st.cache(persist = True)

        #Create function that remove stopwords from a dataframe
        def stop(df, column_name):
                df = df.copy()
    
                rt = lambda tweets: [tweet for tweet in tweets if tweet != 'rt']
                df[column_name] = df[column_name].apply(rt)
                stops = lambda tweets: [tweet for tweet in tweets if tweet not in sw('en')]
                df[column_name] = df[column_name].apply(stops)
                return df
        
        #Call the stop function on the train and test sets
        stopwords_train = stop(tokenized_train, 'message')
        stopwords_test = stop(tokenized_test, 'message')

        #cache the lemmatization function
        st.cache(persist = True)

        #Create function that lemmatizes the words in a dataframe
        def lem(df, column_name):
                #Instantiate WordNetLemmatizer
                lemmatizer = WordNetLemmatizer()
        
                df = df.copy()
                df[column_name] = df[column_name].apply(lambda sentence: [lemmatizer.lemmatize(word) for word in sentence])
                return df

        #Call the lem function on the train and test sets
        lemmatized_train = lem(stopwords_train, 'message')
        lemmatized_test = lem(stopwords_test, 'message')

        #Join the tokenized words into complete sentences
        lemmatized_train['message'] = [' '.join(tweet) for tweet in lemmatized_train['message'].values]
        lemmatized_test['message'] = [' '.join(tweet) for tweet in lemmatized_test['message'].values]     

        #Create the label of the eda page on the side bar
        st.sidebar.subheader("Exploratory Data Analysis")

        #Write a short section description
        st.sidebar.markdown("**Distribution of Tweets per Sentiment**")

        #Create a new dataframe that counts the number of observations per sentiment
        group = lemmatized_train.groupby('sentiment').count()['message'].reset_index().sort_values(by = 'message', ascending = False)

        #Create bar chart checkbox
        bar_chart = st.sidebar.checkbox("Bar Chart")

        #When bar chart is ticked
        if bar_chart:
                st.subheader("The Number of Tweets Per Sentiment")
                #cache function that creates a bar chart
                @st.cache(persist = True)

                #Create function that creates a bar chart
                def bar():
                        
                        fig = go.Figure(go.Bar(x = ['Positive', 'News', 'Neutral', 'Negative'],
                                       y = group['message'], marker = {'color': group['message'],
                                                                       'colorscale': 'plasma'}))
                        fig.update_layout(yaxis_title = 'Tweets', xaxis_title = 'Sentiment')
                        return fig

                #Display bar chart
                st.plotly_chart(bar())
                st.markdown("The positive class is well represented, holding the highest number of tweets,\
                            while the negative class holds the lowest number of tweets. This could be due\
                            to the fact that opinions of disbelief expressed over the internet with regards\
                            to climate change and other scientific theories are often met with a high amount\
                            of criticism, leaving individuals who share these opinions unwilling to voice them.")

        #Create funnel chart checkbox        
        funnel_chart = st.sidebar.checkbox("Funnel Chart")

        #When funnel chart is ticked
        if funnel_chart:
                st.subheader("The Proportion of Tweets Per Sentiment")
                #cache function that creates a funnel chart
                @st.cache(persist = True)

                #Create function that plots a funnel chart
                def funnel():
                        
                        fig = go.Figure(go.Funnelarea(text = ['Positive', 'News', 'Neutral', 'Negative'],
                              values = group['message'], marker = {'colors': group['message']}, 
                              title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"},
                              labels = ['Positive', 'News', 'Neutral', 'Negative']))
                        return fig
                
                #Display the funnel chart
                st.plotly_chart(funnel())

        #Create a wordcloud section heading on the sidebar
        word_cloud = st.sidebar.markdown("**Word Cloud**")        

        #Create a positive wordcloud checkbox         
        word_positive = st.sidebar.checkbox("Positive")
        
        # Create a list of the four most frequent words in the wordclouds
        climate_list = ['climate', 'change', 'global', 'warming']
        
        #When the positive checkbox is ticked
        if word_positive:

                #Select positive words from lemmatized_train 
                positive_words = " ".join([sentence for sentence in lemmatized_train['message'][lemmatized_train['sentiment'] == 1]])

                #Create a new list with words that are not "climate", "change", "global", "warming"
                new_positive = " ".join([word for word in positive_words.split() if word not in climate_list])
                
                #Insert label above wordcloud
                st.header("Pro Man-Made Climate Change")

                #cache positive wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the positive sentiment.
                def gen_poscloud(allWords):
                        positive_wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(positive_wordcloud, interpolation = 'bilinear')
                        
                        plt.axis('off')
                        plt.savefig('positive_wordcloud.jpg')
                        pos = Image.open('positive_wordcloud.jpg')
                        return pos
                
                #cache positive wordcloud function
                @st.cache(persist = True)
                
                #Create a function that creates wordcloud for the negative positive with none of the subject matter related words.
                def gen_posclean(allWords):
                        positive_cleancloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(positive_cleancloud, interpolation = 'bilinear')
                        
                        plt.axis('off')
                        plt.savefig('positive_cleancloud.jpg')
                        pos_clean = Image.open('positive_cleancloud.jpg')
                        return pos_clean

                st.subheader("Word Cloud")
                st.markdown('**Pro Sentiment WordCloud Before Word Removal**')
                #Print out the wordcloud
                pos = gen_poscloud(positive_words)
                st.image(pos)

                st.markdown('**Pro Sentiment WordCloud After Word Removal**')
                #Print out the wordcloud
                clean_pos = gen_posclean(new_positive)
                st.image(clean_pos)

        #Create a negative wordcloud checkbox          
        word_negative = st.sidebar.checkbox("Negative")

        #When the negative checkbox is ticked
        if word_negative:

                #Select negative words from lemmatized_train 
                negative_words = " ".join([sentence for sentence in lemmatized_train['message'][lemmatized_train['sentiment'] == -1]])

                #Create a new list with words that are not "climate", "change", "global", "warming"
                new_negative = " ".join([word for word in negative_words.split() if word not in climate_list])
                
                #Insert label above wordcloud
                st.header("Anti Man-Made Climate Change")

                #cache negative wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the negative sentiment.
                def gen_negcloud(allWords):
                        negative_wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(negative_wordcloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('negative_wordcloud.jpg')
                        neg = Image.open('negative_wordcloud.jpg')
                        return neg

                #cache negative wordcloud function
                st.cache(persist = True)
                
                #Create a function that creates wordcloud for the negative sentiment with none of the subject matter related words.
                def gen_negclean(allWords):
                        negative_cleancloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(negative_cleancloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('negative_cleancloud.jpg')
                        neg_clean = Image.open('negative_cleancloud.jpg')
                        return neg_clean
                
                st.subheader("Word Cloud")
                st.markdown('**Anti Sentiment WordCloud Before Word Removal**')
                #Print out the wordcloud
                neg = gen_negcloud(negative_words)
                st.image(neg)

                st.markdown('**Anti Sentiment WordCloud After Word Removal**')
                #Print out the wordcloud
                clean_neg = gen_negclean(new_negative)
                st.image(clean_neg)
                        
                        
        #Create a neutral wordcloud checkbox
        word_neutral = st.sidebar.checkbox("Neutral")
        
        #When the neutral checkbox is ticked
        if word_neutral:

                #Select neutral words from lemmatized_train
                neutral_words = " ".join([sentence for sentence in lemmatized_train['message'][lemmatized_train['sentiment'] == 0]])

                #Create a new list with words that are not "climate", "change", "global", "warming"
                new_neutral = " ".join([word for word in neutral_words.split() if word not in climate_list])

                #Insert label above wordcloud
                st.header("Neutral on Man-Made Climate Change")

                #cache neutral wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the neutral sentiment.
                def gen_neucloud(allWords):
                        neutral_wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(neutral_wordcloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('neutral_wordcloud.jpg')
                        neu = Image.open('neutral_wordcloud.jpg')
                        return neu
                
                #cache neutral wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the neutral sentiment with none of the subject matter related words.
                def gen_neuclean(allWords):
                        neutral_cleancloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(neutral_cleancloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('neutral_cleancloud.jpg')
                        neu_clean = Image.open('neutral_cleancloud.jpg')
                        return neu_clean

                st.subheader("Word Cloud")
                st.markdown('**Neutral Sentiment WordCloud Before Word Removal**')
                #Print out the wordcloud
                neu = gen_neucloud(neutral_words)
                st.image(neu)

                
                st.markdown('**Neutral Sentiment WordCloud After Word Removal**')
                #Print out the wordcloud
                clean_neu = gen_neuclean(new_neutral)
                st.image(clean_neu)

        #Create a news wordcloud checkbox                
        word_news = st.sidebar.checkbox("News")

        #When the news checkbox is ticked
        if word_news:

                #Select news words from lemmatized_train
                news_words = " ".join([sentence for sentence in lemmatized_train['message'][lemmatized_train['sentiment'] == 2]])

                #Create a new list with words that are not "climate", "change", "global", "warming"
                new_news = " ".join([word for word in news_words.split() if word not in climate_list])

                #Insert label above wordcloud
                st.header("News related to Climate Change")

                #cache news wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the news sentiment.
                def gen_newscloud(allWords):
                        news_wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        
                        plt.imshow(news_wordcloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('news_wordcloud.jpg')
                        news = Image.open('news_wordcloud.jpg')
                        return news

                #cache news wordcloud function
                @st.cache(persist = True)

                #Create a function that creates wordcloud for the news sentiment with none of the subject matter related words.
                def gen_newsclean(allWords):
                        news_cleancloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100, background_color = 'black').generate(allWords)
                        plt.imshow(news_cleancloud, interpolation = 'bilinear')
                        plt.axis('off')
                        plt.savefig('news_cleancloud.jpg')
                        news_clean = Image.open('news_cleancloud.jpg')
                        return news_clean

                st.subheader("Word Cloud")
                st.markdown('**News Sentiment WordCloud Before Word Removal**')
                #Print out the wordcloud
                news = gen_newscloud(news_words)
                st.image(news)

                st.markdown('**News Sentiment WordCloud After Word Removal**')
                #Print out the wordcloud
                clean_news = gen_newsclean(new_news)
                st.image(clean_news)
                
        #Create a bar plot of word frequencies
        st.sidebar.markdown('**Word Frequencies per Sentiment**')
        
        pro = st.sidebar.checkbox("Positive Views")
        anti = st.sidebar.checkbox("Negative Views")
        neutral = st.sidebar.checkbox("Neutral Views")
        news = st.sidebar.checkbox("The News")
        
        # Create function that removes words in climate_list 

        if anti:
                #Create a label for the top 5 most frequent words set image
                st.subheader('Top 5 Most Frequent Words in the Pro Class')

                #Display the image
                st.image(pos_freq)
                        
        if pro:
               #Create a label for the top 5 most frequent words set image
                st.subheader('Top 5 Most Frequent Words in the Pro Class')

                #Display the image
                st.image(neg_freq) 

        if neutral:
                #Create a label for the top 5 most frequent words set image
                st.subheader('Top 5 Most Frequent Words in the Pro Class')

                #Display the image
                st.image(neu_freq)
                

        if news:
                #Create a label for the top 5 most frequent words set image
                st.subheader('Top 5 Most Frequent Words in the Pro Class')

                #Display the image
                st.image(news_freq)
                
                        
        #Create a hashtags frequencies section
        st.sidebar.markdown("**Number of hashtags per sentiment**")
        hasht = st.sidebar.checkbox("Hashtag Frequency Distribution Plot")

        # Create function which extracts hashtags from the dataframe
        @st.cache(persist = True)
        def hashtag_extract(message):
    
                # Create an empty list which will be used to collect hashtags
                hashtags = []
    
                # For every word in message, find words that start with '#' and append them to the empty list
                for tweet in message: # Look through message
                        ht = re.findall(r"#(\w+)", tweet) # Use regex pattern to find hashtags
                        hashtags.append(ht) # Append the extracted hashtags to the empty list
        
                # Return the created list of hashtags
                return hashtags
        if hasht:
                st.cache(persist = True)
                def po_ha():
                        # Use the hashtags extract function to get hashtags associated with the positive class from the original training dataframe
                        positive_hashtags = hashtag_extract(train['message'][train['sentiment'] == 1])

                        # Create a single list for the positive sentiment
                        positive_hashtags = sum(positive_hashtags, [])

                        # Create a distribution plot of the most frequent hashtags in the positive hashtags list 
                        freq = nltk.FreqDist(positive_hashtags)

                        # Create a dataframe from the result of the frequency distribution plot, using the hashtags in one column and frequencies
                        # in a second
                        df = pd.DataFrame({'Hashtags' : list(freq.keys()), 'Count' : list(freq.values())})

                        # Sort the hashtags by order of descending counts and show the first 10 rows i.e the 10 most frequent hashtags
                        df_pos = df.sort_values(by = 'Count', ascending = False)

                        # Create a bar plot using the group dataframe  to visualise the number of tweets per class
                        fig_po = go.Figure(go.Bar(x = df_pos['Hashtags'].head(10), y = df_pos['Count'].head(10).sort_values(ascending = False), # Specify x and y variables
                               marker = {'color': df_pos['Count'],'colorscale': 'viridis'})) # Select a colour for the graph

                        # Add title, x and y axis labels to the bar chart
                        fig_po.update_layout(yaxis_title = 'Hashtag Counts', xaxis_title = 'Hashtags', title = 'Count of Hashtags for Positive Man-Made Climate Change Sentiment')
                        return fig_po

                # Show the bar plot
                st.plotly_chart(po_ha())

                st.cache(persist = True)
                def n_ha():
                        # Use the hashtags extract function to get hashtags associated with the negative class from the original training dataframe
                        negative_hashtags = hashtag_extract(train['message'][train['sentiment'] == -1])
                        # Create a single list for the negative sentiment
                        negative_hashtags = sum(negative_hashtags, [])

                        # Create a distribution plot of the most frequent hashtags in the negative hashtags list
                        freq = nltk.FreqDist(negative_hashtags)

                        # Create a dataframe the results of the frequency distribution plot , using the hashtags in the one column and frequencies
                        # in a second
                        df = pd.DataFrame({'Hashtags' : list(freq.keys()),
                                   'Count' : list(freq.values())})

                        # Sort the hashtags by order descending counts and show the first 10 rows i.e the 10 most frequent hashtags
                        df_neg = df.sort_values(by = 'Count', ascending = False)
                        df_neg.head(10)

                        # Create a bar plot using the group dataframe  to visualise the number of tweets per class
                        fig_n = go.Figure(go.Bar(x = df_neg['Hashtags'].head(10), y = df_neg['Count'].head(10).sort_values(ascending = False), # Specify x and y variables
                                       marker = {'color': df_neg['Count'],'colorscale': 'viridis'})) # Select a colour for the graph

                        # Add title, x and y axis labels to the bar chart
                        fig_n.update_layout(yaxis_title = 'Hashtag Counts', xaxis_title = 'Hashtags', title = 'Count of Hashtags for Negative Man-Made Climate Change Sentiment')
                        return fig_n

                # Show the bar plot
                st.plotly_chart(n_ha())

                @st.cache(persist = True)
                def ne_ha():
                        # Use the hashtags extract function to get hashtags associated with the neutral class from the original training dataframe
                        neutral_hashtags = hashtag_extract(train['message'][train['sentiment'] == 0])

                        # Create a single list for the neutral sentiment
                        neutral_hashtags = sum(neutral_hashtags, [])

                        # Create a distribution plot of the most frequent hashtags in neutral hashtags list
                        freq = nltk.FreqDist(neutral_hashtags)

                        # Create a dataframe the results of the frequency distribution plot, using the hashtags in the one column and frequencies
                        # in a second
                        df = pd.DataFrame({'Hashtags' : list(freq.keys()),
                                   'Count' : list(freq.values())})
                        # Sort the hashtags by order descending counts and show the first 10 rows i.e the 10 most frequent hashtags
                        df_neu = df.sort_values(by = "Count", ascending = False)
                        df_neu.head(10)

                        # Create a bar plot using the group dataframe  to visualise the number of tweets per class
                        fig_ne = go.Figure(go.Bar(x = df_neu['Hashtags'].head(10), y = df_neu['Count'].head(10).sort_values(ascending = False), # Specify x and y variables
                                       marker = {'color': df_neu['Count'],'colorscale': 'viridis'})) # Select a colour for the graph

                        # Add title, x and y axis labels to the bar chart
                        fig_ne.update_layout(yaxis_title = 'Hashtag Counts', xaxis_title = 'Hashtags', title = 'Count of Hashtags for Neutral Man-Made Climate Change Sentiment')
                        return fig_ne

                # Show the bar plot
                st.plotly_chart(ne_ha())
                
                @st.cache(persist = True)
                def news_ha():
                        # Use the hashtags extract function to get hashtags associated with the news class from the original training dataframe
                        news_hashtags = hashtag_extract(train['message'][train['sentiment'] == 2])
                        # Create a single list for the news sentiment
                        news_hashtags = sum(news_hashtags, [])

                        # Create a distribution plot of the most frequent hashtags in news hashtags list
                        freq = nltk.FreqDist(news_hashtags)

                        # Create a dataframe the results of the frequency distribution plot, using the hashtags in the one column and frequencies
                        # in a second
                        df = pd.DataFrame({'Hashtags' : list(freq.keys()),
                                   'Count' : list(freq.values())})

                        # Sort the hashtags by order descending counts and show the first 10 rows i.e the 10 most frequent hashtags
                        df_news = df.sort_values(by = 'Count', ascending = False)
                        df_news.head(10)

                        # Create a bar plot using the group dataframe  to visualise the number of tweets per class
                        fig_news = go.Figure(go.Bar(x = df_news['Hashtags'].head(10), y = df_news['Count'].head(10), # Specify x and y variables
                                       marker = {'color': df_news['Count'],'colorscale': 'viridis'})) # Select a colour for the graph

                        # Add title, x and y axis labels to the bar chart
                        fig_news.update_layout(yaxis_title = 'Hashtag Counts', xaxis_title = 'Hashtags',  title = 'Count of Hashtags for News Climate Change Sentiment')
                        return fig_news

                # Show the bar plot
                st.plotly_chart(news_ha())
               
        #Load Support Vector Classifier model

        # Create a function that cleans the training data and prepares it for modelling
        def preprocessing(string):
                """This function takes a sentence and transforms it to lowercase using the lower() string method, it then removed urls,
                numerical values, punctuation, and rts (retweets) using regex patterns.  The function also use TweetTokenizer from the
                nltk.tokenize library in order to remove twitter handles
       
                Parameters
                ----------
                string : str
                A sentence string which is to go through text cleaning
           
                Returns
                -------
                str
                A string which has been cleaned of noise"""
    
                # Change the casing in the inputted string to lowercase
                string = string.lower()
    
                # Remove url addresses from the string
                string = re.sub(r"http\S+", "", string)
    
                # Instantiate TweetTokenizer with an argument that allows for the stripping of twitter handles
                tknzr = TweetTokenizer(strip_handles = True)
    
                # Tokenize the string using TweetTokenizer in order to remove twitter handles
                string = tknzr.tokenize(string)
    
                # Join the tokenized words together into sentences 
                string = " ".join(string)
                # Remove punctuation from the string 
                string = re.sub(r'[^a-z0-9\s]', '', string)
                string = re.sub(r'[0-9]+', '', string) # replace numbers or number like words with 'number'
                # Remove rt from the string
                message = re.sub(r'^rt', '', string)
                # Return a new string which has been cleaned of noise
                return message

        st.sidebar.subheader("Models")
        selection = st.sidebar.selectbox("Choose Model", ["Choose Model", "Support Vector Classifier", "Logistic Regression Classifier"])
        if selection == "Support Vector Classifier":
                st.info("Classification with Support Vector Classifier")
                tweet_text = st.text_area("Enter Text", "Type Here")
                tweet_text = preprocessing(tweet_text)
                
                

                if st.button("Classify"):
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        classifier = joblib.load(open(os.path.join("resources/SupportVectorClassifier_model.pkl"), "rb"))
                        prediction = classifier.predict(vect_text)

                        st.text("Text Categorized as: {}".format(prediction))
                        
                        if (prediction == 0):
                                st.warning("This Sentiment is Neutral on Man-Made Climate Change")

                        elif (prediction == -1):
                                st.error("This Sentiment is Anti Man-Made Climate Change")

                        elif (prediction == 1):
                                st.success("This Sentiment is Pro Man-Made Climate Change")

                        elif (prediction == 2):
                                st.success("This Sentiment is News related to Man-Made Climate Change")

                        #st.success("Text Categorized as: {}".format(prediction))

        #Load Logistic Regression Classifier model
        if selection == "Logistic Regression Classifier":
                st.info("Classification with Logistic Regression Classifier")
                tweet_text = st.text_area("Enter Text", "Type Here")
                weet_text = preprocessing(tweet_text)
                
                if st.button("Classify"):
                        vect_text = vect_text = tweet_cv.transform([tweet_text]).toarray()
                        classifier = joblib.load(open(os.path.join("resources/LogisticRegression_model.pkl"), "rb"))
                        prediction = classifier.predict(vect_text)

                        st.text("Text Categorized as: {}".format(prediction))
                        
                        if (prediction == 0):
                                st.warning("This Sentiment is Neutral on Man-Made Climate Change")

                        elif (prediction == -1):
                                st.error("This Sentiment is Anti Man-Made Climate Change")

                        elif (prediction == 1):
                                st.success("This Sentiment is Pro Man-Made Climate Change")

                        elif (prediction == 2):
                                st.success("This Sentiment is News related to Man-Made Climate Change")

        #Create a conclusion label on the sidebar
        st.sidebar.markdown("**Conclusion**")
        conclusion = st.sidebar.checkbox("Conclusion")
        if conclusion:
                
                st.subheader("Conclusion")
                st.markdown("As a business, you would like to know how your actions/marketing campaign/product launch is received by the public. We are living in an era where social media is part of everyone's lives and people are not fearful of to make their feelings known. One of the biggest social media platforms is Twitter. Thus, knowing how people receive your product could give you the edge over your competitors.")
        

# Required to let Streamlit instantiate our web app.  
if __name__ == "__main__":
        main()
