import twitter
import json
from collections import Counter
import datetime
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import time
import twython
import string



def company_search(company):
    CONSUMER_KEY = 'fH4YFq25oK61JwakuaJ5g'
    CONSUMER_SECRET = 'S8v2bm0y8jPy3oIsJl8QdZtx6BnDtbkiN2ANK65ZLM'
    OAUTH_TOKEN = '21964998-aeEYdcIHsmaKMrjBM4wqMqpFLlJ8Npy002DepKYsa'
    OAUTH_TOKEN_SECRET = 'fZa21ALNIBiWetskCIuaywLro05EwgG2VjgaczpbRawjB'
    
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    search_results = twitter_api.search.tweets(q=company,count=10000)
    date_status = [(datetime.datetime.strptime(re.sub('\+0000 ','',status['created_at']),'%a %b %d %H:%M:%S %Y').date(),status['text']) for status in search_results['statuses']]
    date_string_dict = {}
    for date,text in date_status:
        if date in date_string_dict:
            date_string_dict[date] = date_string_dict[date]+text
        else:
            date_string_dict[date]=text
    vectorizer = CountVectorizer(min_df=0)
    vectorizer.fit(date_string_dict.values())
    bag_matrix = vectorizer.transform(date_string_dict.values())
    bag_matrix=sparse.csc_matrix(bag_matrix)
    #type(bag_matrix)
    #bag_matrix.toarray()
    return date_string_dict,bag_matrix

def company_search_2(company,onedate):
    CONSUMER_KEY = 'fH4YFq25oK61JwakuaJ5g'
    CONSUMER_SECRET = 'S8v2bm0y8jPy3oIsJl8QdZtx6BnDtbkiN2ANK65ZLM'
    OAUTH_TOKEN = '21964998-aeEYdcIHsmaKMrjBM4wqMqpFLlJ8Npy002DepKYsa'
    OAUTH_TOKEN_SECRET = 'fZa21ALNIBiWetskCIuaywLro05EwgG2VjgaczpbRawjB'
    
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    q="$"+company
    search_results = twitter_api.search.tweets(q=q,count=10000,until=(onedate+datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    date_status = [(datetime.datetime.strptime(re.sub('\+0000 ','',status['created_at']),'%a %b %d %H:%M:%S %Y').date(),status['text']) for status in search_results['statuses']]
    date_string_dict = {}
    for date,text in date_status:
        if date in date_string_dict:
            date_string_dict[date] = date_string_dict[date]+text
        else:
            date_string_dict[date]=text
    #vectorizer = CountVectorizer(min_df=0)
    #vectorizer.fit(date_string_dict.values())
    #bag_matrix = vectorizer.transform(date_string_dict.values())
    #bag_matrix=sparse.csc_matrix(bag_matrix)
    #return date_string_dict,date_status
    

    #just return a single date
    #return date_string_dict[onedate]
    #return as dataframe
    return pd.DataFrame({'company':company,'date':onedate,'text':date_string_dict[onedate]},index=[company],columns=['company','date','text'])


def multisearch(companylist,datelist):
    framelist=[]
    for company in companylist:
        for date in datelist:
            framelist.append(company_search_2(company,date))
    return pd.concat(framelist)

def safe_company_search(company,onedate,CONSUMER_KEY,CONSUMER_SECRET,OAUTH_TOKEN,OAUTH_TOKEN_SECRET):
    #CONSUMER_KEY = 'fH4YFq25oK61JwakuaJ5g'
    #CONSUMER_SECRET = 'S8v2bm0y8jPy3oIsJl8QdZtx6BnDtbkiN2ANK65ZLM'
    #OAUTH_TOKEN = '21964998-aeEYdcIHsmaKMrjBM4wqMqpFLlJ8Npy002DepKYsa'
    #OAUTH_TOKEN_SECRET = 'fZa21ALNIBiWetskCIuaywLro05EwgG2VjgaczpbRawjB'
    
    #auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,CONSUMER_KEY, CONSUMER_SECRET)
    #twitter_api = twitter.Twitter(auth=auth)

    twy_api = twython.Twython(CONSUMER_KEY,CONSUMER_SECRET,OAUTH_TOKEN,OAUTH_TOKEN_SECRET)

    q="$"+company
    #search_results = twitter_api.search.tweets(q=q,count=10000,until=(onedate+datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    search_results = twy_api.search(q=q,count=10000,until=(onedate+datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    date_status = [(datetime.datetime.strptime(re.sub('\+0000 ','',status['created_at']),'%a %b %d %H:%M:%S %Y').date(),status['text']) for status in search_results['statuses']]
    date_string_dict = {}
    for date,text in date_status:
        if date in date_string_dict:
            date_string_dict[date] = date_string_dict[date]+text
        else:
            date_string_dict[date]=text
    remaining =  twy_api.get_lastfunction_header('x-rate-limit-remaining')

    if onedate in date_string_dict:
        alltext = date_string_dict[onedate]
        alltext = alltext.encode('ascii','ignore')
        alltext = alltext.translate(string.maketrans("",""),string.punctuation)
        return (True,pd.DataFrame({'company':company,'date':str(onedate),'text':alltext},index=[company],columns=['company','date','text']),remaining)
    else:
        return (False,0,remaining)

def safemultisearch(companylist,datelist):
    ckey_m = 'fH4YFq25oK61JwakuaJ5g'
    csec_m = 'S8v2bm0y8jPy3oIsJl8QdZtx6BnDtbkiN2ANK65ZLM'
    oauth_t_m = '21964998-aeEYdcIHsmaKMrjBM4wqMqpFLlJ8Npy002DepKYsa'
    oauth_t_s_m = 'fZa21ALNIBiWetskCIuaywLro05EwgG2VjgaczpbRawjB'

    ckey_d = 'JlEwT1JOCM9mLaXHkrMw'
    csec_d = 'LIfJ1SX1KbXcSNWvEJsKsWoHRsQj879D5bVastWc'
    oauth_t_d = '540657519-mhy3CCq0zPEmBmBp01WySVw65J4JcIazW9NrPLAf'
    oauth_t_s_d = '26BABqEa8ON3dI6Sj0FU61cFL5EIVLNBli1qu448FaRfF'

    ckey_k = 'RfStMUnZSI8rZXb9SRvxQ'
    csec_k = 'zSAt3nvk9nmKtNnwO0JHiSZXltvCsJsJrEqlH2urvY'
    oauth_t_k = '2241491688-M0Cp1fCSOBOGEIHcelV62Cppmi71XaEPcw2Tyc1'
    oauth_t_s_k = 'Mvm3DWiMqcDoCaz7JMQs8BuMQfnsdzGbgDJVypnhl844U'

    ckey_list = [ckey_m,ckey_d,ckey_k]
    csec_list = [csec_m,csec_d,csec_k]
    oauth_t_list = [oauth_t_m,oauth_t_d,oauth_t_k]
    oauth_t_s_list = [oauth_t_s_m,oauth_t_s_d,oauth_t_s_k]
    
    fullframe_pieces = []
    
    counter = 0
    remaining = 30
    index = 0
    for company in companylist:
        for date in datelist:
            
            notcomplete = True
            errorCounter = 1
            emptyflag = False
            #print counter
            if (counter % 10 )==0:
                print "Current Count is",counter
                print index,remaining
            while notcomplete:
                #print counter % 3,remaining,"the first"
                #print type(remaining)
                #print remaining<25
                ##remaining = 15
                if remaining != None:
                    try:
                        intremain = int(remaining)
                    except ValueError:
                        intremain = 3
                        print "Failed to convert remaining calls, waiting one minute"
                else:
                    intremain = 3
                    print "Remaining calls is Nonetype, waiting one minute"
                if intremain<25:
                    print "Rate Limit Approaching, wait one minute"
                    print "License",counter % 3,"Remaining",remaining
                    time.sleep(60)

                try:
                    index = counter % 3
                    (emptyFlag,oneframe,remaining) = safe_company_search(company,date,ckey_list[index],csec_list[index],oauth_t_list[index],oauth_t_s_list[index])
                    #print index,remaining
                    notcomplete = False
                    #time.sleep(.2)
                except ValueError:
                    if errorCounter<4:
                        delay = errorCounter*5
                    else:
                        delay = 60
                    print "Error number",errorCounter,"waiting",delay,"seconds"
                    if delay==60:
                        print "Error.  Company:",company,"Date",str(date)
                    time.sleep(delay)
                if emptyFlag == True:
                    fullframe_pieces.append(oneframe)
                counter += 1
    return pd.concat(fullframe_pieces)
