from psaw import PushshiftAPI
from datetime import datetime
import pandas as pd
import os

if os.path.exists('/content/data'):
  pass
else:
  os.mkdir('/content/data')

DATA_ROOT = '/content/data'


api = PushshiftAPI()

start_epoch = int(pd.to_datetime('2021-01-01').timestamp())
end_epoch = int(pd.to_datetime('2021-01-02').timestamp())

gen = api.search_submissions(q='GME',  # this is the keyword (ticker symbol) for which we're searching
                               # these are the unix-based timestamps to search between
                               after=start_epoch, before=end_epoch,
                               # one or more subreddits to include in the search
                               subreddit=['wallstreetbets', 'stocks'],
                               filter=['id', 'url', 'author', 'title', 'score',
                                       'subreddit', 'selftext', 'num_comments'],  # list of fields to return
                               limit=2  # limit on the number of records returned
                             )

lst = list(gen)

print("id:",lst[0].id) # this is Reddit's unique ID for this post
print("url:",lst[0].url) 
print("author:",lst[0].author) 
print("title:",lst[0].title)
print("score:",lst[0].score) # upvote/downvote-based score, doesn't seem 100% reliable
print("subreddit:",lst[0].subreddit)
print("num_comments:",lst[0].num_comments) # number of comments in the thread (which we can get later if we choose)
print("selftext:",lst[0].selftext) # This is the body of the post


def convert_date(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S')

lst[0].d_['datetime_utc'] = convert_date(lst[0].d_['created_utc'])


