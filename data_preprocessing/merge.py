import pandas as pd

#The repositery must have a folder named data contaning all the dataframes
users = pd.read_csv("data/Users.csv")
useractivity = pd.read_csv("data/UserActivity.csv")
compsp = pd.read_csv("data/CompetitionPartipation.csv")
comps = pd.read_csv("data/Competition.csv")
blogs = pd.read_csv("data/Blogs.csv")
comments = pd.read_csv("data/Comments.csv")
discussion = pd.read_csv("data/Discussion.csv")
jobs = pd.read_csv("data/Jobs.csv")

original_months = {2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:1, 9:2, 10:3, 11:4, 12:5, 1:6}
users['Created At Month'] = users['Created At Month'].map(original_months).astype('int')
useractivity['datetime Month'] = useractivity['datetime Month'].map(original_months).astype('int')
compsp['Created At Month'] = compsp['Created At Month'].map(original_months).astype('int')
blogs['Published At Month'] = blogs['Published At Month'].map(original_months).astype('int')
comments['Created At Month'] = comments['Created At Month'].map(original_months).astype('int')
discussion['Created At Month'] = discussion['Created At Month'].map(original_months).astype('int')

users['creation_date'] = users.apply(lambda x : pd.Timestamp(f"{x['Created At Month']}-{x['Created At Day_of_month']}"),axis=1)
useractivity['activity_date'] = useractivity.apply(lambda x : pd.Timestamp(f"{x['datetime Month']}-{x['datetime Day_of_month']}"),axis=1)
compsp['participation_date'] = compsp.apply(lambda x : pd.Timestamp(f"{x['Created At Month']}-{x['Created At Day_of_month']}"),axis=1)
blogs['Publish_date'] = blogs.apply(lambda x : pd.Timestamp(f"{x['Published At Month']}-{x['Published At Day_of_month']}"),axis=1)
comments['comment_date'] = comments.apply(lambda x : pd.Timestamp(f"{x['Created At Month']}-{x['Created At Day_of_month']}"),axis=1)
discussion['discussion_date'] = discussion.apply(lambda x : pd.Timestamp(f"{x['Created At Month']}-{x['Created At Day_of_month']}"),axis=1)

submission_count_keys = compsp['Successful Submission Count'].unique()
submission_count_keys = submission_count_keys[~pd.isna(submission_count_keys)]
#count number of competitions for each user
def get_comp_per_user(row):
    account_creation_date = row['creation_date']
    competitions_per_user = compsp.loc[(row['User_ID'] == compsp['User_ID']) 
                                       & (compsp['participation_date'] >= account_creation_date) 
                                       & (compsp['participation_date'] < account_creation_date + pd.DateOffset(months=1))]
    submission_counts = competitions_per_user['Successful Submission Count'].value_counts()
    submission_counts_list = [int(submission_counts.get(key,0)) for key in submission_count_keys]
    return pd.Series([len(competitions_per_user)]+submission_counts_list)

#count number of comments for each user 
###### Why didnt we use the same way as counting comp taking into consideration the days in the next month same for discussion
def get_comments_per_user(row):
    account_creation_date = row['creation_date']
    comments_per_user = comments.loc[(row['User_ID'] == comments['User_ID']) 
                                       & (comments['comment_date'] >= account_creation_date) 
                                       & (comments['comment_date'] < account_creation_date + pd.DateOffset(months=1))]
    return len(comments_per_user)

#count number of discussions for each user
def get_discussion_per_user(row):
    account_creation_date = row['creation_date']
    discussions_per_user = discussion.loc[(row['User_ID'] == discussion['User_ID']) 
                                       & (discussion['discussion_date'] >= account_creation_date) 
                                       & (discussion['discussion_date'] < account_creation_date + pd.DateOffset(months=1))]
    return len(discussions_per_user)

#split the time to hour, minute, and second
def split_time(row):
    time = pd.Timestamp(row['Created At time'])
    hour = int(time.hour)
    minutes = int(time.minute)
    seconds = int(time.second)
    return pd.Series([hour,minutes,seconds])

#count number of jobs for each user
###### do we want to not count a job if its there twice?
jobs_activity = useractivity[useractivity['Title'].str.startswith('job')]
unique_jobs = jobs_activity.drop_duplicates(subset=['Title','User_ID'])
job_count = unique_jobs.groupby('User_ID')['Title'].nunique()
job_count = pd.Series(job_count)
users['job_count'] = users['User_ID'].map(job_count).fillna(0).astype('int')

#count number of blogs for each user
blogs_activity = useractivity[useractivity['Title'].str.startswith('job')]
unique_blogs = blogs_activity.drop_duplicates(subset=['Title','User_ID'])
blog_count = unique_blogs.groupby('User_ID')['Title'].nunique()
blog_count = pd.Series(blog_count)
users['blog_activity_count'] = users['User_ID'].map(blog_count).fillna(0).astype('int')

#apply the functions above
submission_count_keys_renamed = list(map(lambda x : "subm "+ x,submission_count_keys))
users[['competitons_count'] + submission_count_keys_renamed] = users.apply(get_comp_per_user,axis=1)
users['comments_count'] = users.apply(get_comments_per_user,axis=1)
users['discussions_count'] = users.apply(get_discussion_per_user,axis=1)
users[['hour','minute','second']] = users.apply(split_time,axis=1)

#count the number of active days for each user
###### we need to drop users who signed up on the last month as we dont have targets for them
useractivitymerged = useractivity.merge(users[['User_ID','creation_date']],on='User_ID',how='left')
discussionmerged = discussion.merge(users[['User_ID','creation_date']],on='User_ID',how='left')
commentsmerged = comments.merge(users[['User_ID','creation_date']],on='User_ID',how='left')

useractivitymergedfiltered = useractivitymerged.loc[(useractivitymerged['activity_date'] >= useractivitymerged['creation_date'])
                                                    & (useractivitymerged['activity_date'] < useractivitymerged['creation_date'] + pd.DateOffset(months=1))] 
activity_days = users['User_ID'].map(useractivitymergedfiltered['User_ID'].value_counts()).fillna(0).astype('int')

discussionmergedfiltered = discussionmerged.loc[(discussionmerged['discussion_date'] >= discussionmerged['creation_date'])
                                                    & (discussionmerged['discussion_date'] < discussionmerged['creation_date'] + pd.DateOffset(months=1))] 
discussion_days = users['User_ID'].map(discussionmergedfiltered['User_ID'].value_counts()).fillna(0).astype('int')

commentsmergedfiltered = commentsmerged.loc[(commentsmerged['comment_date'] >= commentsmerged['creation_date'])
                                                    & (commentsmerged['comment_date'] < commentsmerged['creation_date'] + pd.DateOffset(months=1))] 
comments_days = users['User_ID'].map(commentsmergedfiltered['User_ID'].value_counts()).fillna(0).astype('int')

all_activity_days = activity_days + discussion_days + comments_days
users['activity_count'] = all_activity_days

#do we have to count the activity days in the comments and discussions also??
#building the target based on the criteria that each user has done atleast one activity in the next month after account creation
###### remove +1 from starting day of new month as the day it created in in the next month is first day of next month and remove +2 from its cap as it should be just less than that exact day 
useractivitymergedfilteredtarget = useractivitymerged.loc[(useractivitymerged['activity_date'] >= useractivitymerged['creation_date'] + pd.DateOffset(months=1))
                                                    & (useractivitymerged['activity_date'] < useractivitymerged['creation_date'] + pd.DateOffset(months=2))]
useractivitymergedfilteredtarget = users['User_ID'].map(useractivitymergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

discussionmergedfilteredtarget = discussionmerged.loc[(discussionmerged['discussion_date'] >= discussionmerged['creation_date'] + pd.DateOffset(months=1))
                                                    & (discussionmerged['discussion_date'] < discussionmerged['creation_date'] + pd.DateOffset(months=2))]
discussionmergedfilteredtarget = users['User_ID'].map(discussionmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

commentsmergedfilteredtarget = commentsmerged.loc[(commentsmerged['comment_date'] >= commentsmerged['creation_date'] + pd.DateOffset(months=1))
                                                    & (commentsmerged['comment_date'] < commentsmerged['creation_date'] + pd.DateOffset(months=2))]
commentsmergedfilteredtarget = users['User_ID'].map(commentsmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

target = (commentsmergedfilteredtarget) | (discussionmergedfilteredtarget) | (useractivitymergedfilteredtarget)
users['target'] = target

users.to_csv('data/data.csv',index=False)