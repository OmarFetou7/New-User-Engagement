import pandas as pd
from sklearn.preprocessing import LabelEncoder

#The repositery must have a folder named data contaning all the dataframes
users = pd.read_csv("data/Users.csv")
useractivity = pd.read_csv("data/UserActivity.csv")
compsp = pd.read_csv("data/CompetitionPartipation.csv")
comps = pd.read_csv("data/Competition.csv")
blogs = pd.read_csv("data/Blogs.csv")
comments = pd.read_csv("data/Comments.csv")
discussion = pd.read_csv("data/Discussion.csv")
jobs = pd.read_csv("data/Jobs.csv")
sample = pd.read_csv("data/SampleSubmission.csv")

submission_count_keys = compsp['Successful Submission Count'].unique()
submission_count_keys = submission_count_keys[~pd.isna(submission_count_keys)]
#count number of competitions for each user
def get_comp_per_user(row):
    account_creation_date = row['Created At Month']
    competitions_per_user = compsp.loc[(row['User_ID'] == compsp['User_ID']) 
                                       & (compsp['Created At Month'] == account_creation_date)]
    submission_counts = competitions_per_user['Successful Submission Count'].value_counts()
    submission_counts_list = [int(submission_counts.get(key,0)) for key in submission_count_keys]
    return pd.Series([len(competitions_per_user)]+submission_counts_list)

#count number of comments for each user 
def get_comments_per_user(row):
    account_creation_date = row['Created At Month']
    comments_per_user = comments.loc[(row['User_ID'] == comments['User_ID']) 
                                       & (comments['Created At Month'] == account_creation_date)]
    return len(comments_per_user)

#count number of discussions for each user
def get_discussion_per_user(row):
    account_creation_date = row['Created At Month']
    discussions_per_user = discussion.loc[(row['User_ID'] == discussion['User_ID']) 
                                       & (discussion['Created At Month'] >= account_creation_date)]
    return len(discussions_per_user)

#split the time to hour, minute, and second
def split_time(row):
    time = pd.Timestamp(row['Created At time'])
    hour = int(time.hour)
    minutes = int(time.minute)
    seconds = int(time.second)
    return pd.Series([hour,minutes,seconds])

#count number of jobs for each user
useractivitymerged = useractivity.merge(users[['User_ID','Created At Month']],on='User_ID',how='left')
timemask = (useractivitymerged["datetime Month"] == useractivitymerged["Created At Month"])

jobs_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('job') & (timemask)]
job_count = jobs_activity.groupby('User_ID')['Title'].nunique()
job_count = pd.Series(job_count)
users['job_activity_count'] = users['User_ID'].map(job_count).fillna(0).astype('int')

#count number of compID for each user
comps_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('comp') & (timemask)]
comp_count = comps_activity.groupby('User_ID')['Title'].nunique()
comp_count = pd.Series(comp_count)
users['comp_activity_count'] = users['User_ID'].map(comp_count).fillna(0).astype('int')

#count number of blogs for each user
blogs_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('blog') & (timemask)]
blog_count = blogs_activity.groupby('User_ID')['Title'].nunique()
blog_count = pd.Series(blog_count)
users['blog_activity_count'] = users['User_ID'].map(blog_count).fillna(0).astype('int')

#count number of badges for each user
badges_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('badge') & (timemask)]
badge_count = badges_activity.groupby('User_ID')['Title'].nunique()
badge_count = pd.Series(badge_count)
users['badge_activity_count'] = users['User_ID'].map(badge_count).fillna(0).astype('int')

#label encoding for countries
users['Countries_ID'] = users['Countries_ID'].fillna("unspecified")
le = LabelEncoder()
users['Countries_ID'] = le.fit_transform(users['Countries_ID'])

#count rest of activities for each user
mask_activities =  (~(useractivitymerged['Title'].str.startswith('job')))\
                    & (~(useractivitymerged['Title'].str.startswith('comp')))\
                    & (~(useractivitymerged['Title'].str.startswith('blog')))\
                    & (~(useractivitymerged['Title'].str.startswith('badge')))\
                    & (~(useractivitymerged['Title'].str.startswith('Signed Up')))\
                    & (~(useractivitymerged['Title'].str.startswith('$create_alias')))\
                    & (~(useractivitymerged['Title'].str.startswith('$identify')))\
        
#mask_activities = (useractivitymerged['Title'].value_counts() > 100)
keys = (useractivitymerged['Title'][mask_activities]).unique()
rest_activities = useractivitymerged[useractivitymerged["Title"].isin(keys) & (timemask)]
counts = rest_activities.groupby(["User_ID", "Title"]).size().unstack(fill_value=0)
users = users.merge(counts, on="User_ID", how="left").fillna(0)

#apply the functions above
submission_count_keys_renamed = list(map(lambda x : "subm "+ x,submission_count_keys))
users[['competitons_count'] + submission_count_keys_renamed] = users.apply(get_comp_per_user,axis=1)
users['comments_count'] = users.apply(get_comments_per_user,axis=1)
users['discussions_count'] = users.apply(get_discussion_per_user,axis=1)
users[['hour','minute','second']] = users.apply(split_time,axis=1)

#extract the test ids
test_ids = sample.User_ID_Next_month_Activity.str.replace("_Month_5","")
users_train = users.loc[~(users.User_ID.isin(test_ids))]
users_test = users.loc[(users.User_ID.isin(test_ids))]
users_train = users_train.drop(index=users_train.loc[users_train['Created At Month'] == 5].index)

discussionmerged = discussion.merge(users_train[['User_ID','Created At Month']],on='User_ID',how='left')
commentsmerged = comments.merge(users_train[['User_ID','Created At Month']],on='User_ID',how='left')
useractivitymergedfilteredtarget = useractivitymerged.loc[(useractivitymerged['datetime Month'] == (useractivitymerged['Created At Month'] + 1)%12)]
useractivitymergedfilteredtarget = users_train['User_ID'].map(useractivitymergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

discussionmergedfilteredtarget = discussionmerged.loc[(discussionmerged['Created At Month_x'] == (discussionmerged['Created At Month_y']+1)%12)]
discussionmergedfilteredtarget = users_train['User_ID'].map(discussionmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

commentsmergedfilteredtarget = commentsmerged.loc[(commentsmerged['Created At Month_x'] == (commentsmerged['Created At Month_y']+1)%12)]
commentsmergedfilteredtarget = users_train['User_ID'].map(commentsmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

target = (commentsmergedfilteredtarget) | (discussionmergedfilteredtarget) | (useractivitymergedfilteredtarget)
users_train.loc[:,'target'] = target

users_train.to_csv('data/datatrain.csv',index=False)
users_test.to_csv('data/datatest.csv',index=False)