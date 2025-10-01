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

original_months = {2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:1, 9:2, 10:3, 11:4, 12:5, 1:6}
users['Original Month'] = users['Created At Month'].map(original_months).astype('int')
users['creation_date'] = users.apply(lambda x : pd.Timestamp(year=x['Created At Year'],month=x['Original Month'],day=x['Created At Day_of_month']),axis=1)
users['days_left_in_month'] = users.apply(lambda x : ((x['creation_date'] + pd.offsets.MonthEnd(0)) - x['creation_date']).days,axis=1)
useractivity['Original Month'] = useractivity['datetime Month'].map(original_months).astype('int')
useractivity['activity_date'] = useractivity.apply(lambda x : pd.Timestamp(year=2020,month=x['Original Month'],day=x['datetime Day_of_month']) + pd.to_timedelta(x['datetime time']),axis=1)


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


useractivitymerged = useractivity.merge(users[['User_ID','Created At Month']],on='User_ID',how='left')
timemask = (useractivitymerged["datetime Month"] == useractivitymerged["Created At Month"])

#count number of times a user visited the website and std
std = useractivitymerged[timemask].groupby("User_ID")['activity_date'].std().dt.days
count = useractivitymerged[timemask].groupby("User_ID")['Title'].count()
count_days = useractivitymerged[timemask].groupby("User_ID")['datetime Day_of_month'].nunique()
count_days_std = useractivitymerged[timemask].groupby("User_ID")['datetime Day_of_month'].unique().apply(lambda x: x.std())
# users['visit_std_days'] = users['User_ID'].map(count_days_std).fillna(0).astype('float')
# users['visit_count_days'] = users['User_ID'].map(count_days).fillna(0).astype('int')
users['visit_std'] = users['User_ID'].map(std).fillna(0).astype('float')
users['visit_count'] = users['User_ID'].map(count).fillna(0).astype('int')

#count number of jobs for each user
jobs_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('job') & (timemask)]
job_count = jobs_activity.groupby('User_ID')['Title'].count()
job_count = pd.Series(job_count)
users['job_activity_count'] = users['User_ID'].map(job_count).fillna(0).astype('int')

#count number of compID for each user
comps_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('comp') & (timemask)]
comp_count = comps_activity.groupby('User_ID')['Title'].count()
comp_count = pd.Series(comp_count)
users['comp_activity_count'] = users['User_ID'].map(comp_count).fillna(0).astype('int')

#count number of blogs for each user
blogs_activity = useractivitymerged[useractivitymerged['Title'].str.startswith('blog') & (timemask)]
blog_count = blogs_activity.groupby('User_ID')['Title'].count()
blog_count = pd.Series(blog_count)
users['blog_activity_count'] = users['User_ID'].map(blog_count).fillna(0).astype('int')

#count number of badges for each user
badge_df = useractivitymerged[useractivitymerged['Title'].str.startswith('badge')]
badge_count = badge_df.pivot_table(
    index='User_ID',
    columns='Title',
    aggfunc='size',
    fill_value=0
    ).reset_index()
users = users.merge(badge_count, on='User_ID', how='left')

#label encoding for countries
users['Countries_ID'] = users['Countries_ID'].fillna("unspecified")
le = LabelEncoder()
users['Countries_ID'] = le.fit_transform(users['Countries_ID'])

#count rest of activities for each user
mask_activities =  (~(useractivitymerged['Title'].str.startswith('job')))\
                    & (~(useractivitymerged['Title'].str.startswith('comp')))\
                    & (~(useractivitymerged['Title'].str.startswith('blog')))\
                    & (~(useractivitymerged['Title'].str.startswith('badge')))\
                    #& (~(useractivitymerged['Title'].str.startswith('Signed Up')))\
                    #& (~(useractivitymerged['Title'].str.startswith('$create_alias')))\
                    #& (~(useractivitymerged['Title'].str.startswith('$identify')))\
        
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

#extract the test ids
test_ids = sample.User_ID_Next_month_Activity.str.replace("_Month_5","")
users = users.drop(columns=['creation_date','Original Month'])
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
target2 = (useractivitymergedfilteredtarget)
# print(target.value_counts())
# print(target2.value_counts())
users_train.loc[:,'target'] = target2

users_train.to_csv('data/datatrain.csv',index=False)
users_test.to_csv('data/datatest.csv',index=False)