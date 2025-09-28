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

submission_count_keys = compsp['Successful Submission Count'].unique()
submission_count_keys = submission_count_keys[~pd.isna(submission_count_keys)]
#count number of competitions for each user
def get_comp_per_user(row):
    account_creation_day = int(row['Created At Day_of_month'])
    account_creation_month = int(row['Created At Month'])
    nextmonth = (account_creation_month + 1) % 12 
    nextday = (account_creation_day + 1) % 31 #Is 31 always right? some months are 30 (check in the future)
    competitions_per_user = compsp.loc[(row['User_ID'] == compsp['User_ID']) 
                                       & (((compsp['Created At Month'] == account_creation_month) 
                                           & (compsp['Created At Day_of_month'] >= account_creation_day)) 
                                           |((compsp['Created At Month'] == nextmonth) 
                                           & (compsp['Created At Day_of_month'] < nextday)))]
    submission_counts = competitions_per_user['Successful Submission Count'].value_counts()
    submission_counts = [int(submission_counts.get(key,0)) for key in submission_count_keys]
    return pd.Series([len(competitions_per_user)]+submission_counts)

#count number of comments for each user
def get_comments_per_user(row):
    account_creation_month = row['Created At Month']
    comments_per_user = comments.loc[(row['User_ID'] == comments['User_ID']) & (comments['Created At Month'] == account_creation_month) ]
    return len(comments_per_user)

#count number of discussions for each user
def get_discussion_per_user(row):
    account_creation_month = row['Created At Month']
    discussions_per_user = discussion.loc[(row['User_ID'] == discussion['User_ID']) & (discussion['Created At Month'] == account_creation_month) ]
    return len(discussions_per_user)

#split the time to hour, minute, and second
def split_time(row):
    time = pd.Timestamp(row['Created At time'])
    hour = int(time.hour)
    minutes = int(time.minute)
    seconds = int(time.second)
    return pd.Series([hour,minutes,seconds])

#count number of jobs for each user
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
submission_count_keys = list(map(lambda x : "subm "+ x,submission_count_keys))
users[['competitons_count'] + submission_count_keys] = users.apply(get_comp_per_user,axis=1)
users['comments_count'] = users.apply(get_comments_per_user,axis=1)
users['discussions_count'] = users.apply(get_discussion_per_user,axis=1)
users[['hour','minute','second']] = users.apply(split_time,axis=1)

#count the number of active days for each user
useractivitymerged = useractivity.merge(users[['User_ID','Created At Month','Created At Day_of_month']],on='User_ID',how='left')
useractivitymergedfiltered = useractivitymerged.loc[((useractivitymerged['datetime Month'] == 
                                                      useractivitymerged['Created At Month'])
                                                    & (useractivitymerged['datetime Day_of_month'] >= 
                                                       useractivitymerged['Created At Day_of_month'])) 
                                                    | ((useractivitymerged['datetime Month'] == 
                                                        (useractivitymerged['Created At Month'] + 1)%12)
                                                    & (useractivitymerged['datetime Day_of_month'] < 
                                                       (useractivitymerged['Created At Day_of_month']+1)%31))] 
                                                    #Is 31 always right? some months are 30 (check in the future)
users['activity_days_count'] = users['User_ID'].map(useractivitymergedfiltered['User_ID'].value_counts()).fillna(0).astype('int')

#building the target based on the criteria that each user has done atleast one activity in the next month after account creation
useractivitymergedfilteredtarget = useractivitymerged.loc[((useractivitymerged['datetime Month'] == 
                                                            (useractivitymerged['Created At Month']+1)%12)
                                                    & (useractivitymerged['datetime Day_of_month'] >= 
                                                       (useractivitymerged['Created At Day_of_month']+1)%31)) 
                                                    | ((useractivitymerged['datetime Month'] == 
                                                        (useractivitymerged['Created At Month'] + 2)%12)
                                                    & (useractivitymerged['datetime Day_of_month'] < 
                                                       (useractivitymerged['Created At Day_of_month']+2)%31))]
useractivitymergedfilteredtarget = users['User_ID'].map(useractivitymergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

discussionmerged = discussion.merge(users[['User_ID','Created At Month','Created At Day_of_month']],on='User_ID',how='left')
discussionmergedfilteredtarget = discussionmerged.loc[((discussionmerged['Created At Month_x'] == 
                                                        (discussionmerged['Created At Month_y']+1)%12)
                                                    & (discussionmerged['Created At Day_of_month_x'] >= 
                                                       (discussionmerged['Created At Day_of_month_y']+1)%31)) 
                                                    | ((discussionmerged['Created At Month_x'] == 
                                                        (discussionmerged['Created At Month_y'] + 2)%12)
                                                    & (discussionmerged['Created At Day_of_month_x'] < 
                                                       (discussionmerged['Created At Day_of_month_y']+2)%31))]
discussionmergedfilteredtarget = users['User_ID'].map(discussionmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

commentsmerged = comments.merge(users[['User_ID','Created At Month','Created At Day_of_month']],on='User_ID',how='left')
commentsmergedfilteredtarget = commentsmerged.loc[((commentsmerged['Created At Month_x'] == 
                                                    (commentsmerged['Created At Month_y']+1)%12)
                                                    & (commentsmerged['Created At Day_of_month_x'] >= 
                                                       (commentsmerged['Created At Day_of_month_y']+1)%31)) 
                                                    | ((commentsmerged['Created At Month_x'] == 
                                                        (commentsmerged['Created At Month_y'] + 2)%12)
                                                    & (commentsmerged['Created At Day_of_month_x'] < 
                                                       (commentsmerged['Created At Day_of_month_y']+2)%31))]
commentsmergedfilteredtarget = users['User_ID'].map(commentsmergedfilteredtarget['User_ID'].value_counts() > 0).fillna(0).astype('int')

target = (commentsmergedfilteredtarget) | (discussionmergedfilteredtarget) | (useractivitymergedfilteredtarget)
users['target'] = target

users.to_csv('data/data.csv',index=False)