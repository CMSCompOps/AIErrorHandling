import json
import re
import datetime
import pandas as pd

r = re.compile( '.*_(?P<date>[0-9][0-9][0-9][0-9][0-9][0-9])_(?P<time>[0-9][0-9][0-9][0-9][0-9][0-9])_.*')
with open('all2') as f:
    j = json.load(f)

nUsersPerMonth = {}
nActionsPerMonth = {}
allUsers = []
allActions = []
for a,b in j.items():
    match = r.match( a )
    if not match:
        print( a )
    else:
        date = match.groupdict()['date']
        time = match.groupdict()['time']

        dt = datetime.date( 2000+int(date[0:2]),
                            int(date[2:4]),
                            15 )
        #                    int(date[4:6]) )
        #int(time[0:2]),
        #                       int(time[2:4]),
        #                       int(time[4:6]) )
        aieh = 'AIEH' in b['Reasons']
        hasuser =  b['user'] if 'user' in b.keys() else 'Unknown'
        user = 'Auto' if aieh else hasuser
        action = b['Action']

        #if user=='Unknown':
        #    print( b)
        #print('{0},{1},{2}'.format( dt.strftime('%m/%d/%Y') , user , action ) )
        if not dt in nUsersPerMonth:
            nUsersPerMonth[dt] = {}
            nActionsPerMonth[dt] = {}
        if not user in nUsersPerMonth[dt]:
            nUsersPerMonth[dt][user] = 0
        nUsersPerMonth[dt][user] += 1

        if not action in nActionsPerMonth[dt]:
            nActionsPerMonth[dt][action] = 0
        nActionsPerMonth[dt][action] += 1

        allUsers.append( user )
        allActions.append( action )

allmonths = sorted(list(nUsersPerMonth.keys()))        
allUsers = set(allUsers)

def constructUserCount(user):
    ret = []
    for m in allmonths:
        if user in nUsersPerMonth[m]:
            ret.append( nUsersPerMonth[m][user] )
        else:
            ret.append( 0 )
    return ret
dictUsers = { u:constructUserCount(u) for u in allUsers }
dictUsers['dates'] = allmonths
tblUsersPerMonth = pd.DataFrame( dictUsers )

allActions = set(allActions)
def constructActionCount(action):
    ret = []
    for m in allmonths:
        if action in nActionsPerMonth[m]:
            ret.append( nActionsPerMonth[m][action] )
        else:
            ret.append( 0 )
    return ret
dictActions = {a:constructActionCount(a) for a in allActions}
dictActions['dates'] = allmonths
tblActionsPerMonth = pd.DataFrame(dictActions)
#print( tblActionsPerMonth)
#print(tblUsersPerMonth)

#tblUsersPerMonth.to_csv('user.csv')
#tblActionsPerMonth.to_csv('actions.csv')
import plotly.express as px
#import plotly.graph_objects as go

fig = px.area(tblUsersPerMonth, x="dates", y=tblUsersPerMonth.columns,
              hover_data={"dates": "|%B %d, %Y"},title='custom tick labels')
fig.update_layout(
    title="Console usage over time",
    xaxis_title="Date",
    yaxis_title="Number of submitted workflows",
    legend_title="User",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig.show()

print( tblUsersPerMonth )
