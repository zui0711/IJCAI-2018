import pandas as pd

def data_split():
    print '********************************************'
    print 'data split'
    print '********************************************\n'

    dtrain = pd.read_csv('feats/dtrain.csv')
    dtest = pd.read_csv('feats/dtest.csv')

    # 18 19 20 21 22 23 || 24
    train1 = dtrain[(dtrain['day'] >= 18) & (dtrain['day'] <= 21)]
    train1['day'] = train1['day'].map(lambda x: x-18)
    test1 = dtrain[dtrain['day'] == 22]
    test1['day'] = test1['day'].map(lambda x: x-18)

    train2 = dtrain[(dtrain['day'] >= 19) & (dtrain['day'] <= 22)]
    train2['day'] = train2['day'].map(lambda x: x-19)
    test2 = dtrain[dtrain['day'] == 23]
    test2['day'] = test2['day'].map(lambda x: x-19)

    train_online = dtrain[(dtrain['day'] >= 20) & (dtrain['day'] <= 23)]
    train_online['day'] = train_online['day'].map(lambda x: x-20)
    test_online = dtest
    test_online['day'] = test_online['day'].map(lambda x: x-20)

    train1.to_csv('feats/train1.csv', columns=train1.columns, index=False)
    train2.to_csv('feats/train2.csv', columns=train2.columns, index=False)
    test1.to_csv('feats/test1.csv', columns=test1.columns, index=False)
    test2.to_csv('feats/test2.csv', columns=test2.columns, index=False)

    train_online.to_csv('feats/train_online.csv', columns=train_online.columns, index=False)
    test_online.to_csv('feats/test_online.csv', columns=test_online.columns, index=False)

if __name__ == '__main__':
    data_split()