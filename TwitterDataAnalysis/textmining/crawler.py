import tweepy
import json
import os.path
import pymongo as mgo
import pickle
import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

auth0 = tweepy.OAuthHandler('GNBLYoaGVNSV8biVYSpBkJKOB', 'b4qlV2dnGpXvWSmy6u96MYTqMgvJoxkXgxSgJeom4j9LTZcPmd')
auth0.set_access_token('949819789-1DhjAAlbkFSDZZQqCRrbkb25iAjSDx8ivPa4MjMD',
                       'GkTPYFmY7J0vrZDSp1LTIK8suCmySXKHOHtj0tTeeyXqj')

auth1 = tweepy.OAuthHandler('pQAh4G7BcXgAhufW1ntpCT31d', 'zNSWOwzvxHvMz4UpxlWDOTfwT32wSxz1RR93CicFCVWf3kTuy6')
auth1.set_access_token('949819789-XU9BwDZzelAQPphkNXfuzxWapcBFmAnair0Hcpdm',
                       'zErs4VGhf3zTOT3JGZ1gi8RQb7102IdEWSibFjbvnu5IZ')

auth2 = tweepy.OAuthHandler('IbCeIYSSJTKb752XEtLZAmGyH', 'CzXk63Zn2SWc9wNPmUdZiO2Y1jC5DYPW3C7edRlbpuW8b5Pe65')
auth2.set_access_token('966033852056657920-H2SFX3T0dpG5WMWY5DPGg6WgPLDOIFB',
                       'KRtxb8RTCHFADJM459UImncAHrktBlLt2xOOnq1mq2ZoE')

auth3 = tweepy.OAuthHandler('YoJMBKTvBo0ae8Rx97Mm0XzG1', '9RH3ZvnDKkv2oC7Tq57J0ilHUeghwnNrLdANNH8KsDApztcbeH')
auth3.set_access_token('949819789-4fRdAd4d6vrlAPN1IyosTOHivQoX6V0Gag9jF87I',
                       'DXONsq8LIAoZbNH5iq1rzXQFfQpLsv4PNLVdLSDTYtNoV')


authList = [auth0, auth1, auth2, auth3]

# api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#    print(tweet.text)

# public_tweets = api.search(q="svm", lang='en')
# for tweet in public_tweets:
#     print(json.dumps(tweet._json))
#     print()


# users = api.search_users('Vani Mandava')
# for user in users:
#     print(user)


# geos = api.reverse_geocode(lat=37.7821120598956, long=-122.400612831116)
# for i in geos:
#     print(i)

# class DiscoverUsers:
#     def __init__(self):
#         pass
#
#     def find(self):
#         users = api.search_users(q='place:fbd6d2f5a4e4a15e svm')
#         with open('users.bin', 'wb') as fp:
#             pickle.dump(users, fp)
#
#
# class Search:
#     def __init__(self):
#         pass
#
#     def show(self):
#         public_status = api.search('place:fbd6d2f5a4e4a15e svm')
#         for i in public_status:
#             print(i.text)
#

if __name__ == "__main__":
    from stream import StatusCollector, create_option
    import optparse

    opt = optparse.OptionParser()
    opt.add_option('-a', '--auth', type=int, default=0, metavar='index',
                   help="which auth key to use, range 0 ~ %d" % len(authList))
    opt.add_option_group(create_option(opt))

    opt, left = opt.parse_args()
    print(opt)
    print('simple', opt.simple)
    print('filter', opt.filter)
    print('lang', opt.lang)
    print('locations', opt.locations)
    print('output', opt.output)
    # DiscoverUsers().find()

    # ref: https://developer.twitter.com/en/use-cases/analyze
    # api = tweepy.API(auth, proxy='socks5://127.0.0.1:998')
    api = tweepy.API(authList[opt.auth])

    StatusCollector(opt, api).collect()
    # a = GeoData()
    # a.load()
