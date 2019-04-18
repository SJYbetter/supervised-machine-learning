import tweepy

auth = tweepy.OAuthHandler('GNBLYoaGVNSV8biVYSpBkJKOB', 'b4qlV2dnGpXvWSmy6u96MYTqMgvJoxkXgxSgJeom4j9LTZcPmd')
auth.set_access_token('949819789-1DhjAAlbkFSDZZQqCRrbkb25iAjSDx8ivPa4MjMD',
                      'GkTPYFmY7J0vrZDSp1LTIK8suCmySXKHOHtj0tTeeyXqj')

# ref: https://developer.twitter.com/en/use-cases/analyze
api = tweepy.API(auth, proxy='socks5://127.0.0.1:998')
