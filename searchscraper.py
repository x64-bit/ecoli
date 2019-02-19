from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=480,
                    proxies={'https': 'https://34.203.233.13:80'})

kw_list = ["e coli"]
pytrends.build_payload(
    kw_list=kw_list)
