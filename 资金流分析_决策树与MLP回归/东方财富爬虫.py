import requests
import csv
import json
from tqdm import tqdm

def get_html(page):
    cookies = {
        'qgqp_b_id': '203515e1156840777be2ccfed7b155ed',
        'st_si': '97180606255487',
        'st_asi': 'delete',
        'st_pvi': '78627694871677',
        'st_sp': '2023-04-01%2012%3A08%3A11',
        'st_inirUrl': '',
        'st_sn': '3',
        'st_psi': '202304011214128-113300300813-3572248660',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Connection': 'keep-alive',
        # 'Cookie': 'qgqp_b_id=203515e1156840777be2ccfed7b155ed; st_si=97180606255487; st_asi=delete; st_pvi=78627694871677; st_sp=2023-04-01%2012%3A08%3A11; st_inirUrl=; st_sn=3; st_psi=202304011214128-113300300813-3572248660',
        'Referer': 'https://data.eastmoney.com/',
        'Sec-Fetch-Dest': 'script',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54',
        'sec-ch-ua': '"Microsoft Edge";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }
    url = f'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112305600646902036972_1680323354687&fid=f62&po=1&pz=50&pn={page}&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5&fs=m%3A0%2Bt%3A6%2Bf%3A!2%2Cm%3A0%2Bt%3A13%2Bf%3A!2%2Cm%3A0%2Bt%3A80%2Bf%3A!2%2Cm%3A1%2Bt%3A2%2Bf%3A!2%2Cm%3A1%2Bt%3A23%2Bf%3A!2%2Cm%3A0%2Bt%3A7%2Bf%3A!2%2Cm%3A1%2Bt%3A3%2Bf%3A!2&fields=f12%2Cf14%2Cf2%2Cf3%2Cf62%2Cf184%2Cf66%2Cf69%2Cf72%2Cf75%2Cf78%2Cf81%2Cf84%2Cf87%2Cf204%2Cf205%2Cf124%2Cf1%2Cf13'
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r


def preprocess(res):
    js = json.loads(res.text[42:-2])
    stock_data = []
    for data in js['data']['diff']:
        data_output = []
        data_output.append(data['f12'])
        data_output.append(data['f14'])
        data_output.append(data['f2'])
        data_output.append(data['f3'])
        data_output.append(data['f62'])
        data_output.append(data['f184'])
        data_output.append(data['f66'])
        data_output.append(data['f69'])
        data_output.append(data['f72'])
        data_output.append(data['f75'])
        data_output.append(data['f78'])
        data_output.append(data['f81'])
        data_output.append(data['f84'])
        data_output.append(data['f87'])
        data_dict = {'股票代码': data_output[0], '股票名称': data_output[1], '最新价': data_output[2], '今日涨跌额': data_output[3],
                     '今日主力净流入净额': data_output[4], '今日主力净流入净占比': data_output[5], '今日超大单净流入净额': data_output[6],
                     '今日超大单净流入净占比': data_output[7], '今日大单净流入净额': data_output[8], '今日大单净流入净占比': data_output[9],
                     '今日中单净流入净额': data_output[10], '今日中单净流入净占比': data_output[11], '今日小单净流入净额': data_output[12],
                     '今日小单净流入净占比': data_output[13]}
        stock_data.append(data_dict)
    return stock_data


def saveStockList(s):
    with open('东方财富个股资金流.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f,
                                fieldnames=['股票代码', '股票名称', '最新价', '今日涨跌额', '今日主力净流入净额', '今日主力净流入净占比',
                                            '今日超大单净流入净额', '今日超大单净流入净占比', '今日大单净流入净额', '今日大单净流入净占比',
                                            '今日中单净流入净额', '今日中单净流入净占比', '今日小单净流入净额', '今日小单净流入净占比'])
        # writer.writerow()
        writer.writeheader()
        for data in s:
            writer.writerows(data)


li = []
with tqdm (total=102) as bar:
    for page in range(1, 103):
        s = get_html(page)
        li.append(preprocess(s))
        bar.set_description_str(f"正在爬取第{page}页")
        bar.update(1)
saveStockList(li)