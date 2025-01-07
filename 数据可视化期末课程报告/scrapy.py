from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from urllib.parse import urlparse, parse_qs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv
import pytz
import datetime
from fake_useragent import UserAgent
import random
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

options = {
    'ignore_http_methods': ['GET', 'POST'],  # 提取XHR请求，通常为GET或POST。如果你不希望忽略任何方法，可以忽略此选项或设置为空数组
    'custom_headers': {
        'X-Requested-With': 'XMLHttpRequest'  # 筛选XHR请求
    }
}

# 配置 Chrome 选项
chrome_options = Options()
chrome_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# 先访问bilibili
driver.get("https://www.bilibili.com")

# 设置cookies
cookies = [
    {
        "name": "SESSDATA",
        "value": "XXXXXXXXXXXXXXXXXXXXX",
        # 替换为你的SESSDATA
        "domain": ".bilibili.com"
    },
    {
        "name": "bili_jct",
        "value": "XXXXXXXXXXXXXXXXXXXXXXXXXXX",  # 替换为你的bili_jct
        "domain": ".bilibili.com"
    },
    {
        "name": "DedeUserID",
        "value": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # 替换为你的DedeUserID
        "domain": ".bilibili.com"
    }
]

# 添加cookies
for cookie in cookies:
    driver.add_cookie(cookie)

# 刷新页面，使cookie生效
driver.refresh()

# 直接访问目标视频页面
driver.get("https://www.bilibili.com/video/BV1xt421P7o4/?spm_id_from=333.337.search-card.all.click&vd_source=d86699fa5e9d33a11a0bee02a61ff84c")

# 等待页面加载完成
time.sleep(3)

# 获取视频标题
try:
    video_title = driver.find_element(By.CSS_SELECTOR, "h1.video-title").text
    # 清理标题中的非法字符，避免文件名出错
    video_title = "".join(char for char in video_title if char not in r'\/:*?"<>|')
except Exception as e:
    print(f"获取视频标题失败: {str(e)}")
    video_title = "未知视频"

# 指定comments文件夹的绝对路径
comments_dir = r'D:\2024_pycharm\2024_Learning\pythonProject\comments'
if not os.path.exists(comments_dir):
    os.makedirs(comments_dir)

# 修改文件路径，使用视频标题
file_path_1 = os.path.join(comments_dir, f'{video_title}_主评论.csv')
file_path_2 = os.path.join(comments_dir, f'{video_title}_二级评论.csv')

# 检查是否已经登录
try:
    # 首先尝试检查是否已经登录（比如查找用户头像或其他登录状态的标识）
    wait = WebDriverWait(driver, 5)
    avatar = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "header-avatar-face")))
    print("已经登录，无需重新登录")
except:
    try:
 
        login_selectors = [
            "header-login-entry",
            "login-entry",
            "nav-user-center > div.unlogin-popover"
        ]

        for selector in login_selectors:
            try:
                login_div = wait.until(EC.presence_of_element_located((By.CLASS_NAME, selector)))
                login_div.click()
                break
            except:
                continue
    except Exception as e:
        print(f"登录过程出错: {str(e)}")

# 获取捕获的网络请求
# 初始化一个变量，用来保存最后一个符合条件的请求
last_request = None

# 遍历所有请求
for request in driver.requests:
    if "main?oid=" in request.url and request.response:
        # 更新last_request为当前请求
        last_request = request

# 检查是否找到了符合条件的请求
if last_request:
    print("URL:", last_request.url)
    # 从URL中提取oid
    parsed_url = urlparse(last_request.url)
    query_params = parse_qs(parsed_url.query)
    oid = query_params.get("oid", [None])[0]
    type = query_params.get("type", [None])[0]
    print("OID:", oid)
    print("type:", type)

    # 从WebDriver中获取所有cookies
    all_cookies = driver.get_cookies()
    cookies_dict = {cookie['name']: cookie['value'] for cookie in all_cookies}
    cookies_str = '; '.join([f"{name}={value}" for name, value in cookies_dict.items()])

    # 从cookies中获取bili_jct的值
    bili_jct = cookies_dict.get('bili_jct', '')
    print("bili_jct:", bili_jct)
    sessdata = cookies_dict.get('SESSDATA', '')
    print("SESSDATA:", sessdata)
    # 打印请求头
    response = last_request.response

driver.quit()

# 重试次数限制
MAX_RETRIES = 5
# 重试间隔（秒）
RETRY_INTERVAL = 10

# 创建CSV文件并写入表头
with open(file_path_1, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['昵称', '性别', '时间', '点赞', '评论', 'IP属地', '二级评论条数', '等级', 'uid', 'rpid'])

with open(file_path_2, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['昵称', '性别', '时间', '点赅', '评论', 'IP属地', '二级评论条数,条数相同说明在同一个人下面', '等级', 'uid', 'rpid'])

beijing_tz = pytz.timezone('Asia/Shanghai')  # 时间戳转换为北京时间
ua = UserAgent()  # 创立随机请求头

ps = 20

down = 1  # 开始爬页数a
up = 30  # 结束爬的页数

one_comments = []
all_comments = []  # 构造数据放在一起的容器  总共评论，如果只希望含有一级评论，请注释 line 144
all_2_comments = []  # 构造数据放在一起的容器 二级评论
comments_current = []
comments_current_2 = []

with requests.Session() as session:
    retries = Retry(total=3,  # 最大重试次数，好像没有这个函数
                    backoff_factor=0.1,  # 间隔时间会乘以这个数
                    status_forcelist=[500, 502, 503, 504])

    for page in range(down, up + 1):
        for retry in range(MAX_RETRIES):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    # 'ua.random',#随机请求头，b站爸爸别杀我，赛博佛祖保佑
                    'Cookie': cookies_str,
                    'SESSDATA': sessdata,
                    'csrf': bili_jct,
                }
                url = 'https://api.bilibili.com/x/v2/reply?'  # 正常api，只能爬8k
                url_long = 'https://api.bilibili.com/x/v2/reply/main'  # 懒加载api，理论无上限
                url_reply = 'https://api.bilibili.com/x/v2/reply/reply'  # 评论区回复api
                # 示例：https://api.bilibili.com/x/v2/reply/main?next=1&type=1&oid=544588138&mode=3（可访问网站）
                data = {
                    'next': str(page),  # 页数，需要转换为字符串，与pn同理，使用懒加载api
                    'type': type,  # 类型 11个人动态 17转发动态 视频1）
                    'oid': oid,  # id，视频为av，文字动态地址栏id，可自查
                    'ps': ps,  # (每页含有条数，不能大于20)用long话不能大于30
                    'mode': '3'  # 3为热度       0 3：仅按热度      1：按热+按时间 2：仅按时间 使用懒加载api
                }
                proxies = {
                    # "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel},
                    # "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel}
                    # 代理ip来源：https://www.kuaidaili.com/free/inha/
                }
                prep = session.prepare_request(requests.Request('GET', url_long, params=data, headers=headers))
                print(prep.url)
                response = session.get(url_long, params=data, headers=headers)
                # 检查响应状态码是否为200，即成功
                if response.status_code == 200:
                    json_data = response.json()  # 获得json数据
                    if 'data' in json_data and 'replies' in json_data['data']:  # 以下为核心内容，爬取的数据
                        for comment in json_data['data']['replies']:
                            # one_comments.clear()
                            count = comment['rcount']
                            rpid = str(comment['rpid'])
                            name = comment['member']['uname']
                            sex = comment['member']['sex']
                            ctime = comment['ctime']
                            dt_object = datetime.datetime.fromtimestamp(ctime, datetime.timezone.utc)
                            formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S') + ' 北京时间'  # 可以加上时区信息，但通常不需要
                            like = comment['like']
                            message = comment['content']['message'].replace('\n', ',')
                            # 检查是否存在 location 字段
                            location = comment['reply_control'].get('location', '未知')  # 如果不存在，使用 '未知'
                            location = location.replace('IP属地：', '') if location else location
                            # 将提取的信息追加到列表中
                            current_level = comment['member']['level_info']['current_level']
                            mid = str(comment['member']['mid'])
                            all_comments.append(
                                [name, sex, formatted_time, like, message, location, count, current_level, mid, rpid])
                            comments_current.append(
                                [name, sex, formatted_time, like, message, location, count, current_level, mid, rpid])

                            # 在获取到评论数据后立即写入CSV
                            if comments_current:  # 如果有主评论数据
                                with open(file_path_1, mode='a', newline='', encoding='utf-8-sig') as file:
                                    writer = csv.writer(file)
                                    writer.writerows(comments_current)
                                comments_current.clear()  # 清空临时存储
                                
                            if comments_current_2:  # 如果有二级评论数据
                                with open(file_path_2, mode='a', newline='', encoding='utf-8-sig') as file:
                                    writer = csv.writer(file)
                                    writer.writerows(comments_current_2)
                                comments_current_2.clear()  # 清空临时存储

                            if (count != 0):
                                print(f"在第{page}页中含有二级评论,该条回复下面总共含有{count}个二级评论")
                                total_pages = ((count // 20) + 2) if count > 0 else 0
                                for page_pn in range(total_pages):
                                    data_2 = {
                                        # 二级评论的data
                                        'type': type,  # 类型
                                        'oid': oid,  # id
                                        'ps': ps,  # 每页含有条数，不能大于20
                                        'pn': str(page_pn),  # 二级评论页数，需要转换为字符串
                                        'root': rpid  # 一级评论的rpid
                                    }
                                    if page_pn == 0:
                                        continue
                                    response = session.get(url_reply, params=data_2, headers=headers, proxies=proxies)
                                    prep = session.prepare_request(
                                        requests.Request('GET', url_reply, params=data_2, headers=headers))
                                    print(prep.url)

                                    if response.status_code == 200:
                                        json_data = response.json()  # 获得json数据
                                        if 'data' in json_data and 'replies' in json_data['data']:
                                            if not json_data['data']['replies']:  # 检查replies是否为空，如果为空，跳过这一页
                                                print(f"该页replies为空，没有评论")
                                                continue
                                            for comment in json_data['data']['replies']:
                                                rpid = str(comment['rpid'])
                                                name = comment['member']['uname']
                                                sex = comment['member']['sex']
                                                ctime = comment['ctime']
                                                dt_object = datetime.datetime.fromtimestamp(ctime,
                                                                                            datetime.timezone.utc)
                                                formatted_time = dt_object.strftime(
                                                    '%Y-%m-%d %H:%M:%S') + ' 北京时间'  # 可以加上时区信息，但通常不需要
                                                like = comment['like']
                                                message = comment['content']['message'].replace('\n', ',')
                                                # 检查是否存在 location 字段
                                                location = comment['reply_control'].get('location',
                                                                                        '未知')  # 如果不存在，使用 '未知'
                                                location = location.replace('IP属地：', '') if location else location
                                                current_level = comment['member']['level_info']['current_level']
                                                mid = str(comment['member']['mid'])
                                                all_2_comments.append(
                                                    [name, sex, formatted_time, like, message, location, count,
                                                     current_level, mid, rpid])
                                                comments_current_2.append(
                                                    [name, sex, formatted_time, like, message, location, count,
                                                     current_level, mid, rpid])
                                                with open(file_path_2, mode='a', newline='',
                                                          encoding='utf-8-sig') as file:  # 二级评论条数
                                                    writer = csv.writer(file)
                                                    writer.writerows(all_2_comments)
                                                all_2_comments.clear()
                                        else:
                                            # print(f"在第{page_pn + 1}页的JSON响应中缺少 'data' 或 'replies' 键。跳过此页。")
                                            print(f"在页面{page}下第{page_pn + 1}条评论没有子评论。")
                                    else:
                                        print(f"获取第{page_pn + 1}页失败。状态码: {response.status_code}")
                                random_number = random.uniform(0.2, 0.3)
                                time.sleep(random_number)
                        print(f"已经爬取第{page}页. 状态码: {response.status_code} ")
                    else:
                        print(f"在页面 {page} 的JSON响应中缺少 'data' 或 'replies' 键。跳过此页。")
                else:
                    print(f"获取页面 {page} 失败状态码: {response.status_code} 即为失败，请分析原因并尝试重试")

                random_number = random.uniform(0.2, 0.3)
                print(random_number)
                time.sleep(random_number)
                break
            except requests.exceptions.RequestException as e:
                print(f"连接失败: {e}")
                if retry < MAX_RETRIES - 1:
                    print(f"正在重试（剩余尝试次数：{MAX_RETRIES - retry - 1}）...")
                    time.sleep(RETRY_INTERVAL)  # 等待一段时间后重试
                else:
                    raise  # 如果达到最大重试次数，则抛出原始异常
