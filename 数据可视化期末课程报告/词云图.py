import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 停用词加载函数
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return stopwords

# 读取CSV文件并提取第四列
file_paths = [
    "C:/Users/ty01/Desktop/出生人口评论/【深度】主评论_情绪标签.csv",
    "C:/Users/ty01/Desktop/出生人口评论/【深度】二级评论_情绪标签.csv",
    "C:/Users/ty01/Desktop/出生人口评论/中国真实出生率 主评论_情绪标签.csv",
    "C:/Users/ty01/Desktop/出生人口评论/中国真实出生率 二级评论_情绪标签.csv"
]

# 停用词路径
stopwords_path = "stopwords.txt"
stopwords = load_stopwords(stopwords_path)

# 分词和停用词处理函数
def process_text(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return filtered_words

# 提取所有评论文本并进行分词
all_words = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    # 假设第四列是评论文本
    comments = df.iloc[:, 3]  # 如果第四列是评论列
    for comment in comments:
        all_words.extend(process_text(comment, stopwords))

# 统计词频
from collections import Counter
word_counts = Counter(all_words)

# 获取前十个最常见的词
top_words = word_counts.most_common(30)
top_words_dict = dict(top_words)

# 生成词云
wordcloud = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf', width=800, height=400, background_color='white').generate_from_frequencies(top_words_dict)

# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
