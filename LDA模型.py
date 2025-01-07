import pandas as pd
from gensim import corpora, models
from nltk.tokenize import word_tokenize
import os
import nltk
nltk.download('punkt_tab')

# 文件路径
file_path = 'C:/Users/ty01/Desktop/出生人口评论/【深度】我国出生人口骤降！为什么还不出台大规模“救生”政策？_二级评论.csv'
output_path = 'C:/Users/ty01/Desktop/出生人口评论/【深度】二级评论_情绪标签.csv'

# 读取数据
data = pd.read_csv(file_path)

# 提取评论列和时间列和性别列
comments = data.iloc[:, 4].astype(str)  # 第五列是评论
genders = data.iloc[:, 1].astype(str)  # 第1列是性别
times = data.iloc[:, 2].astype(str)    # 第三列是时间

# 文本预处理（分词）
texts = [word_tokenize(comment.lower()) for comment in comments]

# 构建词典和语料库
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 训练 LDA 模型（3个主题）
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# 查看主题关键词并人工标注情绪倾向
topics = lda_model.print_topics(num_words=10)
print("主题关键词:")
for idx, topic in enumerate(topics):
    print(f"Topic {idx}: {topic}")

# 假设：根据关键词标注主题情绪标签
# Topic 0 -> Positive, Topic 1 -> Negative, Topic 2 -> Neutral
topic_sentiment_map = {
    0: "正面",
    1: "负面",
    2: "中立"
}

# 文档主题分布并分类情绪标签
def classify_sentiment(doc_topics):
    dominant_topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0]  # 获取主导主题
    return topic_sentiment_map[dominant_topic[0]]

# 为每条评论预测情绪标签
sentiments = [classify_sentiment(lda_model[doc]) for doc in corpus]

# 构建结果 DataFrame
result_df = pd.DataFrame({
    "情绪标签": sentiments,
    "时间": times,
    "性别": genders,
    "评论": comments
})

# 保存结果到文件
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"分析完成，结果已保存到 {output_path}")
