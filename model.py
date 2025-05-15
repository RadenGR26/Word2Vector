from nltk.tokenize import TreebankWordTokenizer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ===== DATA DARI SLIDE =====
sentences = [
    "Pemrograman komputer adalah keterampilan yang penting.",
    "Data digunakan untuk analisis dalam berbagai bidang.",
    "Kecerdasan buatan membantu mengembangkan teknologi baru.",
    "Pembelajaran mesin adalah bagian dari kecerdasan buatan.",
    "Jaringan internet menghubungkan perangkat di seluruh dunia.",
    "Perangkat keras dan perangkat lunak adalah komponen komputer.",
    "Sistem informasi mengelola data dan pengetahuan.",
]

# ===== TOKENISASI =====
tokenizer = TreebankWordTokenizer()
tokenized = [tokenizer.tokenize(sent.lower()) for sent in sentences]

# ===== LATIH MODEL WORD2VEC =====
def train_model():
    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=0,  # 0 = CBOW, 1 = Skip-gram
        epochs=10
    )
    return model

def get_vocab(model):
    return model.wv.index_to_key

def get_vector(model, word):
    try:
        return model.wv[word].tolist()
    except KeyError:
        return None

def get_similar_words(model, word):
    try:
        return model.wv.most_similar(word, topn=5)
    except KeyError:
        return []

def generate_pca_plot(model, output_path='static/word2vec_plot.png'):
    words = model.wv.index_to_key[:30]
    vectors = [model.wv[word] for word in words]

    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(result[:, 0], result[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.title("Visualisasi Word2Vec dengan PCA")
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
