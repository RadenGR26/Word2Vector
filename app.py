from flask import Flask, render_template, request
from model import train_model, get_similar_words, get_vector, get_vocab, generate_pca_plot

app = Flask(__name__)
model = train_model()
generate_pca_plot(model)

@app.route('/', methods=['GET', 'POST'])
def index():
    similar_words = []
    vector = None
    selected_word = ''

    if request.method == 'POST':
        selected_word = request.form['word']
        similar_words = get_similar_words(model, selected_word)
        vector = get_vector(model, selected_word)

    vocab = get_vocab(model)
    return render_template('index.html', vocab=vocab, vector=vector,
                           similar_words=similar_words, selected_word=selected_word)

if __name__ == '__main__':
    app.run(debug=True)
