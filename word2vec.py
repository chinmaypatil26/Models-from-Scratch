text = """Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks."""

# tokenize the text
import re
import numpy as np

def tokenize(text):
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text.lower())


# create word to id conversions
def mapping(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word


# generate one hot encode
def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res


def concat(*iterables):
    for iterable in iterables:
        yield from iterable


# generate training data
def generate_training_data(tokens, window, word_to_id):

    X = []
    y = []

    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(range(i, max(0, i - window)), range(min(n_tokens, i + window + 1)))

        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)

# X.shape = number of words x vocab_size 

# initiate the model
# 60, 10
# 10, 60
def init_network(vocab_size, embedding_size):
    
    model = {
        "w1": np.random.randn(vocab_size, embedding_size),
        "w2": np.random.randn(embedding_size, vocab_size),
    }

    return model


## Created the model, now forward pass
## a1 = X*w1
## a2 = a1 * w2
## z = softmax(a2)
##a1 = (330, 60)@(60,10)
## a2 = (330,10)@(10,60)
## z = softmax(330,60)
#forward pass
def forward(model, X, return_cache=True):
    cache = {}

    cache['a1'] = X @ model['w1']
    cache['a2'] = cache['a1'] @ model['w2']
    cache['z'] = softmax(cache['a2'])

    if not return_cache:
        return cache['z']
    return cache

# cheated here
def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

#def backward 
#cheated here
def backward(model, X, y, learning_rate):
    cache = forward(model, X)
    da2 = cache['z'] - y
    dw2 = cache['a1'].T @ da2
    da1 = da2 @ model['w2'].T
    dw1 = X.T @ da1

    model['w1'] -= learning_rate * dw1
    model['w2'] -= learning_rate * dw2

    return cross_entropy(cache['z'], y)

def cross_entropy(z, y):
    return -np.sum(np.log(z)*y)

def test_model(model, X, y, learning_rate):
    n_iter = 50
    learning_rate = 0.01

    history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

    return history
if __name__ == "__main__":
    tokens = tokenize(text)
    word_to_id, id_to_word = mapping(tokens)
    X, y = generate_training_data(tokens, 2, word_to_id)

    vocab_size = len(word_to_id)
    embedding_size = 10
    model = init_network(vocab_size, embedding_size)

    history = test_model(model, X, y, 0.05)
    
    
