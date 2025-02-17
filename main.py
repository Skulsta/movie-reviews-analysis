# gensim modules
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

# numpy
import numpy

# the classifier
from sklearn.linear_model import LogisticRegression


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # making sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Some prefix was not unique.')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])


sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}

sentences = LabeledLineSentence(sources)

"""
model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(10):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

model.save('d2v.model')
"""

model = Doc2Vec.load('d2v.model')

# Add reviews to numpy arrays and label them correctly.
train_arrays = numpy.zeros((40000, 100))
train_labels = numpy.zeros(40000)

for i in range(20000):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model[prefix_train_pos]
    train_arrays[20000 + i] = model[prefix_train_neg]
    train_labels[i] = 1
    train_labels[20000 + i] = 0


test_arrays = numpy.zeros((10000, 100))
test_labels = numpy.zeros(10000)

for i in range(5000):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model[prefix_test_pos]
    test_arrays[5000 + i] = model[prefix_test_neg]
    test_labels[i] = 1
    test_labels[5000 + i] = 0

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print("Test score: " + str(classifier.score(test_arrays, test_labels)))
