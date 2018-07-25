import codecs
from nltk.corpus import gutenberg as g
from nltk.corpus import brown as b
from nltk.corpus import reuters

class Sentences(object):
	"""docstring for Sentences"""
	def __init__(self, filepath):
		self.filepath = filepath
	
	def __iter__(self):
		with codecs.open(self.filepath, "r", "utf-8") as rf:
			for line in rf:
				if line.strip():
					yield line.strip().lower()
        

class GutenbergSentences(object):
    """docstring for Gutenberg sentences"""
    def __init__(self):
        self.sentences = g.sents()

    def __iter__(self):
        for line in self.sentences:
            yield " ".join(line).strip().lower()

class BrownSentences(object):
    """docstring for Brown sentences"""
    def __init__(self):
        self.sentences = b.sents()

    def __iter__(self):
        for line in self.sentences:
            yield " ".join(line).strip().lower()

class ReutersSentences(object):
    def __init__(self):
        self.sentences = reuters.sents()

    def __iter__(self):
        for line in self.sentences:
            yield " ".join(line).strip().lower()


