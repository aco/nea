import math
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.data

stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")

ideal_sent_length = 8.0

class Summarizer():

    def __init__(self, articles):
        """ Initialises the Summariser class: a self-contained object accepting a corpus of articles to summarise.

        Args:
            articles ([[headline, body]]): A 2D array, where the 0th subarray is the headline and the 1st the body.
        """        

        self._articles = articles
 

    def tokenize_and_stem(self, text):
        """ Applies standard natural language preprocessing, tokenizing each word, stripping punctuation,
        

        Args:
            text ([str]): List of sentences.

        Returns:
            [str]: Tokenised and lemmanized representation of sentence
        """       

        tokens = [word for sent in nltk.sent_tokenize(text) 
            for word in nltk.word_tokenize(sent)]

        filtered = []

        # filter out numeric tokens, raw punctuation, etc.
        for token in tokens:
            token = token.replace('-', ' ')
            
            if re.search('[a-zA-Z]', token):
                filtered.append(token)

        stems = [stemmer.stem(t) for t in filtered]
        
        return stems


    def score(self, article):
        """ Scores sentences in an article according to its relevance to its headline, length, 
            position (assuming article sentences are written in order of importance).

        Args:
            article ([headline: str, body: str]): Article headline and body
        """

        headline = article[0]
        sentences = self.split_into_sentences(article[1])
        frequency_scores = self.frequency_scores(article[1])

        for i, s in enumerate(sentences):
            headline_score = self.headline_score(headline, s) * 1.5
            length_score = self.length_score(self.split_into_words(s)) * 1.0
            position_score = self.position_score(float(i+1), len(sentences)) * 1.0

            frequency_score = frequency_scores[i] * 4
            score = (headline_score + frequency_score + length_score + position_score) / 4.0
            
            self._scores[s] = score


    def generate_summaries(self, summary_length):
        """ If article is shorter than the desired summary, just return the original articles. """

        # edge case: corpus length < summary_length
        total_num_sentences = 0

        for article in self._articles:
            total_num_sentences += len(self.split_into_sentences(article[1]))

        if total_num_sentences <= summary_length:
            return [x[1] for x in self._articles]

        self.build_TFIDF_model()  # computed once

        self._scores = Counter()

        for article in self._articles:
            self.score(article)

        highest_scoring = self._scores.most_common(summary_length)

        print("## Headlines: ")

        for article in self._articles:
            print("- " + article[0])

        return [sent[0] for sent in highest_scoring]
  

    def split_into_words(self, text):
        """ Parses sentence through a regex into words, omitting punctuation.

        Args:
            text (str): Text to split.

        Returns:
            [str]: Filtered, sanitised words within text (non-distinct).
        """        

        try:
            text = re.sub(r'[^\w ]', '', text)

            return [w.strip('.').lower() for w in text.split()]
        except TypeError:
            return ''


    def split_into_sentences(self, text):
        """ Parses document through string replacement into words, omitting punctuation,
            before tokenizing into sentences.

        Args:
            text (str): Complete text.

        Returns:
            [str]: Tokenized sentences within.
        """        

        text = text.replace('.\n', '. ').replace('\n', ' ')

        tok = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = tok.tokenize(text)
        sentences = [sent.replace('\n', ' ') for sent in sentences if len(sent) > 10]

        return sentences


    def headline_score(self, headline, sentence):
        """ Gives sentence a score between (0,1) based on percentage of words common to the headline. """
        """Scores headline's representativeness 0..1 using proportion of article content present in headline.

        Returns:
            float: Score, calculated from headline and body.
        """        

        title_stems = [stemmer.stem(w) for w in headline if w not in stop_words]
        sentence_stems = [stemmer.stem(w) for w in sentence if w not in stop_words]

        count = 0.0

        for word in sentence_stems:
            if word in title_stems:
                count += 1.0

        score = count / len(title_stems)

        return score


    def length_score(self, sentence):
        """ Scores a sentence 0..1 according to word count against given ideal sentence length.

        Args:
            sentence ([str]): Sentence to score, given as list of words.

        Returns:
            float: Final score.
        """        
        
        len_diff = math.fabs(ideal_sent_length - len(sentence))

        return len_diff / ideal_sent_length


    def position_score(self, i, size):
        """ Scores sentenece according to position in article, assuming article sentences are written
            in order of importance (https://github.com/MojoJolo/textteaser/blob/master/src/main/scala/com/textteaser/summarizer/Parser.scala)
        """

        relative_position = i / size

        if 0 < relative_position <= 0.1:
            return 0.17
        elif 0.1 < relative_position <= 0.2:
            return 0.23
        elif 0.2 < relative_position <= 0.3:
            return 0.14
        elif 0.3 < relative_position <= 0.4:
            return 0.08
        elif 0.4 < relative_position <= 0.5:
            return 0.05
        elif 0.5 < relative_position <= 0.6:
            return 0.04
        elif 0.6 < relative_position <= 0.7:
            return 0.06
        elif 0.7 < relative_position <= 0.8:
            return 0.04
        elif 0.8 < relative_position <= 0.9:
            return 0.04
        elif 0.9 < relative_position <= 1.0:
            return 0.15

        return 0


    def build_TFIDF_model(self):
        """ Constructs a TF-IDF model from the Reuters 11k news dataset. """        

        token_dict = {}

        for article in reuters.fileids():
            token_dict[article] = reuters.raw(article)

        # Use TF-IDF to determine frequency of each word in our article, relative to the
        # word frequency distributions in corpus of 11k Reuters news articles.
        self._tfidf = TfidfVectorizer(tokenizer=self.tokenize_and_stem, stop_words='english', decode_error='ignore')
        
        tdm = self._tfidf.fit_transform(token_dict.values())  # Term-document matrix


    def frequency_scores(self, article_text):
        """ Scores the TF-IDF word frequencies within the article.

        Args:
            article_text (_type_): _description_

        Returns:
            _type_: _description_
        """

        response = self._tfidf.transform([article_text])
        feature_names = self._tfidf.get_feature_names()

        word_prob = {}

        for col in response.nonzero()[1]:
            word_prob[feature_names[col]] = response[0, col]

        sent_scores = []

        for sentence in self.split_into_sentences(article_text):
            score = 0
            sent_tokens = self.tokenize_and_stem(sentence)

            for token in (t for t in sent_tokens if t in word_prob):
                score += word_prob[token]

            sent_scores.append(score / len(sent_tokens)) # normalise against sentence length

        return sent_scores
