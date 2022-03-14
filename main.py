import json
import difflib
import uuid

from summarize import Summarizer
from article_fetch.process import process_topic


def strip_useless_sentences(sentences):
  """Final sanity checks against an array of sentences. Checks for correct punctuation,
     acceptable sentence length, and matching quotations, filtering any which do not match
     this criteria.

  Args:
      sentences ([str]): List of sentences

  Returns:
      [str]: Filtered list of sentences
  """  

  ns = []

  while sentences: 
    sen = sentences.pop()
    ns.append(sen)
    
    sentences = [s for s in sentences
      if s not in difflib.get_close_matches(sen, sentences, cutoff=0.6)]

  for sentence in reversed(ns):
    sen = sentence

    if len(sen) < 1:
      continue

    if (sen[0].isupper() and sen.count(' ') < 80 and sen.count('"') % 2 == 0):
      sentences.append(sentence)
  
  return sentences


def generate_article_from_topic(topic):
  """Accept a series of articles, forming a topic, and return a summarised version.

  Args:
      topic ({sources: [url]}): 

  Returns:
      {sentences: [str], sources: [Article]}: _description_
  """

  sources = topic['sources']
  articles = process_topic(sources)

  intro_input = [(a['title'], a['body'].split('\n')[0]) 
    for a in articles if a['body'] is not None]

  if len(intro_input) < 1:
    return None

  summarizer = Summarizer(intro_input)

  intro_sums = summarizer.generate_summaries(summary_length=2)
  intro = max(intro_sums, key=len)

  summ_input = [(a['title'], '\n'.join(a['body'].split('\n')[1:]))
    for a in articles if a['body'] is not None]

  summarizer = Summarizer(summ_input)

  sentences = summarizer.generate_summaries(summary_length=5)
  
  sentences.insert(0, intro)

  sentences = strip_useless_sentences(sentences)
  sentences = list(dict.fromkeys(sentences))

  return {
    'sentences': sentences,
    'sources': articles
  }


if __name__ == '__main__':
  topics = json.load(open('out.json', 'r'))
  articles = {}

  for topic in topics:
    if len(topic['sources']) < 2:
      continue

    rendered = generate_article_from_topic(topic)

    for source in rendered['sources']:
      _ = source.pop('body', None)

    ident = str(uuid.uuid4())

    articles[ident] = {
      'body': rendered['sentences'],
      'sources': rendered['sources'],
      'date': topic['date'],
      'tags': topic['tags']
    }

  with open('articles.json', 'w') as f:
    json.dump(articles, f, indent=2)
