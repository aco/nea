import os
import threading

from newsplease import NewsPlease

from article_fetch.specification import Specification


def process_corpus(source, corpus):
  """ Parses headline, body, and metadata from given article URLs, applying 
      bulk cleanup operations.

  Args:
      source (_type_): _description_
      corpus (_type_): _description_
  """  
  try:
    article = NewsPlease.from_url(source['url'], timeout=3000)
      
    if article.language not in ['en']:
      return

    if article.maintext is None:
      if not source['name'] == 'The Washington Post':
        corpus.append({
          'title': article.title,
          'name': source['name'],
          'body': None,
          'img': article.image_url,
          'url': source['url']
        })
  except:
    return

  source = source['name']
  clean_spec = os.path.join(r'./clean_specs', f'{source}.json')

  if os.path.isfile(clean_spec):
    cleaned = Specification(clean_spec).cleanup(article.maintext)
  else:
    cleaned = Specification.run_general_cleanup(article.maintext)

  if cleaned is not None and len(cleaned) > 10:
    corpus.append({
      'title': article.title,
      'name': source['name'],
      'body': cleaned,
      'img': article.image_url,
      'url': source['url']
    })

    
def process_topic(topic_sources):
  corpus = []

  threads = [threading.Thread(target=process_corpus, args=(source, corpus))
    for source in topic_sources]

  for thread in threads:
    thread.start()

  for thread in threads:
    thread.join()
    
  return corpus
