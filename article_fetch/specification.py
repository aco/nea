import re
import json

from unidecode import unidecode


class Specification(object):
  def __init__(self, config_path):
    super().__init__()

    self.min_line_length = 4
    self.max_line_length = 100

    start_strip = [' ', ',', 'But', 'After', 'Then', 'See also']

    with open(config_path, 'r') as config_file:	
      attributes = json.load(config_file)

      self.remove_body = 'remove_body' in attributes
      self.remove_starting = tuple(start_strip + attributes['remove_starting'])
      self.regex_delete = attributes['regex_delete']

  
  def assess_line(self, line):
    """ Assesses a sentence for candidacy, base on minimum length, 
        sentence punctuality and casing, and no clause/join word as an opener 
        (this was an issue in the summariser).

    Args:
        line (_type_): _description_

    Returns:
        _type_: _description_
    """    
    line_word_count = line.count(' ')

    return (
      (line_word_count > self.min_line_length) and    # minimum length
      (line.endswith('.')) and                        # complete sentence
      (not line.startswith(self.remove_starting)) and # no clause/join word
      (not line.isupper())                            # not fully uppercase
    )

  
  def run_regex_replacements(self, article):
    """ Runs a series of regexes, taking care of common article content parsing could
        not differentiate.

    Args:
        article (str): Article body.

    Returns:
        str: Cleaned article body.
    """    

    article = re.sub(pattern='(.*?[^\."])\n', repl='\\1.\n', string=article)

    article = re.sub('\bper[ ]?cent\b', '%', article)
    article = re.sub('\[[\w\d]+\]\.?', '', article)
    article = re.sub('[\u00e2\u20ac\u201d-]', '-', article)
    article = re.sub("''", '"', article)
    article = re.sub('\(pictured .*?\)', '', article)
    article = re.sub('.*? \\.\\.\\. .*?\\n', '', article)

    for pattern in self.regex_delete:
      article = re.sub(pattern, '', article)
      
    return article


  @staticmethod
  def run_general_cleanup(article):
    """ Aids in unicode->ACII conversion, to ease summarisation on the model. 

    Args:
        article (str): Article body.

    Returns:
        str: Cleaned article body.
    """    
    return unidecode(article)


  def cleanup(self, article):
    """ Runs general cleanup on an article for easier processing. A source-specific specification is
        run afterward if one exists.

    Args:
        article (str): Article body.

    Returns:
        str: Cleaned article body.
    """    
    if self.remove_body:
      return ''

    article = self.run_general_cleanup(article)
    article = self.run_regex_replacements(article)

    lines = filter(lambda x: self.assess_line(x), article.splitlines())

    return '\n'.join(list(lines)).strip()
