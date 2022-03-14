# Nea

Nea was a pet project aiming to take a series of news articles from different sources, each covering the same event/topic, and author a new article based on common topics and content. It existed as an iOS app with an Express backend, however this repo hosts the core of this project only. 

We take a series of article URLs and crawl it into structured content with the help of [news-please](https://github.com/fhamborg/news-please), applying further cleaning where necessary (since web scraping is messy business). Our cleaned corpus then undergoes standard NL preprocessing and is fed to a Reuters-11k-trained TF-IDF Vectorizer from scikit-learn. A scoring mechanic assesses, ranks, and compiles the output into a final _article_. Please see the code and documentation inline for more.

## License

GNU GPL v3
