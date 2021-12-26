# Projects
*(Also viewable [here](https://raefahwahid.github.io/).)*


## [Abstract Clock](https://raefahwahid.github.io/abstract_clock/index.html) <br />
A creative interpretation of a clock. Made for COMS 4995.9: Intro to Data Visualization.


## Online Extractive Text Summarization Using Deep Reinforcement Learning
[GitHub Repository](https://github.com/raefahwahid/drl_text_summarization) <br /><br />
The task of text summarization is often difficult to automate because of sparse language data and the ambiguous definition of what precisely constitutes a “summary.” For the former issue, deep learning was originally combined with reinforcement learning in order better extract and use multi-dimensional features, like language data. For the latter issue, the reward systems and metrics that are so integral to deep reinforcement learning provide a unique opportunity for a model to learn its own definition of a “summary.” Essentially, deep reinforcement learning can be used to improve upon traditional text summarization techniques, which are often computationally heavy due to their semantic and algorithmic natures. Using a purely extractive online text summarization approach and the BLEU metric as a reward system, we create an agent that produces summaries from news articles. *(Final Project for COMS 6998: Practical Deep Learning Systems by Raefah Wahid & Gauri Narayan.)*


## Exploring Methods of Metaphor Detection Using Similarity
[Jupyter Notebook](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/metaphor_detection/metaphor_detection.ipynb) (compiled using Google Colab)<br />
[Poster](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/metaphor_detection/group08_poster%5BMetaphorDetection%5D-1.pdf) <br /><br />
Metaphors like “This is the price of citizenship” are frequently used in political speeches as powerful persuasive devices. The aim of this project is to detect such uses of metaphors in a corpora of speeches by former President Barack Obama (compiled from The Grammar Lab). We explore different methods of metaphor detection by using Word2Vec and BERT to find word embedding features, finding synonyms with the Lesk algorithm and KNN algorithm, and determining the metaphoricity of a sentence with cosine similarity. *(Final Project for COMS W4995: Semantic Representations for NLP by Raefah Wahid, Corina Hanaburgh, & Tiara Sykes.)*


## Sentiment-Based Stock Market Prediction
[Jupyter Notebook](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/final-project/final-notebook.ipynb) <br />
[Presentation](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/presentation.pdf)<br /><br />
Stock market prediction is widely known as a difficult and challenging task, in part due to the volatile and variable nature of the market itself. The Efficient Market Hypothesis (EMH) proposes that the stock market is primarily affected by new information, such as textual data in the form of news or tweets, rather than technical indicators that rely on past information (Bollen et al., 2011). Following this line of thought, researchers have explored different Natural Language Processing techniques when working with textual information as well as experimented with different prediction models. Our goal for this project is to evaluate the different models that can be used to tackle the problem of stock market prediction. We begin with a [simple Naive Bayes model](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/final-project/naive_bayes.py) for sentiment prediction as a baseline, and then follow it with a [continuous Dirichlet Process Mixture Model](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/final-project/cDPM.py) for topic-based sentiment prediction. We then use [Vector](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/final-project/var_model_NB_outputs.py) [Autoregression](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/market_forecasting/final-project/var_model_DPM_outputs.py) to evaluate how well these models’ outputs work in forecasting stock market closing prices. Due to the highly dynamic nature of both sentiment and the stock market, forecasting with a lag of more than seven days is unlikely to be effective, so we focus on short-term prediction. *(Final Project for COMS 6998: Machine Learning with Probabilistic Programming by Raefah Wahid & Gauri Narayan.)*


## Are Introductory CS Courses Effective and Encouraging?
[Paper](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/efficacy_study/finalpaper.pdf)<br />
[Item Analysis](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/efficacy_study/item_analysis.py)<br />
[Statistical Analysis](https://github.com/raefahwahid/raefahwahid.github.io/blob/main/efficacy_study/stats.py)<br /><br />
Interest in the field of computer science has swelled over the years. The mix of students pursuing computer science degrees is highly diverse, not only in gender and ethnicity, but in experience as well. It is difficult to gauge the true effectiveness of an introductory computer science course when there is a mixture of experienced and inexperienced students taking the class, because the higher grades of the experienced students often skew the average of the class, presenting a misrepresentation of the abilities of students within the class. This study aims to find out how effective and encouraging an introductory CS course can be when taking into account this experience gap. *(Joint project between the Columbia University Computer Science Department and Teachers College. Final Project for COMS W4995: Empirical Methods of Data Science. Student data excluded to protect privacy.)*
