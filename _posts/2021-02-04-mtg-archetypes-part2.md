---
layout: post
title:  "Topic modeling and Magic: The Gathering (Part 2)"
date:   2021-02-04
---

# Topic modelling and Magic: The Gathering (Part 2)

In the previous blog post, I introduced how it is possible to use topic modeling to find MTG archetypes with **Latent Dirichlet allocation (LDA)**. The code to do so is available on my [GitHub](https://github.com/pfr974/topic_modeling_mtg). Let me guide you through the notebook and discuss the results.

## Gathering and processing the data

The data analyzed in this post consists of 3718 Legacy decklists registered in 2020 on [mtgtop8](https://www.mtgtop8.com). To obtain these decklists, I scrapped the content of the website; see the [spider_mtg repo](https://github.com/pfr974/spider_mtg) for more details. The decklists are stored in a stored in a single file, one decklist per line. We have 75 cards in a deck with 60 cards mainboard and 15 cards sideboard:
```
"3  Bayou 1  Dryad Arbor 2  Marsh Flats 3  Misty Rainforest 3  Polluted Delta 1  Snow-Covered Swamp 3  Underground Sea 4  Verdant Catacombs 4  Bloodghast 4  Gravecrawler 4  Hedron Crab 4  Hogaak, Arisen Necropolis 2  Putrid Imp 4  Stitcher\\'s Supplier 4  Vengevine 4  Cabal Therapy 2  Careful Study 4  Altar of Dementia 4  Bridge from Below 3  Chain of Vapor 4  Force of Vigor 4  Leyline of the Void 1  Oko, Thief of Crowns 3  Thoughtseize "
````

Our **document** here is a decklist such as the one above. The **corpus**, a collection of document, consists of all the decklists. For the **vocabulary**, the set of all words used in the corpus, we want to associate each card name in it with a unique integer ID. We do this using gensim.corpora.Dictionary. Following the [core concepts of gensim](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html), we take an extra step by constructing a dictionary without loading all the decklists into memory:
```python
dictionary = gensim.corpora.Dictionary([x.strip() for x in re.split(r"[\d]+", line.replace("\"", ""))] for line in open('single_legacy_2020.txt'))
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(once_ids)  # remove cards that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
```

Likewise for the corpus,  we want it to be memory friendly. We define a class **MyCorpus**:
```python
class MyCorpus(object):
    
    def __iter__(self):
        for line in open('single_legacy_2020.txt'):
            decklist = line.replace("\"", "") # remove start and end tokens
            decklist = re.split(r"([\d]+)", decklist) # split by numbers and card names
            decklist = [x.strip() for x in decklist] # remove whitespace
            decklist = list(filter(None, decklist)) # remove empty words
            cleaned_decklist = [] 
            for i in range(int(len(list(decklist))/2)):
                for j in range(int(len(list(decklist[i*2])))):
                    cleaned_decklist.append(decklist[i*2+1])
            yield dictionary.doc2bow(cleaned_decklist)
    
corpus_memory_friendly = MyCorpus()
```
In the <code><i>__iter__</i></code> function, we retrieve and clean a list of tokens (card names) in each decklist. With the line <code><i>yield dictionary.doc2bow(cleaned_decklist)</i></code> we convert a list of tokenized card names via a dictionary to their ids and yield the resulting bag of words (bow) corpus. To simplify, here we are counting how many times a card name, via its id, appears in a decklist. If we were to add a <code><i>__print__</i></code> function for our <code><i>MyCorpus</i></code> object, we would see something like this for one decklist:
```
[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1)]
```
The first tuple tells you that the card with the ID 1 has been counted 1 time in the document. Note that for all these tuples, the second entry is always 1. This is expected given how the cleaned list of tokenized card names was built in the snipet above.

## LDA model

Now that we have a corpus, we can proceed to transform it with the gensim LDA model:

```python
def compute_models_coherence(dictionary, corpus_memory_friendly, model, limit, start=2, step=3):
    """
    Return topic modeling models and u_mass coherence values for various number of topics.
    For more info about coherence, see:
    - https://radimrehurek.com/gensim/models/coherencemodel.html
    - https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    model : the topic modeling model (lda or nmf)
    start: Starting number of topics
    limit : Max number of topics
    step: increment

    Returns:
    -------
    model_list : List of LDA or NMF topic models
    coherence_values : u_mass Coherence values corresponding to the LDA or NMF model 
    with respective number of topics
    """
    
    coherence_values = []
    model_list = []
    
    # We set iterations and passes to the same number
    iterations = 50
    # See https://groups.google.com/g/gensim/c/z0wG3cojywM to read about the difference between passes and iterations 
    
    np.random.seed(1) # For reproductivity
    unique_cards = len(dictionary.keys())
    
    if model == 'nmf':
        
        for archetypes in range(start, limit, step):
        
            model= Nmf(corpus=corpus_memory_friendly, num_topics=archetypes,id2word=dictionary,chunksize=2000,
                                     passes=iterations,kappa=.1,minimum_probability=0.01,w_max_iter=300,
                                     w_stop_condition=0.0001,h_max_iter=100,
                                     h_stop_condition=0.001,eval_every=10,
                                     normalize=True,random_state=np.random.seed(1))
        
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, corpus=corpus_memory_friendly, 
                                            dictionary=dictionary, coherence='u_mass')
            coherence_values.append(coherencemodel.get_coherence())

    if model == 'lda':
        
        for archetypes in range(start, limit, step):
        
            alpha_prior = [1.0 / archetypes] * archetypes
            beta_prior = [1.0 / archetypes] * unique_cards
        
            model=gensim.models.ldamodel.LdaModel(corpus=corpus_memory_friendly, id2word=dictionary, 
                                                  num_topics=archetypes, passes=iterations, 
                                                  alpha = alpha_prior, eta = beta_prior)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, corpus=corpus_memory_friendly, 
                                            dictionary=dictionary, coherence='u_mass')
            coherence_values.append(coherencemodel.get_coherence())
    
    return model_list, coherence_values

model_list_lda, coherence_values_lda = compute_coherence_values(dictionary=dictionary,
                                                                corpus_memory_friendly = corpus_memory_friendly, 
                                                                model='lda', start=2, limit=20, step=6)
```

As stated in Part 1, LDA is a probabilistic and unsupervised algorithm that assumes that a document (decklist) is a mixture of topics (archetypes). An archetype obtained via LDA can be seen as a probability distribution over card names. What you get in the end is a list of weighted card names for each unnamed archetype that has been found. Moreover, you have to decide how many archetypes to find. To decide what was the best number of archetypes, I added in the <code><i>compute_models_coherence</i></code> fuction the notion of coherence. **The coherence score allows to quantitatively evaluate how good a model is**. At least, that's how I understand it. I won't go into details (for now) and suggest you to read this [short article](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0) and [this paper by Roeder et al.](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf). Please note that here, I am looking at tuning only one hyperparameter: the number of archetypes. If you read through the code snipet above, the <i>alpha_prior</i> and <i>beta_prior</i> are fixed for a given number of archetypes.

## Results

The optimal number of archetypes is given by the plot below:

![u_mass coherence score vs number of archetypes](plot_lda_coherence_74_topics.png)

[Based of my understanding of the u_mass coherence score](https://gensimr.news-r.org/articles/coherence.html), the more negative a value is, the better the LDA model is. It seems then that **74** archetypes worked well. Now, let's check the top 18 cards of one of the archetype discovered, Archetype 0:
```
Archetype 0 
 0.041*Dread Return
 0.041*Lotus Petal
 0.040*Cabal Therapy
 0.039*Narcomoeba
 0.039*Bridge from Below
 0.038*Golgari Grave-Troll
 0.038*Ichorid
 0.038*Golgari Thug
 0.038*Stinkweed Imp
 0.038*Lion's Eye Diamond
 0.037*Ashen Rider
 0.037*Gemstone Mine
 0.036*Cephalid Coliseum
 0.035*City of Brass
 0.035*Faithless Looting
 0.034*Hogaak, Arisen Necropolis
 0.032*Leyline of the Void
 0.032*Careful Study  
 ```
If you want to visualise it:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/archetype0.pdf">
    </div>
</div>
<div class="caption">
    Classic Legacy Dredge.
</div>

Well, our friend Golgari Grave-Troll, the word <i>dredge</i> and Bridge from Below tell us that Archetype 0 matches Dredge ヾ(;ﾟДﾟ;)ｼ. 

Let's check some other archetypes in the table below:

|  Archetype 14              | Archetype 73                   | Archetype 17                | Archetype 54                  |
|----------------------------|--------------------------------|-----------------------------|-------------------------------|
| 0.062*Phyrexian Revoker    |  0.055*Green Sun's Zenith      | 0.034*Goblin Trashmaster    | 0.053*Thassa's Oracle         |
| 0.043*Karn, Scion of Urza  |  0.053*Dryad Arbor             | 0.032*Mountain              | 0.045*Aether Vial             |
| 0.040*Lodestone Golem      |  0.038*Collector Ouphe         | 0.031*Pyrokinesis           | 0.043*Force of Will           |
| 0.039*Stonecoil Serpent    |  0.037*Scavenging Ooze         | 0.031*Goblin Ringleader     | 0.039*Cabal Therapy           |
| 0.037*Wasteland            |  0.036*Savannah                | 0.031*Goblin Matron         | 0.039*Nomads en-Kor           |
| 0.036*Arcbound Ravager     |  0.033*Wasteland               | 0.031*Cavern of Souls       | 0.039*Cephalid Illusionist    |
| 0.034*Mishra's Factory     |  0.031*Sylvan Library          | 0.031*Goblin Cratermaker    | 0.037*Narcomoeba              | 
| 0.034*Chalice of the Void  |  0.029*Swords to Plowshares    | 0.031*Goblin Chainwhirler   | 0.037*Dread Return            |
| 0.034*Thorn of Amethyst    |  0.028*Knight of the Reliquary | 0.030*Goblin Lackey         | 0.036*Shuko                   |
| 0.033*Walking Ballista     |  0.028*Windswept Heath         | 0.030*Goblin Warchief       | 0.035*Daze                    |
| 0.033*Ancient Tomb         |  0.026*Karakas                 | 0.028*Gempalm Incinerator   | 0.035*Underground Sea         |
| 0.033*Umezawa's Jitte      |  0.026*Noble Hierarch          | 0.027*Aether Vial           | 0.034*Polluted Delta          |
| 0.033*City of Traitors     |  0.025*Knight of Autumn        | 0.027*Wasteland             | 0.032*Flooded Strand          |
| 0.032*Mox Opal             |  0.023*Ramunap Excavator       | 0.027*Skirk Prospector      | 0.030*Stoneforge Mystic       |
| 0.031*Karakas              |  0.022*Forest                  | 0.027*Badlands              | 0.030*Recruiter of the Guard  |
| 0.031*Leyline of the Void  |  0.021*Questing Beast          | 0.026*Rishadan Port         | 0.030*Batterskull             |
| 0.031*Sorcerous Spyglass   |  0.019*Verdant Catacombs       | 0.026*Munitions Expert      | 0.029*Ponder                  |
| 0.029*Steel Overseer       |  0.019*Gaddock Teeg            | 0.026*Sling-Gang Lieutenant | 0.028*Brainstorm              |
|                            |                                |                             |                               |
| <b>Steel and Taxes</b>     | <b>Maverick</b>                | <b>Goblins</b>              | <b>Cephalid Breakfast</b>     |


I am happy that among the achetypes discovered classics such as Goblins and Maverick are matched, but I am actually impressed that something such as Cephalid Breakfast can be matched! However there can be some surprising results such as Archetype 42 for instance:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/archetype42.pdf">
    </div>
</div>
<div class="caption">
    Two Goblins decided to trash the Rogue party.
</div>
