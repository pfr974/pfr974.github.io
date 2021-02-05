---
layout: post
title:  "Topic modeling and Magic: The Gathering (Part 2)"
date:   2021-02-05
---

[In the previous blog post](https://pfr974.github.io/blog/2021/mtg-archetypes-part1/), I introduced how it is possible to use topic modeling to find MTG archetypes with **Latent Dirichlet allocation (LDA)**. The code to do so is available on my [GitHub](https://github.com/pfr974/topic_modeling_mtg).  <a href="{{ site.baseurl }}/assets/html/LDA_Visualization_legacy_2020_74_archetypes.html"  target="_blank"><b>To visualise the 74 archetypes found, click here</b></a>. Let me now guide you through the notebook and discuss the results.

## Gathering and processing the data

The data analyzed in this post consists of 3718 Legacy decklists registered in 2020 on [mtgtop8](https://www.mtgtop8.com). To obtain these decklists, I scrapped the content of mtgtop8; [see the spider_mtg repo for more details](https://github.com/pfr974/spider_mtg). Also, [if you are interested in more Legacy data, I have done the same for other years](https://github.com/pfr974/mtg-legacy-data). 

The decklists are stored in a single file, one decklist per line. We have 75 cards in a deck with 60 cards mainboard and 15 cards sideboard:
```
"3  Bayou 1  Dryad Arbor 2  Marsh Flats 3  Misty Rainforest 3  Polluted Delta 1  Snow-Covered Swamp 3  Underground Sea 4  Verdant Catacombs 4  Bloodghast 4  Gravecrawler 4  Hedron Crab 4  Hogaak, Arisen Necropolis 2  Putrid Imp 4  Stitcher\\'s Supplier 4  Vengevine 4  Cabal Therapy 2  Careful Study 4  Altar of Dementia 4  Bridge from Below 3  Chain of Vapor 4  Force of Vigor 4  Leyline of the Void 1  Oko, Thief of Crowns 3  Thoughtseize "
```
Our **document** here is a decklist such as the one above. The **corpus**, a collection of documents, consists of all the decklists. For the **vocabulary**, the set of all words used in the corpus, we want to associate each card name in it with a unique integer ID. We do this using <code><i>gensim.corpora.Dictionary</i></code>. Following the [core concepts of gensim](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html), we take an extra step by constructing a dictionary without loading all the decklists into memory:

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

The first tuple tells you that the card with the ID 1 has been counted one time in the document. Note that for all these tuples, the second entry is always 1.  It is explained by how we created the cleaned list of tokenized card names in the snippet above.

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

As stated in Part 1, LDA is a probabilistic and unsupervised algorithm that assumes that a document (decklist) is a mixture of topics (archetypes). An archetype obtained via LDA can be seen as a probability distribution over card names. What you get in the end is a list of weighted card names for each unnamed archetype that has been found. Moreover, you have to decide how many archetypes to find. 

To decide what was the best number of archetypes, I added in the <code><i>compute_models_coherence</i></code> function the notion of coherence. **The coherence score allows to quantitatively evaluate how good a model is**. At least, that's how I understand it. I won't go into details (for now) and suggest you read this [short article](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0) and [this paper by Roeder et al.](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf). Please note that here, I am looking at tuning only one hyperparameter: the number of archetypes. If you read through the code snippet above, the <i>alpha_prior</i> and <i>beta_prior</i> are fixed for a given number of archetypes.

## Results

The optimal number of archetypes is given by the plot below:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/plot_lda_coherence_74_topics.png">
    </div>
</div>
<div class="caption">
    
</div>


[Based on my understanding of the u_mass coherence score](https://gensimr.news-r.org/articles/coherence.html), the lower a value is, the better the LDA model is. It seems then that **74** archetypes worked well. 

Now, let's check the top 18 cards of one of the archetype discovered, Archetype 0:

<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
}
</style>
</head>
<body>
<table style="width:50%">
  <tr>
    <th style="text-align:center">Archetype 0</th>
  </tr>
  <tr>
 <tr><td style="text-align:center">0.041*Dread Return</td></tr>
 <tr><td style="text-align:center">0.041*Lotus Petal</td></tr>
 <tr><td style="text-align:center">0.040*Cabal Therapy</td></tr>
 <tr><td style="text-align:center">0.039*Narcomoeba</td></tr>
 <tr><td style="text-align:center">0.039*Bridge from Below</td></tr>
 <tr><td style="text-align:center">0.038*Golgari Grave-Troll</td></tr>
 <tr><td style="text-align:center">0.038*Ichorid</td></tr>
 <tr><td style="text-align:center">0.038*Golgari Thug</td></tr>
 <tr><td style="text-align:center">0.038*Stinkweed Imp</td></tr>
 <tr><td style="text-align:center">0.038*Lion's Eye Diamond</td></tr>
 <tr><td style="text-align:center">0.037*Ashen Rider</td></tr>
 <tr><td style="text-align:center">0.037*Gemstone Mine</td></tr>
 <tr><td style="text-align:center">0.036*Cephalid Coliseum</td></tr>
 <tr><td style="text-align:center">0.035*City of Brass</td></tr>
 <tr><td style="text-align:center">0.035*Faithless Looting</td></tr>
 <tr><td style="text-align:center">0.034*Hogaak, Arisen Necropolis</td></tr>
 <tr><td style="text-align:center">0.032*Leyline of the Void</td></tr>
 <tr><td style="text-align:center">0.032*Careful Study</td></tr>
  </tr>
</table>
</body>
</html>


If you want something more visual:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/archetype0_small.png">
    </div>
</div>
<div class="caption">
    Some classic cards for Legacy Dredge displayed here.
</div>

Well, our friend <i>Golgari Grave-Troll</i>, the word <i>dredge</i> and <i>Bridge from Below</i> tell us that Archetype 0 matches [Dredge](https://mtgdecks.net/Legacy/dredge-analysis-5400/all), an archetype so powerful that it creates its own rules ᕕ( ͡° ͜ʖ ͡° )ᕗ. 

We can check some other archetypes in the table below. I have indicated in the last row which archetype they match best:

<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
}
</style>
</head>
<body>
<table style="width:100%">
  <tr>
    <th style="text-align:center"><b>Archetype 14</b></th>
    <th style="text-align:center"><b>Archetype 73</b></th>
    <th style="text-align:center"><b>Archetype 17</b></th>
    <th style="text-align:center"><b>Archetype 54</b></th>
  </tr>
  <tr>

<tr>
<td style="text-align:center">0.062*Phyrexian Revoker</td>    
<td style="text-align:center">0.055*Green Sun's Zenit</td>      
<td style="text-align:center">0.034*Goblin Trashmaste</td>     
<td style="text-align:center">0.053*Thassa's Oracle</td>      
</tr>
<tr>
<td style="text-align:center">0.043*Karn, Scion of Urza</td>   
<td style="text-align:center">0.053*Dryad Arbor        </td>     
<td style="text-align:center">0.032*Mountain           </td>    
<td style="text-align:center">0.045*Aether Vial  </td>
</tr>
<tr>           
<td style="text-align:center">0.040*Lodestone Golem    </td>   
<td style="text-align:center">0.038*Collector Ouphe    </td>     
<td style="text-align:center">0.031*Pyrokinesis        </td>    
<td style="text-align:center">0.043*Force of Will    </td>
</tr>
<tr>       
<td style="text-align:center">0.039*Stonecoil Serpent  </td>  
<td style="text-align:center">0.037*Scavenging Ooze    </td>    
<td style="text-align:center">0.031*Goblin Ringleader  </td>   
<td style="text-align:center">0.039*Cabal Therapy</td>   
</tr>
<tr>        
<td style="text-align:center">0.037*Wasteland        </td>     
<td style="text-align:center">0.036*Savannah         </td>       
<td style="text-align:center">0.031*Goblin Matron    </td>      
<td style="text-align:center">0.039*Nomads en-Kor   </td>
</tr>
<tr>        
<td style="text-align:center">0.036*Arcbound Ravager      </td>
<td style="text-align:center">0.033*Wasteland             </td> 
<td style="text-align:center">0.031*Cavern of Souls       </td>
<td style="text-align:center">0.039*Cephalid Illusionist  </td>
</tr>
<tr>  
<td style="text-align:center">0.034*Mishra's Factory   </td>   
<td style="text-align:center">0.031*Sylvan Library       </td>   
<td style="text-align:center">0.031*Goblin Cratermaker    </td> 
<td style="text-align:center">0.037*Narcomoeba    </td>
</tr>
<tr>          
<td style="text-align:center">0.034*Chalice of the Void </td>  
<td style="text-align:center">0.029*Swords to Plowshares  </td>  
<td style="text-align:center">0.031*Goblin Chainwhirler  </td>  
<td style="text-align:center">0.037*Dread Return       </td>
</tr>
<tr>     
<td style="text-align:center">0.034*Thorn of Amethyst     </td>
<td style="text-align:center">0.028*Knight of the Reliquary </td>
<td style="text-align:center">0.030*Goblin Lackey          </td>
<td style="text-align:center">0.036*Shuko        </td>
</tr>
<tr>           
<td style="text-align:center">0.033*Walking Ballista      </td>
<td style="text-align:center">0.028*Windswept Heath       </td>  
<td style="text-align:center">0.030*Goblin Warchief       </td> 
<td style="text-align:center">0.035*Daze      </td>
</tr>
<tr>              
<td style="text-align:center">0.033*Ancient Tomb         </td> 
<td style="text-align:center">0.026*Karakas              </td>   
<td style="text-align:center">0.028*Gempalm Incinerator  </td>  
<td style="text-align:center">0.035*Underground Sea </td>
</tr>
<tr>        
<td style="text-align:center">0.033*Umezawa's Jitte      </td> 
<td style="text-align:center">0.026*Noble Hierarch       </td>   
<td style="text-align:center">0.027*Aether Vial          </td>  
<td style="text-align:center">0.034*Polluted Delta </td>
</tr>
<tr>         
<td style="text-align:center">0.033*City of Traitors    </td>  
<td style="text-align:center">0.025*Knight of Autumn    </td>    
<td style="text-align:center">0.027*Wasteland           </td>   
<td style="text-align:center">0.032*Flooded Strand  </td>
</tr>
<tr>        
<td style="text-align:center">0.032*Mox Opal             </td> 
<td style="text-align:center">0.023*Ramunap Excavator    </td>   
<td style="text-align:center">0.027*Skirk Prospector     </td>  
<td style="text-align:center">0.030*Stoneforge Mystic   </td>
</tr>
<tr>    
<td style="text-align:center">0.031*Karakas  </td>             
<td style="text-align:center">0.022*Forest     </td>             
<td style="text-align:center">0.027*Badlands   </td>            
<td style="text-align:center">0.030*Recruiter of the Guard </td>
</tr>
<tr> 
<td style="text-align:center">0.031*Leyline of the Void  </td> 
<td style="text-align:center">0.021*Questing Beast</td>          
<td style="text-align:center">0.026*Rishadan Port   </td>       
<td style="text-align:center">0.030*Batterskull   </td>  
</tr>
<tr>        
<td style="text-align:center">0.031*Sorcerous Spyglass </td>   
<td style="text-align:center">0.019*Verdant Catacombs  </td>     
<td style="text-align:center">0.026*Munitions Expert </td>      
<td style="text-align:center">0.029*Ponder </td>          
</tr>
<tr>       
<td style="text-align:center">0.029*Steel Overseer    </td>    
<td style="text-align:center">0.019*Gaddock Teeg  </td>          
<td style="text-align:center">0.026*Sling-Gang Lieutenant  </td>
<td style="text-align:center">0.028*Brainstorm </td>             
</tr>
  </tr>
    <tr>
    <th style="text-align:center"><b>Steel and Taxes</b></th>
    <th style="text-align:center"><b>Maverick</b></th>
    <th style="text-align:center"><b>Goblins</b></th>
    <th style="text-align:center"><b>Cephalid Breakfast</b></th>
  </tr>
</table>
</body>
</html>



I am satisfied that among the archetypes discovered classics such as [Goblins](https://mtgdecks.net/Legacy/goblins) and [Maverick](https://mtgdecks.net/Legacy/maverick-analysis-5498/all) are matched. I was also impressed that the model managed to find something such as [Cephalid Breakfast](https://mtgdecks.net/Legacy/cephalid-breakfast-analysis-7834/all)! However, there can be some surprising results such as Archetype 42 for instance:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/archetype42_small.png">
    </div>
</div>
<div class="caption">
    Two goblins decided to trash the Rogue Party.
</div>

Archetype 42 matches [UB Rogues](https://mtgdecks.net/Legacy/ub-rogues-analysis-9653/all) except for two cards found usually in [Goblins](https://mtgdecks.net/Legacy/goblins): <i>Conspicuous Snoop</i> and <i>Boggart Harbinger</i>. Obviously, it is not too bad since two outliers out of 18 do not prevent identifying existing archetypes. However, it would be interesting to know why it happened.

I do not claim to be an expert of the Legacy format but using my knowledge and [mtgdecks.net](https://mtgdecks.net/Legacy), I have checked personally **44** archetypes out of **74** and found **36** matches to existing archetypes. The **8**  remaining ones seem to be a mix of different existing archetypes.

## Visualising the results with pyLDAvis

To make sense of the different archetypes found by our model, we need to go through them individually. With more than 50 topics, it can get tedious. Even more so when trying to see how different archetypes relate to each other. Fortunately, there is a Python library called [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html) that gives a two-dimensional representation of the topics/archetypes found. In the notebook this is achieved via the lines:
```
pyLDAvis.enable_notebook()
gensim.corpora.MmCorpus.serialize('SerializedCorpus_legacy_2020_74_archetypes.mm'.format(i), corpus_memory_friendly)
SerializedCorpus = gensim.corpora.MmCorpus('SerializedCorpus_legacy_2020_74_archetypes.mm'.format(i))
vis_data = pyLDAvis.gensim.prepare(model_list_lda[-3], SerializedCorpus, dictionary,sort_topics=False)
pyLDAvis.save_html(vis_data, 'LDA_Visualization_legacy_2020_74_archetypes.html'.format(i))
```

The archetypes are then represented like below:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/pyLDAvis74.png">
    </div>
</div>
<div class="caption">
    Different archetypes as seen with pyLDAvis Note that the starting index is 1.
</div>

To see it for yourselves, <a href="{{ site.baseurl }}/assets/html/LDA_Visualization_legacy_2020_74_archetypes.html"  target="_blank"><b>click here and launch interactive exploration of the figure above</b></a>. Feel free to look at it and also to generate others via the notebook! 

For each archetype, represented by a bubble, we get the top 30 most salient card names. The size of a bubble has to do with the percentage of the card names in the corpus. For instance, the archetypes on the right part of the plot (21, 48, or 23) contain cards such as <i>Force of Will</i>, <i>Brainstorm</i>, or <i>Swords to Plowshare</i>. These cards are absolute all-stars in the Legacy formats, being among the most used. Now, if you are curious about the axes, I believe they are not interpretable by themselves: they came out of a multidimensional scaling algorithm. 

This representation is trying to maintain information about the archetype's distances, how closely related they are to each other, as best as it could. It is then possible to see that a lot of archetypes can overlap with each other. Take Archetype 42, the one which matches UB Rogues but contains 2 outliers in <i>Conspicuous Snoop</i> and <i>Boggart Harbinger</i>. Navigating on the plot, you'll be able to see that it is overlapping with Archetype 18, aka Goblins. These two archetypes share cards such as <i>Cavern of Souls</i> or <i>Thorn of Amethyst</i>. This might a start if you are still obsessed about why two Goblins ended up being part of the UB Rogue archetype.


## Conclusion

To end this post, we can confirm that it is possible to find Magic: The Gathering archetypes with Latent Dirichlet Allocation. For a format as complex a Legacy, it is possible to find common (Goblins, Elves) and less common (Steel and Taxes, Cephalid Breakfast) archetypes. It is even possible to start finding variations such as Grixis Delver and Sultai Delver. However, as seen with the pyLDAvis plot, archetypes can share common cards (<i>Force of Will</i>, <i>Brainstorm</i>, <i>Swords to Plowshare</i>) despite very different gameplay. Interpreting LDA model results is not as straightforward as we think. While I am familiar with the subject here, Magic: The Gathering, I can imagine how troublesome this could be when dealing with something more complex like a corpus of tweets.

## To do

Things that we could explore:
- finish identifying the archetypes;
- for a given new deck, find the most relevant archetypes among the ones that the model found;
- for two decks, explore how similar they are using the LDA model;
- hyperparameters tuning;
- test other topic modeling algorithms (Non-Negative Matrix Factorization for instance).