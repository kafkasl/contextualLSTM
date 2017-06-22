from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
paragraphs = ["Space Exploration Technologies Corporation, better known as SpaceX, is an American aerospace manufacturer and space transport services company headquartered in Hawthorne, California. It was founded in 2002 by entrepreneur Elon Musk with the goal of reducing space transportation costs and enabling the colonization of Mars. SpaceX has since developed the Falcon launch vehicle family and the Dragon spacecraft family, which both currently deliver payloads into Earth orbit.", "SpaceX's achievements include the first privately funded liquid-propellant rocket to reach orbit (Falcon 1 in 2008); the first privately funded company to successfully launch, orbit, and recover a spacecraft (Dragon in 2010); the first private company to send a spacecraft to the International Space Station (Dragon in 2012), and the first propulsive landing for an orbital rocket. As of March 2017, SpaceX has since flown ten missions to the International Space Station (ISS) under a cargo resupply contract. NASA also awarded SpaceX a further development contract in 2011 to develop and demonstrate a human-rated Dragon, which would be used to transport astronauts to the ISS and return them safely to Earth.", "SpaceX announced in 2011 they were beginning a privately funded reusable launch system technology development program. In December 2015, a first stage was flown back to a landing pad near the launch site, where it successfully accomplished a propulsive vertical landing. This was the first such achievement by a rocket for orbital spaceflight. In April 2016, with the launch of CRS-8, SpaceX successfully vertically landed a first stage on an ocean drone-ship landing platform. In May 2016, in another first, SpaceX again landed a first stage, but during a significantly more energetic geostationary transfer orbit mission. In March 2017, SpaceX became the first to successfully re-launch and land the first stage of an orbital rocket.", "In 2016, CEO Elon Musk unveiled the mission architecture of the Interplanetary Transport System program, an ambitious privately funded initiative to develop spaceflight technology for use in manned interplanetary spaceflight, and which, if demand emerges, could lead to sustainable human settlements on Mars over the long term. This is the main purpose this System was designed for. In 2017, Elon Musk announced that the company had been contracted by two private individuals to send them in a Dragon spacecraft on a free return trajectory around the Moon. Provisionally launching in 2018, this could become the first instance of lunar tourism."]

space_split = [line.split(" ") for line in paragraphs]

# compile sample documents into a list

# list for tokenized documents in loop
#texts = []

# loop through document list
#for i in paragraphs:
    
    # clean and tokenize document string
#    raw = i.lower()
#    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
#    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
#    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
#    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(space_split)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in space_split]


# Use first paragraph
y = [dictionary.token2id.get(word) for word in space_split[0]]
X = [0] + y
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
