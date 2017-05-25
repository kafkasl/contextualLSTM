
# USING A DICTIONARY FOR POSTERIOR LDA In [405]: empty.add_documents([sentence.split()])

# dictionary = corpora.Dictionary(space_split)
# dictionary.add_documents()
# corpus = [dictionary.doc2bow(text) for text in space_split]
# lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
# lda.print_topics(num_topics=2, num_words=2)


# paragraphs = ["Space Exploration Technologies Corporation, better known as SpaceX, is an American aerospace manufacturer and space transport services company headquartered in Hawthorne, California. It was founded in 2002 by entrepreneur Elon Musk with the goal of reducing space transportation costs and enabling the colonization of Mars. SpaceX has since developed the Falcon launch vehicle family and the Dragon spacecraft family, which both currently deliver payloads into Earth orbit.", "SpaceX's achievements include the first privately funded liquid-propellant rocket to reach orbit (Falcon 1 in 2008); the first privately funded company to successfully launch, orbit, and recover a spacecraft (Dragon in 2010); the first private company to send a spacecraft to the International Space Station (Dragon in 2012), and the first propulsive landing for an orbital rocket. As of March 2017, SpaceX has since flown ten missions to the International Space Station (ISS) under a cargo resupply contract. NASA also awarded SpaceX a further development contract in 2011 to develop and demonstrate a human-rated Dragon, which would be used to transport astronauts to the ISS and return them safely to Earth.", "SpaceX announced in 2011 they were beginning a privately funded reusable launch system technology development program. In December 2015, a first stage was flown back to a landing pad near the launch site, where it successfully accomplished a propulsive vertical landing. This was the first such achievement by a rocket for orbital spaceflight. In April 2016, with the launch of CRS-8, SpaceX successfully vertically landed a first stage on an ocean drone-ship landing platform. In May 2016, in another first, SpaceX again landed a first stage, but during a significantly more energetic geostationary transfer orbit mission. In March 2017, SpaceX became the first to successfully re-launch and land the first stage of an orbital rocket.", "In 2016, CEO Elon Musk unveiled the mission architecture of the Interplanetary Transport System program, an ambitious privately funded initiative to develop spaceflight technology for use in manned interplanetary spaceflight, and which, if demand emerges, could lead to sustainable human settlements on Mars over the long term. This is the main purpose this System was designed for. In 2017, Elon Musk announced that the company had been contracted by two private individuals to send them in a Dragon spacecraft on a free return trajectory around the Moon. Provisionally launching in 2018, this could become the first instance of lunar tourism."]
# # stopped_tokens = [i for i in tokens if not i in en_stop]
# en_stop = get_stop_words('en')
# docs = []
# for p in paragraphs:
#     plist = []
#     for sent in p.split('.'):
#         line = [word for word in sent.lower().split() if ((word.isalpha() or is_number(word)) and (not word in en_stop))]
#         if line:
#             plist.append(line)
#     docs.append(plist)
#
# text = [s for p in docs for s in p]
#
# dictionary = corpora.Dictionary([s for p in docs for s in p])
# paragraph1 = [text for text in docs[0]]
# paragraph2 = [text for text in docs[1]]
#
# corpus1 = [dictionary.doc2bow(s) for s in paragraph1]
# corpus2 = [dictionary.doc2bow(s) for s in paragraph2]
# all_corpus = [dictionary.doc2bow(t) for t in text]
# lda = gensim.models.ldamodel.LdaModel(all_corpus, num_topics=1, id2word=dictionary, passes=20)
# lda.print_topics(num_topics=1, num_words=

# BELONG TO WORD2ID CLASS
    # def __iter__(self):
    #     for root, dirs, files in os.walk(self.dirname):
    #         for filename in [ file for file in files if file.endswith("_clean")]:
    #             file_path = root + '/' + filename
    #             data = open(file_path).read()
    #             docs = [x.strip() for x in data.split("\n--EOD--\n") if x]
    #             for doc in docs:
    #                 paragraphs = doc.split("\n")
    #                 for p in paragraphs:
    #                     is_alpha_word_line = [word for word in
    #                                           p.split()
    #                                           if word.isalpha() or is_number(word)]
    #                     if is_alpha_word_line:
    #                         yield is_alpha_word_line


# TO READ FORMATTED FILES
# data = open("file").read()
# docs = [ doc.strip() for doc in dataA.split("--EOD--") if doc.strip() ]
# paragraphs = docs[0].split("\n")
# sentence = paragraphs[0].split()

import nltk
from nltk.corpus import stopwords

word_list = open("xxx.y.txt", "r")
stops = set(stopwords.words('english'))