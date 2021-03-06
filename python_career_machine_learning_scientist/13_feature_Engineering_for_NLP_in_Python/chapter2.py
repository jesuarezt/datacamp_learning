#Tokenizing the Gettysburg Address
#
#In this exercise, you will be tokenizing one of the most famous speeches of all time: the Gettysburg Address delivered by American President Abraham Lincoln during the American Civil War.
#
#The entire speech is available as a string named gettysburg.


import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)

#Lemmatizing the Gettysburg address
#
#In this exercise, we will perform lemmatization on the same gettysburg address from before.
#
#However, this time, we will also take a look at the speech, before and after lemmatization, and try to adjudge the kind of changes that take place to make the piece more machine friendly.

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))


#Cleaning a blog post
#
#In this exercise, you have been given an excerpt from a blog post. Your task is to clean this text into a more machine friendly format. This will involve converting to lowercase, lemmatization and removing stopwords, punctuations and non-alphabetic characters.
#
#The excerpt is available as a string blog and has been printed to the console. The list of stopwords are available as stopwords.

stopwords = ['fifteen',
 'noone',
 'whereupon',
 'could',
 'ten',
 'all',
 'please',
 'indeed',
 'whole',
 'beside',
 'therein',
 'using',
 'but',
 'very',
 'already',
 'about',
 'no',
 'regarding',
 'afterwards',
 'front',
 'go',
 'in',
 'make',
 'three',
 'here',
 'what',
 'without',
 'yourselves',
 'which',
 'nothing',
 'am',
 'between',
 'along',
 'herein',
 'sometimes',
 'did',
 'as',
 'within',
 'elsewhere',
 'was',
 'forty',
 'becoming',
 'how',
 'will',
 'other',
 'bottom',
 'these',
 'amount',
 'across',
 'the',
 'than',
 'first',
 'namely',
 'may',
 'none',
 'anyway',
 'again',
 'eleven',
 'his',
 'meanwhile',
 'name',
 're',
 'from',
 'some',
 'thru',
 'upon',
 'whither',
 'he',
 'such',
 'down',
 'my',
 'often',
 'whether',
 'made',
 'while',
 'empty',
 'two',
 'latter',
 'whatever',
 'cannot',
 'less',
 'many',
 'you',
 'ours',
 'done',
 'thus',
 'since',
 'everything',
 'for',
 'more',
 'unless',
 'former',
 'anyone',
 'per',
 'seeming',
 'hereafter',
 'on',
 'yours',
 'always',
 'due',
 'last',
 'alone',
 'one',
 'something',
 'twenty',
 'until',
 'latterly',
 'seems',
 'were',
 'where',
 'eight',
 'ourselves',
 'further',
 'themselves',
 'therefore',
 'they',
 'whenever',
 'after',
 'among',
 'when',
 'at',
 'through',
 'put',
 'thereby',
 'then',
 'should',
 'formerly',
 'third',
 'who',
 'this',
 'neither',
 'others',
 'twelve',
 'also',
 'else',
 'seemed',
 'has',
 'ever',
 'someone',
 'its',
 'that',
 'does',
 'sixty',
 'why',
 'do',
 'whereas',
 'are',
 'either',
 'hereupon',
 'rather',
 'because',
 'might',
 'those',
 'via',
 'hence',
 'itself',
 'show',
 'perhaps',
 'various',
 'during',
 'otherwise',
 'thereafter',
 'yourself',
 'become',
 'now',
 'same',
 'enough',
 'been',
 'take',
 'their',
 'seem',
 'there',
 'next',
 'above',
 'mostly',
 'once',
 'a',
 'top',
 'almost',
 'six',
 'every',
 'nobody',
 'any',
 'say',
 'each',
 'them',
 'must',
 'she',
 'throughout',
 'whence',
 'hundred',
 'not',
 'however',
 'together',
 'several',
 'myself',
 'i',
 'anything',
 'somehow',
 'or',
 'used',
 'keep',
 'much',
 'thereupon',
 'ca',
 'just',
 'behind',
 'can',
 'becomes',
 'me',
 'had',
 'only',
 'back',
 'four',
 'somewhere',
 'if',
 'by',
 'whereafter',
 'everywhere',
 'beforehand',
 'well',
 'doing',
 'everyone',
 'nor',
 'five',
 'wherein',
 'so',
 'amongst',
 'though',
 'still',
 'move',
 'except',
 'see',
 'us',
 'your',
 'against',
 'although',
 'is',
 'became',
 'call',
 'have',
 'most',
 'wherever',
 'few',
 'out',
 'whom',
 'yet',
 'be',
 'own',
 'off',
 'quite',
 'with',
 'and',
 'side',
 'whoever',
 'would',
 'both',
 'fifty',
 'before',
 'full',
 'get',
 'sometime',
 'beyond',
 'part',
 'least',
 'besides',
 'around',
 'even',
 'whose',
 'hereby',
 'up',
 'being',
 'we',
 'an',
 'him',
 'below',
 'moreover',
 'really',
 'it',
 'of',
 'our',
 'nowhere',
 'whereby',
 'too',
 'her',
 'toward',
 'anyhow',
 'give',
 'never',
 'another',
 'anywhere',
 'mine',
 'herself',
 'over',
 'himself',
 'to',
 'onto',
 'into',
 'thence',
 'towards',
 'hers',
 'nevertheless',
 'serious',
 'under',
 'nine']

 # Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))


#Cleaning TED talks in a dataframe
#
#In this exercise, we will revisit the TED Talks from the first chapter. You have been a given a dataframe ted consisting of 5 TED Talks. Your task is to clean these talks using techniques discussed earlier by writing a function preprocess and applying it to the transcript feature of the dataframe.
#
#The stopwords list is available as stopwords.

# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)
  
# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])


#POS tagging in Lord of the Flies
#
#In this exercise, you will perform part-of-speech tagging on a famous passage from one of the most well-known novels of all time, Lord of the Flies, authored by William Golding.
#
#The passage is available as lotf and has already been printed to the console.

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)



#Counting nouns in a piece of text
#
#In this exercise, we will write two functions, nouns() and proper_nouns() that will count the number of other nouns and proper nouns in a piece of text respectively.
#
#These functions will take in a piece of text and generate a list containing the POS tags for each word. It will then return the number of proper nouns/other nouns that the text contains. We will use these functions in the next exercise to generate interesting insights about fake news.
#
#The en_core_web_sm model has already been loaded as nlp in this exercise.


nlp = spacy.load('en_core_web_sm')

# Returns number of proper nouns
def proper_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of proper nouns
    return pos.count('PROPN')

print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))

nlp = spacy.load('en_core_web_sm')

# Returns number of other nouns
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count('NOUN')

print(nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))



#Noun usage in fake news
#
#In this exercise, you have been given a dataframe headlines that contains news headlines that are either fake or real. Your task is to generate two new features num_propn and num_noun that represent the number of proper nouns and other nouns contained in the title feature of headlines.
#
#Next, we will compute the mean number of proper nouns and other nouns used in fake and real news headlines and compare the values. If there is a remarkable difference, then there is a good chance that using the num_propn and num_noun features in fake news detectors will improve its performance.
#
#To accomplish this task, the functions proper_nouns and nouns that you had built in the previous exercise have already been made available to you.


headlines['num_propn'] = headlines['title'].apply(proper_nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively"%(real_propn, fake_propn))


headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively"%(real_noun, fake_noun))

#
#Named entities in a sentence
#
#In this exercise, we will identify and classify the labels of various named entities in a body of text using one of spaCy's statistical models. We will also verify the veracity of these labels.

# Load the required model
nlp = spacy.load('en_core_web_sm')

# Create a Doc instance 
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)



#Identifying people mentioned in a news article
#
#In this exercise, you have been given an excerpt from a news article published in TechCrunch. Your task is to write a function find_people that identifies the names of people that have been mentioned in a particular piece of text. You will then use find_people to identify the people of interest in the article.
#
#The article is available as the string tc and has been printed to the console. The required spacy model has also been already loaded as nlp.


tc = "\nIt’s' been a busy day for Facebook  exec op-eds. Earlier this morning, Sheryl Sandberg broke the site’s silence around the Christchurch massacre, and now Mark Zuckerberg is calling on governments and other bodies to increase regulation around the sorts of data Facebook traffics in. He’s hoping to get out in front of heavy-handed regulation and get a seat at the table shaping it.\n"

def find_persons(text):
  # Create Doc object
  doc = nlp(text)
  
  # Identify the persons
  persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
  
  # Return persons
  return persons
  #test
find_persons('hello NLP!')
print(find_persons(tc))