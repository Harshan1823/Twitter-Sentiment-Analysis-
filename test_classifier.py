import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

file = open('output', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

logprior = data[0]
loglikelihood = data[1]

def clean_review(review):
    ps = PorterStemmer()
    stop_words = stopwords.words('english')+['']
    review = re.sub(r'[^\w\s]','', review)
    review = re.sub(r'http\S+','', review)
    review_cleaned = [word.lower() for word in review.split(' ') if word.lower() not in stop_words]
    review_cleaned = [ps.stem(word) for word in review_cleaned]
    review_cleaned = ' '.join(review_cleaned)  
    return review_cleaned

def naive_bayes_predict(review, logprior, loglikelihood):
    word_l = clean_review(review).split()
    total_prob = 0

    total_prob += logprior

    for word in word_l:
        if word in loglikelihood:
            total_prob = total_prob + loglikelihood[word]
    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0
    return total_prob

my_review = input("Enter your review: ")
ans = naive_bayes_predict(my_review, logprior, loglikelihood)
if(ans == 1):
    print("Negative")
else:
    print("Positive")