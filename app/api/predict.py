import re
import joblib
import logging
from joblib import load

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()

classifier = joblib.load("app/api/assets/classifier.pkl")
tfidfVectorizer = joblib.load("app/api/assets/tfidfVectorizer.pkl")

stops = {'cannot', 'sometime', 'put', 'take', 'whereas', 'had', 'which', 'even', 'see', 'those', 'can', 'another', 'five', 'down', 'myself', 'whether', 'an', 'less', 'because', 'be', 'thereupon', 'forty', 'did', 'due', 'any', 'thus', 'when', 'into', 'might', 'eleven', 'hereafter', 'over', 'does', 'quite', 'same', 'while', 'herein', 'own', 'ca', 'keep', 'hers', 'sometimes', "n't", 'everything', 'becoming', 'give', 'until', '‘m', 'now', 'ours', 'very', 'yourselves', "'re", 'by', '’ll', 'against', 'hereupon', 'these', 'whence', 'three', 'someone', '’re', 'the', 'here', 'then', 'onto', 'why', 'whom', '‘s', 'them', 'they', 'herself', 'something', 'else', 'am', 'get', 'just', 'whole', 'done', 'n’t', '’ve', 'twelve', 'seeming', 'thru', 'whenever', 'has', 'however', 'between', 'no', 'our', 'beforehand', 'wherever', 'became', '‘ll', 'first', 'mine', 'last', 'ourselves', 'six', 'their', 'i', 'than', 'formerly', 'therefore', 'do', '’s', 'move', 'otherwise', 'may', 'only', 'but', 'seemed', 're', 'whose', 'one', 'among', 'become', 'how', 'show', 'back', 'from', 'regarding', 'yet', 'every', 'fifteen', 'made', 'where', "'m", "'ve", 'although', 'so', 'beyond', 'more', 'none', 'nevertheless', 'under', 'about', 'it', 'others', 'yourself', 'afterwards', 'nothing', 'below', 'latterly', 'within', 'front', 'for', 'namely', 'ever', 'part', 'whereafter', 'amongst', 'eight', 'top', 'whoever', 'off', 'seems', 'before', 'been', 'have', 'always', 'his', 'alone', 'few', 'around', 'hereby', 'he', 'everyone', 'latter', 'nobody', 'nowhere', 'other', 'further', 'except', 'ten', 'thence', 'fifty', 'whither', 'your', 'meanwhile', 'would', 'some', "'s", 'are', 'thereafter', 'thereby', 'empty', 'various', '’m', 'serious', 'please', 'him', 'behind', 'who', 'next', 'several', 'third', 'itself', 'was', 'at', 'through', 'must', 'amount', 'besides', 'throughout', 'anyone', 'if', 'as', 'go', 'elsewhere', 'were', 'after', 'each', 'during', 'bottom', 'everywhere', 'a', 'say', 'using', 'should', 'she', 'somehow', 'though', 'up', '‘ve', 'again', 'n‘t', 'both', 'many', 'to', 'full', 'nor', "'ll", 'per', '‘re', 'two', 'out', 'all', 'somewhere', 'along', 'its', 'side', 'therein', 'unless', 'whatever', 'yours', 'also', 'anyway', 'enough', 'being', 'moreover', 'noone', 'that', 'above', 'well', 'my', 'towards', 'sixty', 'really', 'already', 'such', 'make', 'via', 'wherein', "'d", 'four', 'since', 'becomes', 'and', 'too', 'themselves', 'toward', 'could', '‘d', 'you', 'not', 'seem', 'twenty', 'never', 'anything', 'almost', 'doing', 'anywhere', 'himself', 'most', 'this', 'once', 'perhaps', 'still', 'anyhow', 'either', 'much', 'will', 'without', 'rather', 'what', 'upon', 'hundred', 'of', 'often', 'together', 'whereby', 'in', 'beside', 'former', 'there', 'is', 'me', 'nine', 'across', 'used', 'neither', 'or', 'we', '’d', 'mostly', 'indeed', 'on', 'hence', 'least', 'name', 'call', 'with', 'whereupon', 'her', 'us'}

def tokenize(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    words = letters_only.lower().split()                                              
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words ))

def get_prediction(input):
    array1 = []
    cleaned1 = tokenize(input)
    array1.append(cleaned1)

    x_tfid = tfidfVectorizer.transform(array1).toarray()
    answer = classifier.predict(x_tfid)
    answer = str(answer[0])
    return answer

@router.post('/predict')
async def predict(item: str):
    """
    # Kickstarter Project NLP Model to Evaluate any Kickstarter Project's Likelihood of Success at being fully funded.
    
    ### 1) Frist click "try it out."
    
    ### 2) Then enter a description of your project, some examples of successful campaigns include: 
    - A fast, fun, easy-to-learn, and easy-to-play strategic card game for the whole family.
    - Dice created to ensure that things escalate quickly! Made from the finest 6061 Aluminum, the Fibonacci dice go from oh to OH!
    - A short film about love...
    - A compilation album and accompanying website that features 35 songs based on Sol LeWitt's 35 Sentences on Conceptual Art.

    ### 3) Next click execute.
    
    ### Output) A response of whether or not the project is likely to meet it's funding goal and be successful.  In this case it will be a "1" meaning the model thinks this could be a successfully funded project!
    
    ## Needed Info:
    - `item`: (str) Project Description Entered.
    
    ## Response: (str)
    - Whether or not the kickstarter project is likely to be a success or not. "1" is a success, "0" is guessed to be a failure.
    """

    success_failure = get_prediction(item)
    return {
        'success_failure' : success_failure
    }
