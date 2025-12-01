import numpy as np
import joblib
import gensim.downloader as api

# Load the saved model
model = joblib.load('sentiment_model.pkl')

# Load word vectors (still needed for embeddings)
word_vectors = api.load("glove-wiki-gigaword-100")

def get_document_embedding(text, vectors, dim=100):
    words = text.lower().split()
    embeddings = [vectors[word] for word in words if word in vectors]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(dim)

# Test the model on new sentences
test_sentences = [
    "Is there any update on when this will be available in the UK? I am currently trialling the Xero Payroll, but I am struggling to see the point if it cannot do this most basic HR function?",
    "Leave reporting in Xero needs to be enhanced, both in terms of the current year but also historic. There is no way of getting even a simple list of who has taken what holiday when. Also an audit trail of changes to settings would be helpful - once the leave year rolls forward you can no longer see any bespoke pay calendars you might have set up",
    "Any update on this one? As others have said, it seems like it should be a basic feature.",
    "Could you please provide an update on the UK leave transactions report? It is urgently needed. Thanks",
    "Yup, need this too. We'll need to look at a different holiday booking tool if it doesn't become available soon.",
    "Can we have an update as to when this will be available in the UK, We are using Xero payroll but will switch soon if we cannot get an update.",
    "This should be a basic reporting function when managing employees annual leave on their system. Seems to be available in NZ & AU but not in the UK. Xero - you need to catch up with this quickly before customers start switching software.",
    "Urgently needed, please ensure it is not restricted to calendar year as many companies operate differently, I.e ours is April to March.",
    "This would be so helpful - we run a lot of payrolls in xero and often clients ask how much leave their staff have left but there is currently no quick way to get this information.",
    "Totally agree. And without a Xero report, how do we compare what is in Xero Me with what is actually accounted for in Payroll? There must be a way to run a report to make sure colleagues are being paid correctly.", 
    "This is a critical report as it pertains to staff member and ultimately to retention of staff members - with out a report we are often scrabbling around trying to find balances and trying to forecast work loads is almost impossible.",
    "Please fast forward this. Why can we not run a report to have visibility on our staff? Especially seeing as it is available in Australia?", 
    "Hi Victoria, If you go to reporting: https://reporting.xero.com/ and search Leave in the search box on the top right, do you have any options? In AU we have https://payroll.xero.com/Reports/LeaveTransactions report which we can export to Excel", 
    "Would be very helpful to have a report where we can export to Excel of employee's annual leave taken as dates from and to and total hours. I cannot find a report that does this. Xero search suggests there are reports but I do not have them as an option- contacted Support- reply was they not available in my region! UK."
]

for sentence in test_sentences:
    embedding = get_document_embedding(sentence, word_vectors)
    prediction = model.predict([embedding])[0]
    confidence = model.predict_proba([embedding])[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Text: '{sentence}'")
    print(f"Sentiment: {sentiment} (Confidence: {confidence[prediction]:.2%})\n")