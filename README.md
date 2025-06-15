> Training Testing Dataset used -> 
     IMDB Dataset of 50K Movie Reviews (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

> Also check -> Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/).

> For word vectorization GloVe dateset is used for word representation and there releation.

> glove.6B.100d.txt This file is required, get it from here - https://nlp.stanford.edu/projects/glove/



> To extract the model for future use add following code snippet to the code
'''python
import joblib
from google.colab import files

# Save scikit-learn models
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(naive_bayes_model, 'naive_bayes_model.pkl')

# Save LSTM model
model.save('lstm_model.h5')

# Save preprocessing tools
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(tokenizer, 'tokenizer.pkl')
'''

# To verify the saved files:
'''python
!ls
'''

# Download all saved files
'''python
files.download('logistic_model.pkl')
files.download('naive_bayes_model.pkl')
files.download('lstm_model.h5')
files.download('tfidf_vectorizer.pkl')
files.download('tokenizer.pkl')
'''
