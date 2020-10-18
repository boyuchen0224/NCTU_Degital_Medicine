## Case1 Smoking Status Detection

## Requirement
- **Jupyter notebook** with **Python 3.7.3**
- `pip3 install -r requirement.txt`
- Run [**analysis.ipynb**](https://github.com/boyuchen0224/NCTU_Digital_Medicine/blob/main/Case1_Smoking_Status_Detection/analysis.ipynb) on jupyter notebook
- Our [**final predictions**](https://github.com/boyuchen0224/NCTU_Digital_Medicine/blob/main/Case1_Smoking_Status_Detection/final_prediction.txt)

## Data pre-processing
**Step 1** : Tokenizing strings in list of strings
**Step 2** : Lemmatization with NLTK
**Step 3** : Stemming words with NLTK
**Step 4** : Removing stop words with NLTK in Python

## Manual data analysis
- Find smoke keywords : 
**smoke, tobacco, cigarette**
- Find detection keywords : 
**deny, no, not, negative, quit, has, was, past, former, ex, is, current**

## Traing and Testing
**Step 1** : **Data pre-processing** 
**Step 2** : **Filter keywords** as mentioned above
**Step 3** : Get **tf-idf** values
**Step 4** : Random **split data** to training and testing
**Step 5** : Use [**SklearnClassifier**](https://pythonprogramming.net/sklearn-scikit-learn-nltk-tutorial/) to train and predict
**Step 6** : Output accuracy and predictions

## Resources
- [GeeksforGeeks](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/?ref=lbp)
- [python使用scikit-learn计算TF-IDF](https://blog.csdn.net/u012052268/article/details/79560768)
- [tfidf : max_df, min_df](https://t.codebug.vip/questions-194618.htm)
- [nltk](https://www.nltk.org/book/)
- [sklearn - classification_report](https://www.cnblogs.com/178mz/p/8558435.html)
- [youtube - nlp tutorial](https://www.youtube.com/watch?v=nxhCyeRR75Q&list=PLIG2x2RJ_4LTF-IIu7-J3y_yg8LRe1WZq&ab_channel=MachineLearningTV)
- [youtube - nlp with nltk](https://www.youtube.com/watch?v=X2vAabgKiuM&t=671s&ab_channel=freeCodeCamp.org)
- [youtube - NLP for Text Classification with NLTK & Scikit-learn | Eduonix](https://www.youtube.com/watch?v=G4UVJoGFAv0&ab_channel=EduonixLearningSolutions)
- [決策樹(Decision Tree)以及隨機森林(Random Forest)介紹](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)
- [sklearn中的梯度下降法（SGD）](http://d0evi1.com/sklearn/sgd/)