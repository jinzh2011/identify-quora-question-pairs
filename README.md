# Identifying Quora Questions Pairs with Same Meanings

## Background
Human languages have complicated structures. We often express the same ideas or ask the same questions using different words and structures. This has enriched our language context, but also brought challenges for Q&A social platforms such as Quora and Stack Overflow.

Separate posts on the same question can dilute information sharing on Q&A platforms and create fragmented user experience. For this project, I solve this challenge by using NLP to predict if a pair of Quora questions have the same semantic meanings.


## Description
I implemented an XGBoost model using NLP features such as Fuzzy distance, tf-idf and Word2Vec to predict whether two Quora questions have the same semantic meanings. I also trained a deep learning model using LSTM Recurrent Neural Networks to improve the accuracy of the model.

## Result
My current model is an extreme gradient boosting  model with log loss of 0.48 and optimized accuracy of 74%.

More details are coming soon!

# Licenses
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
