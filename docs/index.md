## Who should we offer a free coffee? You? Or maybe your friend?

Many offers by all kinds of companies, but especially coffee houses, seem very misplaced and do not motivate the customer to use it. 
But it would be very interesting for a company to find out which users are going to be happy to take the offer to make offers specifically to customers. This is what this blog post is about. 

![Image](coffee-2608864_1920.jpeg)

Now let's see what makes you get the most out of your career.

# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


# Problem Statement

We would like to build a model to predict whether a user answers an offer or not, to enable users to predict which personas a offer could reach. 

For that, we will train a classifier on features based on properties of the persons and offers in the past to see for which kind of offer which persona is likely to accept it. 

We would expect a model to at least perform better than just guessing not responding or responding for all entries in the test set.

# Metrics

We choose the F1 score as a metric because we are interested in both a high precision and recall to predict both the users accepting an offer and declining an offer as accurate as possible.

# Data Exploration

There are 10 offers in the dataset, of which 4 are each buy-one-get-one-free and 4 are discounts, and 2 are informational. 

![image](https://user-images.githubusercontent.com/32015957/132355242-fdebc651-07da-434a-8415-16fb9e031e6d.png)

The income of the customers is almost normally distributed, with most customers earning between 50-60k $ per year.

![image](https://user-images.githubusercontent.com/32015957/132355294-ef56b4e0-91f5-4468-a43c-cb65ef56f2c9.png)

# Data Preprocessing 

We build the new offers DataFrame, which contains the offer_id, person and a list of the unique events that belong to this offer and person. As we later want to predict whether a person reacted to an offer, we first add a list of the events that happened with a specific person and offer, and then add a categorical variable whether the user reacted to this offer or not.
We then merge all 3 dataframes together to add the properties of the offer and the person to the offers Dataframe.

# Comparing models

We chose the decision tree of the top answer from this post (https://stackoverflow.com/questions/2595176/which-machine-learning-classifier-to-choose-in-general) as a guideline for which models to try. This led us to try out LinearSVC and KNeighbours Classifier.

Both LinearSVC and KNeighbours Classifier work pretty well compared to just guessing 0 for all values, which would give 0.58, as both are already around 0.7 f1-score. When improving on params, the KNeighbours Classifier gets slightly better. 

# Conclusion

In the end, it seems that KNeighbors Classifier performs the best. 

Especially interesting on this project was the preparation and Cleaning of the data. This dataset seemed far more realisitic than others, as most of the features had to be worked out in detail.

Very interesting for me was also the implementation of scaling, which improved all models instantly by 0.1 f1 score. 

Improvements could be made by choosing even more models to compare, e.g. SVC or Ensemble classifiers could be an option. 
