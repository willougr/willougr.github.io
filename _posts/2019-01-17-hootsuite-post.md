---
layout: post
title: Setting up a Machine Learning Framework for Production
---
_Note: This blog post was originally written while I was on a co-op term at Hootsuite in the fall of 2017 and published on the [Code@Hootsuite blog](code.hootsuite.com) as [Setting up a Machine Learning Framework for Production](http://code.hootsuite.com/setting-up-a-machine-learning-framework-for-production/). I've duplicated and updated it here in order to bring everything under one roof._

Here at Hootsuite we're always looking for better ways to utilize our time and new technology. That's why we're looking at building an environment that helps developers leverage machine learning (ML) in production and minimize the overhead (i.e. amount of technical debt) required.

**The Problem**

Currently, ML is often done on a very ad-hoc basis. That's because there is no standardized workflow—as of yet—to deal with many of the unique difficulties associated with ML. At Hootsuite, we identified four key components that are needed:

* Data: All training and verification data needs to be validated and versioned so if something goes wrong, we can effectively track down the issue if it's related to our data.
* Model Training: We don't want to be reinventing the wheel every time we need to train a new model (even if the APIs for TensorFlow and scikit-learn do make that process easier).
* Model Validation: Processes need to be put in place to make sure that when we update models in a production environment they actually perform as expected and bring measurable benefits.
* Infrastructure: Broadly, this is everything from being able to easily switch out models in production to making sure that they can be accessed in a uniform format so we're not writing new code when we want to add ML to an existing product (i.e. it should be as easy as making an API call with the data that we want analyzed).

When we lack a standardized process for those four key components we have two issues: teams replicating code when they don't need to be and fragile infrastructure that is based off of unique constraints rather than being robust and extendable across multiple issues. This is the technical debt that we want to minimize from the outset.

**Technical Debt**

![Technical Debt](/img/2019-01-17-hootsuite-post/ml-technical-debt.png)
Photo source: [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) by D. Sculley et al.

Given the complexity of ML systems, it's unsurprising that they can contain some areas of technical debt. What was surprising to me however, was the sheer number of ways that that technical debt can arise. This comes from the actual ML model only being a tiny fraction of the code required to put it into production. D. Scully et al. from Google give a detailed overview of the numerous pitfalls that one has to be aware of when designing a ML system [here](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf). In short, they identify seven keys areas with numerous subcategories:

* Complex Models Erode Boundaries
  * Entanglement: Since ML systems mix all of the information they receive together in order to understand it, no input is actually independent. This means that  Changing Anything Changes Everything.
  * Correction Cascades: Sometimes models that solve slightly different problems are layered on top of each other in order to reduce training time by taking the original model as an input. This creates dependencies that can prevent a deeply buried model from being upgraded because it would reduce system-wide accuracy.
  * Undeclared Consumers: Similar to visibility debt in more classical software engineering, if a consumer uses outputs from a ML model, this creates tight coupling (with all of the potential issues arising from that) and potential hidden feedback loops.
* Data Dependencies
  * Unstable: If ownership of the service producing the input data and ownership of the model consuming it are separate, then potential changes to the input data could break the model's predictive abilities.
  * Underutilized: Some inputs provided limited modelling benefit and if they are changed, there can be consequences. There is a tradeoff to be made about complexity and accuracy (as in most software systems).
* Feedback Loops
  * Direct: Occasionally, such as with bandit algorithms, a model may be able to influence what training data it will use in the future.
  * Hidden: More difficult to recognize than direct loops, these occur when two or more systems interact out in the real world. Imagine if the decisions made by one system affect the inputs of another, changing its output. This output then affects the inputs of the first system, leading each system to optimize behaviour for each other.
* ML-System Anti-Patterns
  * Glue Code: Any code that is needed to transform data so that it can be plugged into generic packages or existing infrastructure.
  * Pipeline Jungles: These appear when data is transformed from multiple sources at various points through scraping, table joins and other methods without there being a clear, holistic view of what is going on.
  * Dead Experimental Codepaths: The remnants of alternative methods that have not been pruned from the codebase. The expectation is that these will not be hit but they could be used in certain real world situations and create unexpected results.
* Configuration Debt
  * This is oftentimes viewed as an afterthought even though the number of lines of configuration required to make a model work in the real world can sometimes exceed the number of lines of code.
* Changes in the External World
  * Fixed Thresholds in Dynamic Systems: If a decision threshold is manually set and then the model is updated using new training data, the previous threshold may be invalid.
  * Monitoring and Testing: Since ML models operate in the real-world, they need real-world monitoring to make sure they work. Three areas where it may be useful to focus on monitoring are:
    * Prediction Bias: the distribution of predicted labels should be equal to the distribution of observed labels
    * Action Limit: systems that take actions in the real-world should have a broad limit on how often that can (or cannot) happen
    * Up-Stream Producers: data pipelines that lead to the model needing to be tested and maintained
* Other
  * Data Testing: Data should be tested on a continuous basis using standardized tests rather than one-off analysis when initially set up.
  * Reproducibility: In an ideal world, the results of a ML model should be reproducible by anyone with the same data and the same source code.
  * Process Management: As many system-level processes, such as updating models and data, assigning computing resources, etc. should be as automated as possible.
  * Cultural: Teams need to reward the cleaning up of models and technical debt as well as improving accuracy when it comes to allocating development time and resources.

As D. Scully et al. have shown, there's a lot more than just tweaking the model that we need to be concerned with when we want to use ML in production environments. The more work we can standardize, the less technical debt we will need to deal with on an ongoing basis.

**Current Offerings**

Currently, there are limited options for all of the infrastructure and versioning needs around trying to deploy a model in production. One of the options out there is [Amazon Machine Learning](https://aws.amazon.com/aml/) (AML). The nice thing about AML is that it plays nice with data imports from S3 and Redshift, both services currently in use at Hootsuite. The not-so-nice things about AML is that you can only build one kind of model (a logistic regression) and any tuning you might want to make needs to be done through their platform using their built-in 'recipe' language. This is less than ideal for the range and complexity of models that we want to be deploying across Hootsuite.

_Side note: AML has been supplanted by [AWS SageMaker](https://aws.amazon.com/blogs/aws/sagemaker/) which was announced at AWS re:Invent this year (2017). It looks like it could have some useful applications and does seem to handle many of the issues raised here._

_2019 Update: Amazon, along with the other cloud providers have continued to pour more resources into building out their managed machine learning options over the past year. Going with a managed product definitely has its benefits that were glossed over in the original writing of this post since they were not as well developed at the time. I'm not sure I would go with the same recommendation today as below._

So, there's not a commercial off-the-shelf solution that would solve our problems. What to do? Build one ourselves! This lack of availability means we are evaluating open-source alternatives that we could cobble together into a cohesive package that will fit all of our needs. That search is what led us to [TensorFlow Extended](http://www.kdd.org/kdd2017/papers/view/tfx-a-tensorflow-based-production-scale-machine-learning-platform).

**TensorFlow Extended**

A long time ago at KDD2017 in August 2017, a team of Google developers presented their solution to the problem we are having: building a production-scale platform for effective ML. While, unfortunately for us, not all of their work was released as open-source, their analysis and framework gave us a starting place and guidance on how to do this at Hootsuite.

Denis Baylor et al. presented a similar (and more detailed) framework compared to what we had already developed internally about what exactly was needed to build a ML platform. They also had a number of constraints for what they wanted at Google:

* Building one ML platform for many different learning tasks
* Continuous training and serving
* Human-in-the-loop
* Production-level reliability and scalability

Their eventual system design ended up looking like this:

![System Design](/img/2019-01-17-hootsuite-post/ml-system-design.png)
Photo source: [TFX: A TensorFlow-Based Production-Scale Machine Learning Platform](http://www.kdd.org/kdd2017/papers/view/tfx-a-tensorflow-based-production-scale-machine-learning-platform) by Denis Baylor et al.

While this is an ideal Google-scale solution to our ML problem, it's not necessarily a Hootsuite-scale solution. So we now have an idea of what components we want; this idea has been validated as being similar to the metrics Google used for the design of their own system. Next steps: figure out what open-source tooling is out there for each of those components so we don't reinvent the wheel.

**Hootsuite Requirements**

Since we're operating at significantly different scales, there's a number of components that Google is using that we can drop from consideration for now (though who knows where we'll be at in 5 years). We don't need to be building an integrated front-end since there simply is not enough demand for it as of yet. Garbage collection is not important given the size of data we are working with and data access controls are already being implemented. And finally, for the moment, each team will need to be responsible for maintaining and documenting their own ML pipelines. While we're ignoring pipelines for the moment, there is interesting work being done on [trying to simplify construction on a distributed system](http://jmlr.org/papers/volume17/15-237/15-237.pdf) and even [automating pipeline construction entirely](http://shivaram.org/publications/keystoneml-icde17.pdf).

So that leaves us with categories to focus on: All Things Data; Cradle-to-Grave Models; Serving It Right.

**All Things Data**

Data is what drives ML so obviously, there are some pretty extensive infrastructure needs when it comes to handling it.

* Data Versioning: The short story is versioning data is hard, really hard. The long story is slightly more complicated. There's a good overview of why it's hard and some principles about how to tackle it [here](https://github.com/leeper/data-versioning). What it boils down to is that because there are so many different formats data can be stored in, there is not an easy-to-use and efficient option yet, like a git for data. Git works by tracking changes in a line in a text file, and it's just not feasible to store all of the data we may want to use as a csv file. GitHub does offer [Git Large File Storage](https://git-lfs.github.com/), though it has some issues. That big issue is that it stores each version of the data as another file rather than just tracking differences. That means having 10 versions of a 10GB dataset would take up 100GB of space! That's going to get very expensive very quickly. As a result, we're currently doing the same thing in principle but in S3 because that's where our data lives anyways. However, the process is by no means ideal because it opens up the opportunity for more human error and takes up lots of space.
* Data Ingestion: Since we're storing everything in S3 at the moment and not dealing with datasets that are too, too big (yet), right now we're just pulling in the data directly from there. This works given our needs and workflow but could definitely use some optimization as the data we handle gets bigger and bigger. Connecting compute resources in Redshift directly to storage in S3 through Redshift Spectrum definitely seems one promising avenue to pursue here.
* Data Cleaning: Cleaning data is normally a very manual process: clean some data, test it, clean some more, do a bit of analysis, and on and on. Depending on the nature of the project, this could also be beneficial as you get to understand the data better. However, there is tooling out there that could help with process (as long as the ML model being built is convex loss). That tooling is [ActiveClean](https://activeclean.github.io/files/activeclean-vldb16.pdf). For those specific use cases, this could help clean the data quicker and more efficiently than manual processing.
* Data Analysis: This is where things start to get a bit interesting. Every dataset is unique and every business need it is being used to solve is different. However, there are a number of descriptive statistics that could be calculated on included features and over their values to provide insight into the shape of each dataset. This would give us a good overview of how the data looks which means anomalies could be spotted sooner. A number of existing libraries in various languages could be used to do this efficiently.
* Data Transformation: This is going to be very specific to each ML model developed so limited optimizations can be done. However, making sure that any assets that need to be reused for both training and prediction (such as vectorizers) are automatically exported would remove one more place for human error.
* Data Validation: To do data validation, we (and Google) would rely on having a schema that defines what kind of data should be in the dataset. Some examples that Google uses to define its schemas are:
  * Features present in the data
  * The expected type of each feature
  * The expected presence of each feature
  * The expected valency of the feature in each example
  * The expected domain of a feature

As part of a separate project to organize all of the data in use at Hootsuite, we'll have schemas for existing datasets and new ones could be produced automatically as data is fed into a ML model. Not all of the definitions are necessary but at a minimum, the features present and the expected type would be a good starting point to confirm.

**Cradle-to-Grave Models**

* Model Versioning: Just like with software, we need a way to be able to keep track of how models evolve over time so that if a new version is less effective than before, it's easy to roll back. Otherwise, it's just a free-for-all of updates with no governance. This is still a relatively new area of focus; however, there are some interesting open-source solutions, such as [Data Version Control](https://dataversioncontrol.com/), out there that at least attempt to solve this problem.
* Model Training: When training a model, a process that could potentially take days, we want to be able to streamline the process as much as possible. Google has a great idea on how to do that: warm-starting the models. Basically, this means relying on transfer learning to generalize a set of parameters that represent the base state in order to help initialize the new training state. Rather than starting a new version of model from scratch, it is instead given a fuzzy view of the world. Thus, training time is significantly shorter. Being able to optimize this time spent is definitely something to keep in mind as we build production infrastructure.
* Model Evaluation: Using Google's definition of evaluation as "human-facing metrics of model quality", we want to be able to easily help teams prove their models work. This would basically be applying the model to offline data (so we're not having to deal with real-time user traffic) and figuring out a way to translate training loss to whatever business metric is most relevant to the experiment. While this can—and is—done on an ad-hoc basis, it could also be automated in order to save developer time and reduce glue code.
* Model Validation: Again, from Google, validation can be seen as "machine-facing judgment of model goodness". These metrics basically boil down to tests that can be run on models in production against a baseline (i.e. the previous version of the model). If the new version is not beating the previous one, it gets taken offline and the previous version is rolled out. Automating this would help make sure that models are actually providing value to the business when used.

**Serving It Right**

* Development Workflow: We want to make sure that the development workflow is as seamless as possible. This means making sure that devs can hook Jupyter Notebooks up to GPUs on the cloud and that they have all the libraries they need pre-installed and ready to run. We also need to be putting some guidelines in place about how to be developing models at Hootsuite so we don't end up with a different standard for each team.
* Serving: Once the model has been developed, we need to be able to actually serve it in production. This is where Google—again—comes in. With [TensorFlow Serving](https://www.tensorflow.org/serving/), we're able to create a docker container that can be deployed on Kubernetes and an API endpoint we can hit with any of our microservices in order to get predictions. This means that any ML that happens can be easily folded into our existing microservice architecture.

**Where we're at**

Right now, we're working on developing the base framework that will underlie the eventual goals outlined here. We need to make sure the foundation is solid in order to build on it. It's great to have the example of companies like Google that have put their infrastructure details out there so that we can take inspiration from them when we're building our own. Machine learning definitely has a place at Hootsuite; we just need to make it as painless as possible.

**Further reading**

If you're interested in a much deeper dive on this topic, check out *Machine Learning Logistics: Model Management in the Real World* by Ted Dunning and Ellen Friedman from O'Reilly Media. They examine some really interesting nuances of serving models in the wild as well as introduce a new design pattern: the rendezvous architecture. It's a worthwhile read.