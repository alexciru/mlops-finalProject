---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 

Group 23

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *s222913, sXXXXXX, sXXXXXX*
>
> Answer:
Q to verify: 23, 12, 15     
s222999, 

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In our project, we used the TIMM framework which is built on top of the PyTorch framework. It provides a set of pre-trained models and tools for fine-tuning those models on specific tasks. Using the TIMM framework we were able to download the MobileNetV3 (small) model architecture and also a pre-trained version of this model. Although the chosen model was trained with 1000 output classes using another functionality we were able to add an additional layer to the model’s architecture to map the output to our 43 classes from the GTRSB dataset. Using finetuning functionality we were able to update weights on all layers during training to adapt the MobileNetV3 model to recognize images from the GTRSB dataset.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

During the development of our project, we were using Anaconda to manage our virtual environments. A new member of the team would need to create an environment with the same libraries to be able to work with our project. To create a .txt file with the necessary requirements one needs to run “conda create --name <env> --file requirements.txt”. It is also possible to create a .txt file with pip using the “pip freeze > requirements.txt” command. A new member to join our project would need to be added to the Google Cloud, git repository, and wandb project using his email by one of the members.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

For the project, we used the cookiecutter data science project template. Where each of the folders had a specific purpose. The `data` folder is where the data is stored with different subfolders for the original data, the intermediated folder and the final data used for the model. Here we added a new folder to store the raw data from our dataset. The `doc` folder is where the documentation is stored. The `model` folder is where all the info related to the model such as checkpoints and predictions. `Notebooks` is used to store the Jupiters that were used during development, ` references` is used to store manuals, `reports` is where the Html and reports are generated, `src` is where all the source files are. These are divided into subfolders depending on the purpose of the files. We also added the tests folder where we store unit tests scripts


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We implemented pep8 guidelines in order to standardise the code style and flake and black libraries to check and adapt to these guidelines. It is important to use style guidelines, especially in big projects with different developers to accomplish consistency in the project making it easier to read and to understand different parts of the code.
## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

In our project, we have 2 tests to check if the model has been created correctly and 4 test to verify the correctness of the dataset. We implemented unittests to check if the loaded data has the correct size and later if all of the images in the used dataset have the correct shape. Our tests are also checking if every image has been loaded with a label. We implemented also 2 tests for the model, we are checking if the model’s output has the correct shape.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage is 65%. Despite covering more than 50% of the code it does not mean that the code is error-free. The coverage only determines the % of the lines of code that are being executed in the test. But this does not cover different inputs in functions or edge cases making the test an efficient way to test only some parts of the code. To improve the coverage for our project we should implement testing the training code and also the code responsible for making the predictions. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project we used both branches and pull requests for software version control. First we distributed tasks among group members, one person was responsible for creating the project structure and creating remote repository. Then rest of the members were invited to join the repository. All main features of the project were developed on separate branches. Once the feature has been developed person responsible was responsible for creating unit tests for this part. After the tests were implemented the code from branch has been pushed to the main. We only used pull requests on the branches that passed all the implemented tests and were accepted (if possible by another member). This process turned out to be beneficial in terms of maintaining the good quality of code and not introducing error to the main branch. Once the code from the develop branch has been merged with the main branch all members were notified that new version of exists and they should pull it to their local main branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC initially with the integration with google drive but we quickly moved to use DVC integrated with bucket storage in the cloud. During the project development, DVC enabled all members to get the original dataset using the terminal and DVC pull command without the struggle of extracting a huge file with 50000 images into the correct subdirectories inside the project. Later on, using DVC was very useful to ensure that all members were using data preprocessed in the same way. Using DVC also enables the storage of datasets that were preprocessed differently. With different datasets stored using DVC, it was easier to test the model on different data.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

TODO: add more
We created diferent workflows to have CI in our project. Initially we created a test workflow to check if the code that was being pushed could pass the unit testing to protect the main branch. Also the isort workflow was added to sort the imports alphabetially and ensure consistency in the project.


Pre-commit
generate docker img ??


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

In our project, we use configuration files and Hydra software. Changing values in the config.yaml changes the hyperparameters in the model’s training process. To run an experiment with different parameters one would have to change it in the config.yaml file, save it and then run the train_model.py. To re-run a previous experiment it is possible to track hyperparameters used during the training process in the hydra’s logs folder.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure the same configuration throughout the experiments we are using config files. In the config file, we store hyperparameters of the model such as the number of epochs that the model is trained on, batch size and learning rate. For this part, we are using Hydra software. Moreover, Hydra is connected to the W&B project so that the hyperparameters can be seen for each run and for each figure saved. Changing hyperparameter in the config.yaml file changes the hyperparameter used during the training process. To reproduce an experiment it is possible to check saved hyperparameters in the W&B runn or in thy hydra’s log folder.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

During the development of our project, we created a couple of docker images. We created trainer.dockerfile that creates the image after the training process, then saves the model’s weights and exports them to the google cloud container. Prediction.dockerfile loads the model’s weights from the google container and it is used to make the prediction based on the created image. Inference.dockerfile this image is based on the PyTorch server, the difference between it and the prediction image is that we are able to send 1 image to the inference docker image.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

The debugging method varied depending on the group member. The most common debugging solution was to use a built-in debugger from VisualStudio IDE and to put breakpoints in the code structure. This allowed for the variable inspection in the middle of the run. To inspect the run time for different functions we did profiling. However, at some point in the project development, we switched to using the PythonLightning. Using the PythonLightning provided minimalization of the boilerplate and made the process of logging and exporting important values much easier. At the end of code development, we run profiling and sorted the output can be found in the gtsrb_out.prof file. It turns out that the most cumulative time has been used to call torch functions, 110.8s torch.conv2d, 110.4 torch.run_backward.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[GCP bucket](figures/gcp_bucket.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[GCP container registry](figures/gcp_container_registry.jpg)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[GCP cloud build history](figures/gcp_build_history.jpg)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy the model in the GCP cloud. However, first, we trained the model locally and afterwards saved it and loaded it for the predictions, that were run locally. Once this was completed we moved to set up the model in the cloud. The workflow of the model deployment in the cloud can be described as follows: whenever a team member successfully pushes and merges code with the main branch in the GitHub repository a docker file is automatically created in the GCP cloud. Then with the latest docker image, we are training the model in the cloud using Vertex AI. Once the training has finished the .pck file is created and we deploy this model in the cloud and use it for future projections. 

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We managed to implement data driving check and it has been shown that all data from our dataset, at the current state, is not influenced by driving. It could be expected as we compared training data with the test data. We managed to set up a simple monitoring in the google cloud. We monitor the total space available for our project, this included the dataset and all necessary google bucket files.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

During the development of the project, we come across many bugs and problems. One of the biggest issues was using the chosen model (MobileNetV3) for the recognition of our dataset. Our dataset GTRSB is not included in the PyTorch framework so it is not possible to download it just using the data loader. Once we downloaded this dataset from a different source we had to create our own script that loaded and created training files in the correct shape and format. Because of that, we had to adjust the model’s structure, training and prediction process. It took us a long time to identify mistakes we had made in the data preparation process and later a mistake with updating the weights of the pre-trained model. We faced a lot of issues during the deployment into the cloud phase of the project. We were verily quickly able to generate and deploy the model locally using PyTorch sever. However, we were struggling to send requests to it once it has been deployed. We had the same issue when using Google Cloud functions for model deployment into the cloud. While using Google cloud functions for model deployment we had issues with library dependencies which we managed to solve. However, we faced the issue that a function could not be deployed because of a memory shortage.  

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 

Group 23

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *s222913, sXXXXXX, sXXXXXX*
>
> Answer:
Q to verify: 23, 12, 15     
s222999, 

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In our project, we used the TIMM framework which is built on top of the PyTorch framework. It provides a set of pre-trained models and tools for fine-tuning those models on specific tasks. Using the TIMM framework we were able to download the MobileNetV3 (small) model architecture and also a pre-trained version of this model. Although the chosen model was trained with 1000 output classes using another functionality we were able to add an additional layer to the model’s architecture to map the output to our 43 classes from the GTRSB dataset. Using finetuning functionality we were able to update weights on all layers during training to adapt the MobileNetV3 model to recognize images from the GTRSB dataset.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

During the development of our project, we were using Anaconda to manage our virtual environments. A new member of the team would need to create an environment with the same libraries to be able to work with our project. To create a .txt file with the necessary requirements one needs to run “conda create --name <env> --file requirements.txt”. It is also possible to create a .txt file with pip using the “pip freeze > requirements.txt” command. A new member to join our project would need to be added to the Google Cloud, git repository, and wandb project using his email by one of the members.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

For the project, we used the cookiecutter data science project template. Where each of the folders had a specific purpose. The `data` folder is where the data is stored with different subfolders for the original data, the intermediated folder and the final data used for the model. Here we added a new folder to store the raw data from our dataset. The `doc` folder is where the documentation is stored. The `model` folder is where all the info related to the model such as checkpoints and predictions. `Notebooks` is used to store the Jupiters that were used during development, ` references` is used to store manuals, `reports` is where the Html and reports are generated, `src` is where all the source files are. These are divided into subfolders depending on the purpose of the files. We also added the tests folder where we store unit tests scripts


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We implemented pep8 guidelines in order to standardise the code style and flake and black libraries to check and adapt to these guidelines. It is important to use style guidelines, especially in big projects with different developers to accomplish consistency in the project making it easier to read and to understand different parts of the code.
## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

In our project, we have 2 tests to check if the model has been created correctly and 4 test to verify the correctness of the dataset. We implemented unittests to check if the loaded data has the correct size and later if all of the images in the used dataset have the correct shape. Our tests are also checking if every image has been loaded with a label. We implemented also 2 tests for the model, we are checking if the model’s output has the correct shape.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage is 65%. Despite covering more than 50% of the code it does not mean that the code is error-free. The coverage only determines the % of the lines of code that are being executed in the test. But this does not cover different inputs in functions or edge cases making the test an efficient way to test only some parts of the code. To improve the coverage for our project we should implement testing the training code and also the code responsible for making the predictions. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project we used both branches and pull requests for software version control. First we distributed tasks among group members, one person was responsible for creating the project structure and creating remote repository. Then rest of the members were invited to join the repository. All main features of the project were developed on separate branches. Once the feature has been developed person responsible was responsible for creating unit tests for this part. After the tests were implemented the code from branch has been pushed to the main. We only used pull requests on the branches that passed all the implemented tests and were accepted (if possible by another member). This process turned out to be beneficial in terms of maintaining the good quality of code and not introducing error to the main branch. Once the code from the develop branch has been merged with the main branch all members were notified that new version of exists and they should pull it to their local main branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC initially with the integration with google drive but we quickly moved to use DVC integrated with bucket storage in the cloud. During the project development, DVC enabled all members to get the original dataset using the terminal and DVC pull command without the struggle of extracting a huge file with 50000 images into the correct subdirectories inside the project. Later on, using DVC was very useful to ensure that all members were using data preprocessed in the same way. Using DVC also enables the storage of datasets that were preprocessed differently. With different datasets stored using DVC, it was easier to test the model on different data.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

TODO: add more
We created diferent workflows to have CI in our project. Initially we created a test workflow to check if the code that was being pushed could pass the unit testing to protect the main branch. Also the isort workflow was added to sort the imports alphabetially and ensure consistency in the project.


Pre-commit
generate docker img ??


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

In our project, we use configuration files and Hydra software. Changing values in the config.yaml changes the hyperparameters in the model’s training process. To run an experiment with different parameters one would have to change it in the config.yaml file, save it and then run the train_model.py. To re-run a previous experiment it is possible to track hyperparameters used during the training process in the hydra’s logs folder.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure the same configuration throughout the experiments we are using config files. In the config file, we store hyperparameters of the model such as the number of epochs that the model is trained on, batch size and learning rate. For this part, we are using Hydra software. Moreover, Hydra is connected to the W&B project so that the hyperparameters can be seen for each run and for each figure saved. Changing hyperparameter in the config.yaml file changes the hyperparameter used during the training process. To reproduce an experiment it is possible to check saved hyperparameters in the W&B runn or in thy hydra’s log folder.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

During the development of our project, we created a couple of docker images. We created trainer.dockerfile that creates the image after the training process, then saves the model’s weights and exports them to the google cloud container. Prediction.dockerfile loads the model’s weights from the google container and it is used to make the prediction based on the created image. Inference.dockerfile this image is based on the PyTorch server, the difference between it and the prediction image is that we are able to send 1 image to the inference docker image.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

The debugging method varied depending on the group member. The most common debugging solution was to use a built-in debugger from VisualStudio IDE and to put breakpoints in the code structure. This allowed for the variable inspection in the middle of the run. To inspect the run time for different functions we did profiling. However, at some point in the project development, we switched to using the PythonLightning. Using the PythonLightning provided minimalization of the boilerplate and made the process of logging and exporting important values much easier. At the end of code development, we run profiling and sorted the output can be found in the gtsrb_out.prof file. It turns out that the most cumulative time has been used to call torch functions, 110.8s torch.conv2d, 110.4 torch.run_backward.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[GCP bucket](figures/gcp_bucket.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[GCP container registry](figures/gcp_container_registry.jpg)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[GCP cloud build history](figures/gcp_build_history.jpg)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy the model in the GCP cloud. However, first, we trained the model locally and afterwards saved it and loaded it for the predictions, that were run locally. Once this was completed we moved to set up the model in the cloud. The workflow of the model deployment in the cloud can be described as follows: whenever a team member successfully pushes and merges code with the main branch in the GitHub repository a docker file is automatically created in the GCP cloud. Then with the latest docker image, we are training the model in the cloud using Vertex AI. Once the training has finished the .pck file is created and we deploy this model in the cloud and use it for future projections. 

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We managed to implement data driving check and it has been shown that all data from our dataset, at the current state, is not influenced by driving. It could be expected as we compared training data with the test data. We managed to set up a simple monitoring in the google cloud. We monitor the total space available for our project, this included the dataset and all necessary google bucket files.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

During the development of the project, we come across many bugs and problems. One of the biggest issues was using the chosen model (MobileNetV3) for the recognition of our dataset. Our dataset GTRSB is not included in the PyTorch framework so it is not possible to download it just using the data loader. Once we downloaded this dataset from a different source we had to create our own script that loaded and created training files in the correct shape and format. Because of that, we had to adjust the model’s structure, training and prediction process. It took us a long time to identify mistakes we had made in the data preparation process and later a mistake with updating the weights of the pre-trained model. We faced a lot of issues during the deployment into the cloud phase of the project. We were verily quickly able to generate and deploy the model locally using PyTorch sever. However, we were struggling to send requests to it once it has been deployed. We had the same issue when using Google Cloud functions for model deployment into the cloud. While using Google cloud functions for model deployment we had issues with library dependencies which we managed to solve. However, we faced the issue that a function could not be deployed because of a memory shortage.  

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Member s222999 contributed to the preprocessing of data so that they can be used to train the model, writing and validating the model_train.py and predict_model.py scripts, writing unittest, and profiling of the code.

