
# Breast cancer classification

Breast cancer is the most common cancer among women worldwide, accounting for 25 percent of all cancer.
cases and affected two point one million people in 2015, early diagnosis significantly increases the.
chances of survival.

The key challenge in cancer detection is how to classify tumors into malignant or benign machine learning techniques can dramatically improves the accuracy of diagnosis.

Research indicates that most experienced physicians can diagnose cancers with 79 percent accuracy, while 91 percent correct diagnosis is achieved using machine learning techniques in this case study.

Our task is to classify tumors into malignant or benign tumors using features of patients from several detect the cancer is benign.

And then we extract out of this basically 30 features, which indicating radius, texture, perimeter
And in this case study, we have 212 malignant cases and 357 benign cases.

So it's kind of the output is a binary in a form indicating zero or one for malignant or benign. and I thought SVM was present for support vector machine.

machine learning techniques was able to classify tumors effectively into - Malignant - Benign ,at the images and tell us, this cancer is malignant or malignant and this cancer is benign, right where the tumor is benign.

So as part of the conclusion, part of the benefits of that case study is that our early breast cancer detection can dramatically save lives, especially in the developing world.

the technique can be further improved by combining computer vision and machine learning and deep learning techniques to classify cancer directly using tissue images.

We actually reached on precision of around 97 percent, training an SVM model to make accurate breast cancer classifications, improving the performance of an SVM model, and testing model accuracy using Confusion Matrix.

And let's build a cancer free world.

## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

Predicting if the cancer diagnosis is benign or malignant based on several observations/features

30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569

Class Distribution: 212 Malignant, 357 Benign

Target class:

   - Malignant
   - Benign



Project Task

In this study, my task is to classify tumors into malignant (cancerous) or benign (non-cancerous) using features obtained from several cell images.
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

It’s important to start with the intuition for SVM with the special linearly separable classification case.
If classification of observations is “linearly separable”, SVM fits the “decision boundary” that is defined by the largest margin between the closest points for each class. This is commonly called the “maximum margin hyperplane (MMH)”.


Model Training

From our dataset, let’s create the target and predictor matrix
“y” = Is the feature we are trying to predict (Output). In this case we are trying to predict if our “target” is cancerous (Malignant) or not (Benign). i.e. we are going to use the “target” feature here.
“X” = The predictors which are the remaining columns (mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.)


Create the training and testing data

Now that we’ve assigned values to our “X” and “y”, the next step is to import the python library that will help us split our dataset into training and testing data.
Training data = the subset of our data used to train our model.
Testing data = the subset of our data that the model hasn’t seen before (We will be using this dataset to test the performance of our model).

Improving our Model


The first process we will try is by normalizing our data
Data normalization is a feature scaling process that brings all values into range [0,1]
X’ = (X-X_min) / (X_max — X_min)


## Appendix


Breast Cancer Wisconsin (Diagnostic) Data Set

Abstract: Diagnostic Wisconsin Breast Cancer Database


Data Set Characteristics:  

Multivariate

Number of Instances:

569


Area:

Life


Attribute Characteristics:

Real


Number of Attributes:

32

Date Donated

1995-11-01


Associated Tasks:

Classification

Missing Values?

No


Number of Web Hits:

1518690


Source:

Creators:

1. Dr. William H. Wolberg, General Surgery Dept.
University of Wisconsin, Clinical Sciences Center
Madison, WI 53792
wolberg '@' eagle.surgery.wisc.edu

2. W. Nick Street, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
street '@' cs.wisc.edu 608-262-6619

3. Olvi L. Mangasarian, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
olvi '@' cs.wisc.edu

Donor:

Nick Street


Data Set Information:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at 

Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/


Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)


Relevant Papers:

First Usage:

W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.

O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.

Medical literature:

W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 163-171.

W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Image analysis and machine learning applied to breast cancer diagnosis and prognosis. Analytical and Quantitative Cytology and Histology, Vol. 17 No. 2, pages 77-87, April 1995.

W.H. Wolberg, W.N. Street, D.M. Heisey, and O.L. Mangasarian. Computerized breast cancer diagnosis and prognosis from fine needle aspirates. Archives of Surgery 1995;130:511-516.

W.H. Wolberg, W.N. Street, D.M. Heisey, and O.L. Mangasarian. Computer-derived nuclear features distinguish malignant from benign breast cytology. Human Pathology, 26:792--796, 1995.

  
## Authors

- [@nitinkumar388](https://github.com/nitinkumar388)

  
## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

  
## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

  
