# ie684-lab-09-solved
**TO GET THIS SOLUTION VISIT:** [IE684 Lab 09 Solved](https://www.ankitcodinghub.com/product/ie684-lab-09-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;97623&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;IE684 Lab 09 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Binary Classification Problem

In the last lab, we developed an optimization method to solve the optimization problem associated with binary classification problem. In this lab, we will introduce constraints to the optimization problem and try to extend the scheme we developed in the last lab.

Recall that for a data set D = {(xi, yi)}ni=1 where xi ∈ X ⊆ Rd, yi ∈ {+1, −1}, we solve:

</div>
</div>
<div class="layoutArea">
<div class="column">
form:

</div>
</div>
<div class="layoutArea">
<div class="column">
λ 1􏰂n min f(w) = ∥w∥2 +

</div>
</div>
<div class="layoutArea">
<div class="column">
L(yi, w⊤xi). (1)

where we considered the following loss functions:

􏰁 Lh(yi,w⊤xi)=max{0,1−yiw⊤xi} (hinge)

􏰁 Ll(yi, w⊤xi) = log(1 + exp(−yiw⊤xi)) (logistic)

􏰁 Lsh(yi, w⊤xi) = (max{0, 1 − yiw⊤xi})2. (squared hinge)

Solving the optimization problem (1) facilitates in learning a classification rule h : X → {+1, −1}. We used the following prediction rule for a test sample xˆ:

h(xˆ) = sign(w⊤xˆ). (2) In the last lab, we used a decomposition of f(w) and solved an equivalent problem of the following

</div>
</div>
<div class="layoutArea">
<div class="column">
w∈Rd 2n

</div>
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
n

min f(w) = min 􏰂 fi(w). (3)

ww

i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
In this lab, we will consider a constrained variant of the optimization problem (1). For a data set D = {(xi, yi)}ni=1 where xi ∈ X ⊆ Rd, yi ∈ {+1, −1}, we solve:

L(yi, w⊤xi), s.t. w ∈ C (4)

where C ⊂ Rd is a closed convex set.

Hence we would develop optimization method to solve the following equivalent constrained problem of (4):

n

min f(w) = min 􏰂 fi(w). (5)

w∈C w∈C

i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
λ 1􏰂n min f(w) = ∥w∥2 +

</div>
</div>
<div class="layoutArea">
<div class="column">
w2n

i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 09 16-March-2022

Exercise 0: Data Preparation

<ol>
<li>Use the following code snippet. Load the wine dataset from scikit-learn package using the following code. We will load the features into the matrix A such that the i-th row of A will contain the features of i-th sample. The label vector will be loaded into y.
(a) [R] Check the number of classes C and the class label values in wine data. Check if the class labels are from the set {0,1,…,C −1} or if they are from the set {1,2,…,C}.

(b) When loading the labels into y do the following:

􏰁 If the class labels are from the set {0,1,…,C −1} convert classes 0,2,3,…,C −1 to −1.

􏰁 If the class labels are from the set {1,2,…,C} convert classes 2,3,…,C to −1. Thus, you will have class labels eventually belonging to the set {+1, −1}.
</li>
<li>Normalize the columns of A matrix such that each column has entries in range [−1, 1].</li>
<li>Note that a shuffled index array indexarr is used in the code. Use this index array to partition the data and labels into train and test splits. In particular, use the first 80% of the indices to create the training data and labels. Use the remaining 20% to create the test data and labels. Store them in the variables train data, train label, test data, test label.
import numpy as np

#we will load the wine data from scikit-learn package

from sklearn.datasets import load_wine

wine = load_wine()

#check the shape of wine data

print(wine.data.shape)

A = wine.data

#Normalize columns of A so that all entries are in the range [-1,+1] #for i in range(A.shape[1]):

# A[i] = ???

#check the shape of wine target

print(wine.target.shape)

#How many labels does wine data have?

#C=num_of_classes

#print(C)

n = wine.data.shape[0] #Number of data points

d = wine.data.shape[1] #Dimension of data points

<pre>     #In the following code, we create a nx1 vector of target labels
</pre>
<pre>     y = 1.0*np.ones([A.shape[0],])
     for i in range(wine.target.shape[0]):
</pre>
<pre>        # y[i] = ???? # Convert class labels that are not 1 into -1
</pre>
<pre>     #Create an index array
     indexarr = np.arange(n) #index array
     np.random.shuffle(indexarr) #shuffle the indices
     #print(indexarr) #check indexarr after shuffling
</pre>
<pre>     #Use the first 80% of indexarr to create the train data and the remaining 20% to
          create the test data
</pre>
<pre>     #train_data = ????
     #train_label = ????
     #test_data = ????
     #test_label = ????
</pre>
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 09 16-March-2022

4. Use the python function developed in last lab where you had implemented the prediction rule in eqn. (2).

5. Use the python function developed in the previous lab which takes as input the model pa- rameter w, data features and labels and returns the accuracy on the data.

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 09 16-March-2022

Exercise 2: An Optimization Algorithm

<ol>
<li>To solve the problem (5), we shall use the following method (denoted by ALG-LAB9).
Assume that the training data contains ntrain samples.

(a) For t = 1,2,3,…, do:

i. Sample i uniformly at random from {1, 2, . . . , ntrain}.

ii. wt+1 = ProjC(wt − ηt∇fi(wt)).

The notation ProjC (z) = arg minu∈C ∥u − z∥2 denotes the orthogonal projection of point z onto set C. In other words, we wish to find a point u⋆ ∈ C which is closest to z in terms of l2 distance. For specific examples of set C, the orthogonal projection has a nice closed form.
</li>
<li>When C = {w ∈ Rd : ∥w∥∞ ≤ 1}, find an expression for ProjC(z). (Recall: For a w = 􏰆w1 w2 …wd􏰇⊤ ∈ Rd, we have ∥w∥∞ = max{|w1|,|w2|,…,|wd|}.)</li>
<li>Consider the hinge loss function Lh. Use the python modules developed in the last lab to compute the loss function Lh, and objective function value. Also use the modules developed in the last lab to compute the gradient (or sub-gradient) of fi(w) for the loss function Lh. Denote the (sub-)gradient by gi(w) = ∇wfi(w).</li>
<li>Define a module to compute the orthogonal projection onto set C. def compute_orthogonal_projection(z):
<pre>           #complete the code
</pre>
</li>
<li>Modify the code developed in the previous lab to implement ALG-LAB8. Use the following template.
def OPT1(data,label,lambda, num_epochs): t=1

<pre>       #initialize w
       #w = ???
       arr = np.arange(data.shape[0])
       for epoch in range(num_epochs):
</pre>
<pre>         np.random.shuffle(arr) #shuffle every epoch
         for i in np.nditer(arr): #Pass through the data points
</pre>
<pre>           # step = ???
           # Update w using w &lt;- Proj_C(w - step * g_i (w))
           t = t+1
           if t&gt;1e4:
</pre>
t=1 return w
</li>
<li>In OPT1, use num epochs=500, step=1t. For each λ ∈ {10−3,10−2,0.1,1,10}, perform the following tasks:
<ol>
<li>(a) &nbsp;[R] Plot the objective function value in every epoch. Use different colors for different λ values.</li>
<li>(b) &nbsp;[R] Plot the test set accuracy in every epoch. Use different colors for different λ values.</li>
<li>(c) &nbsp;[R] Plot the train set accuracy in every epoch. Use different colors for different λ values.</li>
<li>(d) &nbsp;[R] Tabulate the final test set accuracy and train set accuracy for each λ value.</li>
</ol>
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 09 16-March-2022

(e) [R] Explain your observations.

7. [R] Repeat the experiments (with num epochs=500) for different loss functions Ll and Lsh. Explain your observations.

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 09 16-March-2022

Exercise 3: A different constraint set

<ol>
<li>When C = {w ∈ Rd : ∥w∥1 ≤ 1}, find an expression for ProjC(z). (Recall: For a w =
􏰆w1 w2 …wd􏰇⊤ ∈ Rd, we have ∥w∥1 = 􏰈di=1 |wi|.)
</li>
<li>Consider the hinge loss function Lh. Use the python modules developed in the last lab to compute the loss function Lh, and objective function value. Also use the modules developed in the last lab to compute the gradient (or sub-gradient) of fi(w) for the loss function Lh. Denote the (sub-)gradient by gi(w) = ∇wfi(w).</li>
<li>Define a module to compute the orthogonal projection onto set C. def compute_orthogonal_projection_ex3(z):
<pre>             #complete the code
</pre>
</li>
<li>In OPT1, use num epochs=500, step=1t. For each λ ∈ {10−3,10−2,0.1,1,10}, perform the following tasks:
<ol>
<li>(a) &nbsp;[R] Plot the objective function value in every epoch. Use different colors for different λ values.</li>
<li>(b) &nbsp;[R] Plot the test set accuracy in every epoch. Use different colors for different λ values.</li>
<li>(c) &nbsp;[R] Plot the train set accuracy in every epoch. Use different colors for different λ values.</li>
<li>(d) &nbsp;[R] Tabulate the final test set accuracy and train set accuracy for each λ value.</li>
<li>(e) &nbsp;[R] Explain your observations.</li>
</ol>
</li>
<li>[R] Repeat the experiments (with num epochs=500) for different loss functions Ll and Lsh. Explain your observations.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
7

</div>
</div>
</div>
