# EECS738
### MachineLearning_course_project1

### Willy Lee, BioE

## data 1 structure
the attributes are 
1) sepal.length
2) sepal.width
3) petal.length
4) petal.width

## data 2 structure
The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it )
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10) Color intensity
11) Hue
12) OD280/OD315 of diluted wines
13) Proline

## project overview
### The goal of this project is to make use of multiple distribution models and model the given data, over the project I worked with _normal distribution, binomial distrubution, GMM(gaussian mixture model)_ to address the problem, seaborn python library was used only for visually verify my distribution result. In terms of the results for two dataset, I only provided extract analysis here for the second data in GMM model since it's a bit different in either data extraction and analyzation. Other than that the basic distributions are mostly identical in the code so i saved the redundent result for those.
## Codes
Most codes should be concated to .ipynb file for easy overview, if not, I named the source .py file for different distributions
_**libraies**_ used in the project: pandas, numpy, matplotlib, scipy.stats 


**Running normal distribution** 

go to norm.py and run lines, no input is needed.

**results**

This is just a generalized data sorting for myself to familiarize with the data and potentially make choices of which distributions to go with. Iterations were made in the four attributes and distribution curves are shown in terms of three different types of flower

![image](https://user-images.githubusercontent.com/42806161/109463884-5c8f2700-7a2b-11eb-85ef-568e73bd18f9.png)

![image](https://user-images.githubusercontent.com/42806161/109463900-64e76200-7a2b-11eb-97c6-65b9fcc72e37.png)

![image](https://user-images.githubusercontent.com/42806161/109463908-6ca70680-7a2b-11eb-91b7-bf02fdb4b623.png)

![image](https://user-images.githubusercontent.com/42806161/109463920-729ce780-7a2b-11eb-8c2a-af5184ef4f1f.png)


**Running binomial distribution**

given the feature of binomial function, the probability is divided so as the P{class of interest}=1/3, and P{the rest classes}=2/3
run binorm.py

**results**

Probability density is as shown in the figure where blue curve stands for the 'interested' class with mu=1/3. There's clearly difference in the two 'classes', but not properly indicating the 'distance' bwteen classes.  
![image](https://user-images.githubusercontent.com/42806161/109464932-189d2180-7a2d-11eb-896b-026bafbb7ad7.png)

**Running GMM**

Gaussian mixture model and EM algorithm is used to iterate over the ramdomized initial distributions. Modify `filename` under main function of GMM.py to `'iris.data'` to run GMM on iris data, and `'wine.data'` if analyzing wine data (also uncomment the two line below `# if wine` to replace the read iris dat line). Note that in this Iris data we had 3 classes, so the 2nd input in GMM class should be the class number 3, the 4th input in GMM function is the iterations you wish to run, it can be adjusted until you have the desired result to fit the data, here i set 10 as default. For the wine data, remember to adjust the GMM input as desired, class=13.

** Data1 results**

As shown in the figures titles, GMM are iterated through data column by column, the distributions are much more fit to the color-code class data scatter below after a few iterations 

![image](https://user-images.githubusercontent.com/42806161/109467632-189f2080-7a31-11eb-8df6-4ada37a337a2.png)
![image](https://user-images.githubusercontent.com/42806161/109467657-218ff200-7a31-11eb-8abc-da27e34d4743.png)

![image](https://user-images.githubusercontent.com/42806161/109467964-95ca9580-7a31-11eb-8cb7-94ef53cbe888.png)
![image](https://user-images.githubusercontent.com/42806161/109467988-9bc07680-7a31-11eb-9951-377945f48852.png)

**algorithm verification**

Comparing to the 2D distribution seaborn plot, our GMM model worked out best to fit the data which is as expected since we had also the EM algorithm run for higher accuracy.

![image](https://user-images.githubusercontent.com/42806161/109468768-b2b39880-7a32-11eb-940c-4d0abe0537ef.png)

**Data2 results**

In order to save space, only partial of columns' initial and final distribution plot is shown here.

![image](https://user-images.githubusercontent.com/42806161/109472656-04aaed00-7a38-11eb-8f56-0fe94575e622.png)
![image](https://user-images.githubusercontent.com/42806161/109476682-ccf27400-7a3c-11eb-93ce-3a602d3c33e9.png)
![image](https://user-images.githubusercontent.com/42806161/109476715-d380eb80-7a3c-11eb-8ae4-f5e7b3ab774c.png)
![image](https://user-images.githubusercontent.com/42806161/109476743-daa7f980-7a3c-11eb-92b2-396b54a0c346.png)
![image](https://user-images.githubusercontent.com/42806161/109476764-e1cf0780-7a3c-11eb-83c3-a5a5351ab300.png)
![image](https://user-images.githubusercontent.com/42806161/109476780-e7c4e880-7a3c-11eb-98bc-7d575dfe91b9.png)
![image](https://user-images.githubusercontent.com/42806161/109476826-f4494100-7a3c-11eb-872d-484688ebfa6a.png)
![image](https://user-images.githubusercontent.com/42806161/109476864-fd3a1280-7a3c-11eb-8cf6-710642e5f4ba.png)
![image](https://user-images.githubusercontent.com/42806161/109476903-05924d80-7a3d-11eb-956d-4b7f04f4c792.png)
![image](https://user-images.githubusercontent.com/42806161/109476922-0aef9800-7a3d-11eb-8b6b-a3253f306f63.png)

turns out the distributions are distinguishable in different labels, but clearly, some labels won't work and can't provide a good distribution, that's where we found it not helpful to simply apply gmm on every factor, so correlations between factors were aligned as shown below in a heatmap.

![image](https://user-images.githubusercontent.com/42806161/109554299-a7478800-7a99-11eb-953b-bb03675774fb.png)

From the heatmap above, we can conclude that the correlation between

"Flavanoids" and "Total phenols" (0.86);
"OD280" and "Flavanoids" (0.79);
"OD280" and "Total phenols" (0.7);
are large.

the weights and means of GMM is as shown:

![image](https://user-images.githubusercontent.com/42806161/109565777-93575280-7aa8-11eb-848c-6fcd44e42fcf.png)

## Conclusion and feedback

Overall, the GMM worked well in giving a good distribution over the dataset and aligned well with the reference pairplot. but the EM probably need improvement since the gaussian scales in my plot didn't seem to be changing, i'm pretty sure i would need the fixed in the "E-step" but playing around with the likelihood formula crashed the code... I guess i need more help on implementing the distribution from scratch in terms of how to fit in the data and how to address the formula properly without meesing up the matrices.
