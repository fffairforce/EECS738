# EECS738
MachineLearning_course_projects

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
### The goal of this project is to make use of multiple distribution models and model the given data, over the project I worked with _normal distribution, binomial distrubution, GMM(gaussian mixture model)_ to address the problem, seaborn python library was used only for visually verify my distribution result
_**libraies**_ used in the project: pandas, numpy, matplotlib, scipy.stats 


**Running normal distribution** 

go to norm.py and run lines, no input is needed.

**results**

iterations were made in the four attributes and distribution curves are shown in terms of three different types of flower

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

![image](https://user-images.githubusercontent.com/42806161/109467681-2c4a8700-7a31-11eb-9bf7-79b33cb49fd8.png)

![image](https://user-images.githubusercontent.com/42806161/109467713-37051c00-7a31-11eb-8192-5642405aecc1.png)

![image](https://user-images.githubusercontent.com/42806161/109467746-45533800-7a31-11eb-863f-85e3ea387e19.png)

![image](https://user-images.githubusercontent.com/42806161/109467772-4e440980-7a31-11eb-8c93-8e00aec81acd.png)

![image](https://user-images.githubusercontent.com/42806161/109467796-569c4480-7a31-11eb-8846-c475a6499dda.png)

![image](https://user-images.githubusercontent.com/42806161/109467881-73387c80-7a31-11eb-9dd4-8f870aee7025.png)

![image](https://user-images.githubusercontent.com/42806161/109467891-792e5d80-7a31-11eb-8707-6a42e4c40cdf.png)

![image](https://user-images.githubusercontent.com/42806161/109467911-7fbcd500-7a31-11eb-8e81-bb1ba4d5ad4b.png)

![image](https://user-images.githubusercontent.com/42806161/109467930-88151000-7a31-11eb-9bfe-0780be1d3a9e.png)

![image](https://user-images.githubusercontent.com/42806161/109467944-8e0af100-7a31-11eb-89b6-6ca3724dd899.png)

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

turns out the distributions are distinguishable in different labels, but clearly, some labels won't work and can;t provide a good distribution
