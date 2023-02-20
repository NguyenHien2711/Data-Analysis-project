# Case study description <br>
**Description:** Geely Auto is a leading automobile manufacturer based in Hangzhou that plans to expand business to the American market through establishing manufacturing factories with ambition to win against both domestic and European competitors in the US market. <p>
**Objectives:**  <p>
  - Analyze meaningful features having huge impact on car price
  - The correlation between these features and car price <p>
    
**Dataset info:** <p>
- car_ID : Car ID 
- symboling: Security ratings (ranging from -2 to 3, note -2: maximum, 3: minimum)
- carName: Name of car 
- fueltype: type of fueltype 
- aspiration: aspiration types (turbo, std)
- doornumber: number of door 
- carbody: car body types (sedan, wagon, hatchback)
- drivewheel: type of drivewheel 
- enginelocation: location of engine 
- wheelbase: length of wheelbase
- carlength: length of car 
- carwidth: width of car 
- carheight: height of car 
- curbweight: Weight of car without people or luggage 
- enginetype: type of engine 
- cylindernumber: number of cylinder 
- enginesize: size of engine 
- fuelsystem: fuel system of car 
- boreratio: car bore-ratio 
- stroke: number of stroke
- compressionratio: compression ratio
- horsepower: Horsepower
- peakrpm: maximum engine speed
- citympg: miles per gallon (~4.5 l) of fuel in the city
- highwaympg: miles per gallon (~4.5 l) of fuel in the highway 
- price: Car price 
**Code**: [Analyse the features influence Car Price](https://github.com/NguyenHien2711/Data-Analysis-project/tree/main/CarPrice) <pr>

# A. Approach 
**Car project:** This project aims to build a linear regression model to identify and evaluate the primary features which have substantial impact on Car Price through Python Pandas, Numpy, Seaborn, Matplotlib, Scipy, Sklearn. <br>
  
The detail steps to approach the cases is:<br>
  
**1. Overview**  <br>
  - Acquire some basic knowledge about data, how the various features are distributed and whether there are missing values, untreated format data. 
  - Subdivide variables into 2 types concluding categorical and numerical features, this process help to check distribution (apply describe() function) <br>
  
**2. Exploratory Data Analysis** <br>
  - Univariate Descriptive Statistics:
    + Analysis each feature in more detail 
    + Graphically approach:
      + Hisplot and kdeplot to identify how data is distributed (normal distribution, skew distribution) 
      + Boxplot: to check outliers of numerical variables
      + Countplot: visualize frequency of values 
  - Bivariate analysis: 
    + Numerical: lmplot, heatmap and pearson to check relationship between independent features and dependent features 
    + Categorical: Violinplot to visualize how categorical variables affect price <br>
  
**3. Data Preprocessing** <br>
  - Feature Selection: Filter selection through check correlation (criteria >0.7)
  - Split Dataset in to Data train set and data test set
  - Feature Scaling: Check which type of feature scaling technique is optimal to apply for dataset
  - Apply feature scaling for training set and test set <br>
  
**4. Build model and evaluate model** <br>
  - Build OLS model
  - Calculate R squared, mean_squared_error and accuracy_score 

# B. Summary result 
### 1.Market overview
![price](https://user-images.githubusercontent.com/109226305/219996196-ee1d83ac-36ea-4c6d-ac76-a2651c658e51.jpg)  <img src="https://user-images.githubusercontent.com/109226305/219996359-99052d71-ce05-47fa-ae23-3e6a316c0e16.jpg" width=30% height=30%> <br>
Suppose: According to price classification, we could necessarily subdivide the competitors into 3 groups including luxurious car brands having price more than $20000, ordinary car brands having price less than $20000 and over $12000, and the cars having price under $12000 were grouped into lower car brands. <br>
The histogram graph shows the frequency of car prices and the pie chart describes the proportion of car types in 3 kinds of car brands classified by price in the US market. <br>
Looking from an overall perspective, in the US automobile market, the car price is apparently diverse  and present in many segments. <br>
The average price of a car is about $13277, ranging from over $5000 to around $45400. <br>
Americans seem to have high-demand for lower car prices,  accounting for the majority percentage of the total car, at around 58.5%, the figure for medium price and luxury price were considerably lower, at approximately 29,8% and 12.7%. <br>
### 2. Competitors 
<img src="https://user-images.githubusercontent.com/109226305/219997189-9ebbbcb6-98af-43d6-894c-7b9a53569705.jpg" width=90% height=90%> <br>
As we can see, Toyota is the dominant company in producing the highest number of different types of cars, at approximately 15.6% and this is followed by Nissan and Mazda at about 8.8%, 8.3% respectively. However, the car price of three companies is at about $9685.81, $10415.7, $10652.9 respectively, all below the average car price, at $13277. <br>
### 3. The features have significant influence in car prices
**3.1. Numerical features**
Based on Pearsonr test, We can make the following observations:<br>
  - Carheight, stroke, compression ratio, peak rpm don’t have correlation with price, which could be said that the change in these attributes would not have a statistically meaningful impact on car price. <br>
  - Length of wheelbase, length of car, width of car, weight of car without people or luggage, size of engine, car bore-ratio, Horsepower have a positive correlation with car price. <br>
  - In the meanwhile, miles per gallon (~4.5 l) of fuel in the city and miles per gallon (~4.5 l) of fuel on the highway have a negative correlation with car price.<br> 
  
**3.2. Categorical features** <br>
  - Symboling:The security ratings (ranging from -2 to 3, note -2: maximum, 3: minimum) seem to not significantly affect car price, the lowest level of security have average price is higher than level 1 and 2 <br>
  - Carbody: Most of the body of cars in the US market is sedan, hatchback, wagon, at 46.8%, 34.2% and 12.2% respectively. Whereas, convertible and hardtop are relatively less common and their price is nearly twice as expensive as the rest. <br>
=> The change in what kind of car body could affect to price, the price of seden, hatchback, wagon is nearly twice as cheap as convertible and hardtop<br>
  - Enginetype: ohc was used the most, at 72.2% and have the cheapest price in the list<br>
  - Cylindernumber: the more cylinders were used, the more expensive the price was. The car commonly has four cylinders, accounting at nearly 77.6% of the total.<br>
  - Fueltype: gas was used the most, at roughly 90.2% and the price is cheaper than diesel <br>
  - Aspiration: std is the most popular, at 82%, and price is cheaper than turbo <br>
  - Drivewheel: there are 3 type of drive wheel, fwd, rwd and 4wd and they account at, rwd have the highest price and account at 37% the total wheel produced, fwd is the lowest price and was used the most, at 58.5% <br>
  - Enginelocation: Nearly 97% engine was located in front <p>
### 4. Machine Learning Model<br>
By using the Sklearn library, I have built a basic model in order to predict car price based on the top most influential features. OLS Regression could be shown as follow:
![Capture](https://user-images.githubusercontent.com/109226305/219998717-de325d14-9bbb-4fc2-97f6-3905ce880cd3.PNG) <br>
Some conclusion we can made for the OLS Regression model:
  - When carwidth is reduced by 1, the car price is reduced by 0.1665 times,on the condition that all other features do not change

---------------------------------------------------------------------------------------------------
Thank you for your time and consideration. Hope this repo would help you to assess my Python skills!<br>
If you have any questions or comments, feel free contact to me through contact information below, I appreciate all of your constructive, precious feedback to keep striving for the best.<p>
### Contact
***
Feel free to contact me via: <br>
- Gmail: nguyenhien12t1@gmail.com <br>
- Linkedin: [Hien Nguyen](https://www.linkedin.com/in/hien-nguyen-a7b9a4201/) <br>
- Facebook: [Nguyễn Hiền](https://www.facebook.com/hien.nguyenthithuy.562) <br>
- Mobile phone: (+84) 0337557244 <br>
