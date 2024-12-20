# Achyuth Kolluru

## Contact Information
- **Email:** achyuthkolluru2@gmail.com
- **GitHub:** [Github](https://github.com/AchyuthKolluru)
- **LinkedIn:** [Linkedin](https://www.linkedin.com/in/achyuth-kolluru)

## About
A current Master's student in the Masters of Information and Data Science Program at UC Berkeley, with a strong academic background from the University of California, Merced, holding a major in Computer Science and a minor in Applied Mathematics. Fascination with vision-based systems in computer science led to internships at Lawrence Livermore National Laboratory and Morning Star, working on advanced projects involving ECG data analysis and computer vision. A deep interest in machine learning, particularly in its applications in biology and cognitive science, continues to drive academic pursuits, alongside a dedication to advocating for individual privacy in AI advancements, ensuring that technological innovation aligns with ethical considerations.

## Education
- **University of California, Berkeley** (January 2024 - May 2025)  
  *Master of Information and Data Science*

- **University of California, Merced** (August 2019 - May 2023)  
  *Bachelor of Science, Major in Computer Science, Minor in Applied Mathematics*  
  - **Top Finisher** for Innovate to Grow Event for 2022 Fall Software Engineering Capstone  
  - **3rd Place Winner** for the Water Hack Challenge 2023 Issued by Secure Water Future

**Selected Coursework:**  
Applied Machine Learning | Computer Vision | Modern Applied Statistics | Fundamentals of Data Engineering | Statistical Methods of Time Series Data | Machine Learning at Scale

<br />

## Programming Skills
- **Languages:** Python, C++, MATLAB, R, C, SQL
- **Technologies:** TensorFlow, Pytorch, Scikit Learn, Keras, Git, ROS, Linux, Jupyter, Pandas, PostgreSQL, Redis

## Experience

### Data Science Intern | Lawrence Livermore National Laboratory | June 2023 - July 2023
Conducted ECG data analysis, focusing on heartbeat classification, activation map reconstruction, and trans-membrane potential interpretation. Developed a convolutional neural network, based on AlexNet, that achieved 94% test accuracy in reconstructing 75 trans-membrane voltage signals over 500 milliseconds, enhancing the accuracy of predictive models.

**Project Presentation:**  
![LLNL Data Science Project](assets/images/DSC_poster_template.jpg)

### Research Assistant | University of California Merced | November 2022 - May 2023
Developed a cubic interpolation method for video frame rate enhancement by incorporating the derivative of acceleration (Jerk) with deep learning. Applied optical flow estimation techniques, including PWC-Net, to improve the Peak Signal-to-Noise Ratio (PSNR) of interpolated images by 20% using PyTorch and CUDA.

### Software Engineer Intern | Morning Star | August 2022 - December 2022
Led a computer vision project using Yolov4 Tiny and DeepSort for real-time detection, tracking, and counting of tomatoes during harvesting. Managed data extraction and labeling for a 3,000-image dataset, which facilitated efficient model training. Developed a notification system to alert operators when tomato counts exceeded thresholds. The project achieved 78% accuracy and was recognized as the Top Finisher at the Innovate to Grow Event.

**Project Poster:**  
![Morning Star Software Internship](assets/images/2022-08-Fall-CSE-Team315-poster.png)

**Project Demonstration:**  
<div style="text-align: center;">
  <img src="assets/gifs/results.gif" alt="Morning Star Project GIF" width="800px"/>
</div>

<br />
<br />

## Relevant Projects

### Flight Delay Prediction and Analysis
**Description:**  
Developed a machine learning-based system for predicting flight delays using a large-scale historical flight dataset (approximately 30GB). The project involved extensive **data preprocessing**, including handling missing values, encoding categorical variables, and feature engineering to enhance model performance. Various machine learning models, including **Random Forest**, **XGBoost**, and **Linear Regression**, were applied to predict flight delays. 

Model performance was evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1 score** to assess the effectiveness of each algorithm. To efficiently process the large dataset, **PySpark ML** was leveraged for **parallelized data processing** and **model training**. This approach utilized only **CPU parallelization**, ensuring scalable and high-performance execution even with the large 30GB dataset. The system demonstrated the ability to scale efficiently with large data and deliver real-time predictive capabilities for flight delay analysis.

The project showcases expertise in **predictive modeling**, **feature engineering**, and **distributed computing** with a focus on time-sensitive predictions, making it well-suited for real-world flight delay forecasting and optimization.

**GitHub Repository:**  
[https://github.com/AchyuthKolluru/Flight-Delay-Prediction-and-Analysis]


### Delivery Simulation
A package delivery simulation system was created, integrating SQL, MongoDB, and Redis to provide dynamic, real-time feedback throughout the delivery process. Neo4J was employed for advanced route optimization and live tracking of delivery trucks. This solution enhanced logistics efficiency by providing real-time updates and optimized routing, leading to quicker deliveries and more effective resource allocation.

*Project Demonstration:*  
<div style="text-align: center;">
  <img src="assets/gifs/Delivery.gif" alt="Delivery Simulation" width="800px"/>
</div>

<br />

### Analyzing Airbnb Listings in NYC (2019)
Conducted a thorough analysis of NYC Airbnb 2019 data, focusing on feature engineering to extract meaningful insights and enhance model performance. Multiple predictive models, including linear regression, decision trees, and ensemble methods, were developed and rigorously compared for pricing accuracy. By evaluating key metrics such as RMSE and MAE, the best-performing model was selected to optimize rental price predictions.

### Model Error Analysis
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <h4>Error Metrics</h4>
    <img src="assets/images/Errors.png" alt="Errors" width="600px"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <h4>Model Performance Comparison</h4>
    <img src="assets/images/Comparison_models.png" alt="Model Comparisons" width="600px"/>
  </div>
</div>

<br />

### Image Classification with Transfer Learning
Transfer learning techniques were implemented using pre-trained models such as VGG16 and ResNet50 to classify images from the CIFAR-10 dataset, achieving an accuracy of 85%.

<br />

### Time Series Forecasting and Statistical Analysis of CO2 Emissions Trends
Time series forecasting models, including ARIMA and SARIMA, were developed to analyze trends and seasonal patterns in CO2 emissions data. Extensive statistical analysis, including residual diagnostics and model validation, was conducted to ensure robust predictions. Anomalies in the data were identified and corrected, enhancing the accuracy and reliability of the forecasting models.

The data spans from 1958 to 1997, using ARIMA to predict both near-term (2022) and long-term (2100) CO2 levels. The upper, expected, and lower bounds indicate when CO2 concentrations are likely to cross critical thresholds of 420 ppm and 500 ppm. These projections help in understanding potential future CO2 trends under various scenarios, providing valuable insights into long-term environmental planning and policy-making.

| Model      | sigma^2   | log_lik  | AIC      | AICc     | BIC      |
|------------|-----------|----------|----------|----------|----------|
| arima_full | 0.0860309 | -85.59152| 181.1830 | 181.3167 | 201.7845 |
| arima_test | 0.0857657 | -85.91671| 181.8334 | 181.9668 | 202.4459 |

![ARIMA Models](assets/images/ARIMA.png)

## Awards and Recognition
- **Top Finisher** for Innovate to Grow Event for 2022 Fall Software Engineering Capstone
- **3rd Place Winner** for the Water Hack Challenge 2023 Issued by Secure Water Future
