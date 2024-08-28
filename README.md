# B.Tech-Research-project

This repository consists of my final year B.Tech project. I am an Electronics and Communications Engineering undergraduate from Calcutta University. 

Project: PPG-Based Continuous Authentication and Subject Identification: A Machine Learning Approach with Hardware Implementation

In the current digital landscape, where security is of utmost importance, this project is of significant value as it aims to develop a robust continuous authentication (CA) system using photoplethysmography (PPG) signals. By leveraging the unique physiological characteristics of PPG signals, the system offers highly accurate user verification, effectively reducing the risks associated with unauthorized access and data breaches. The project achieves over 95% accuracy and F1 score, demonstrating its potential to significantly enhance digital security through reliable user authentication. Utilizing the BIDMC dataset and advanced machine learning algorithms such as Random Forest, SVM, and XGBoost, the project offers a high level of confidence in the audience about the reliability of the system.

The system is designed as a two-stage model. The first stage addresses the authorization process using eXtreme Gradient Boosting Outlier Detection (XGBOD), while the second stage focuses on identifying authorized users with the help of a Random Forest Classifier.

However, several challenges emerged in the real-time implementation of this model. One of the primary issues is the time-varying nature of PPG signals, which the model does not account for, leading to false positives over time. This was particularly evident during video analysis. Another challenge was the need for a robust outlier detection system, as the model was prone to misidentifying outliers as unauthorized users. I collected 10 minutes of PPG data from 7 subjects in a controlled lab environment for this project. However, unauthorized users were correctly identified for the most part. 

The hardware implementation, which involved integrating a Raspberry Pi 3B+, a custom ADC board, and a PPG sensor, successfully demonstrated the feasibility of real-time deployment of the CA system. This successful demonstration should reassure the audience about the practicality of the system, despite the challenges faced. This project paves the way for future advancements in continuous authentication systems, particularly in addressing time-varying signals for enhanced accuracy. This potential for future improvements should inspire confidence in the audience about the evolution of digital security.


The .ipynb file, used to test the model on the benchmark dataset, has been uploaded. The final .py code, finally implemented in the RPI 3B+, has also been uploaded. The .jpeg shows a snap of the model in action (Sahana was the name of my project partner)
