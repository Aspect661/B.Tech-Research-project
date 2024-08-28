# B.Tech-Research-project

This is my final year B.Tech project. I am an Electronics and Communications Engineering undergraduate from Calcutta University. 

Project: PPG-Based Continuous Authentication and Subject Identification: A Machine Learning Approach with Hardware Implementation

In today's digital age, where security is paramount, this project aims to develop a robust continuous authentication (CA) system utilizing photoplethysmogram (PPG) signals. By harnessing the unique physiological characteristics of PPG signals, the system provides highly accurate user verification, effectively mitigating risks associated with unauthorized access and data breaches. Leveraging the BIDMC dataset, the project employs advanced machine learning algorithms such as Random Forest, SVM, and XGBoost, achieving over 95% accuracy and F1 score, thereby demonstrating its significant potential to enhance digital security through reliable user authentication.

The system is designed as a two-stage model. The first stage addresses the authorization process using eXtreme Gradient Boosting Outlier Detection (XGBOD), while the second stage focuses on identifying authorized users with the help of a Random Forest Classifier.

However, several challenges emerged in the real-time implementation of this model. One of the primary issues is the time-varying nature of PPG signals, which the model does not account for, leading to false positives over time. This was particularly evident during video analysis. For this project, I collected 10 minutes of PPG data from 7 different subjects in a controlled lab environment. However, unauthorized users were correctly identified for the most part. 

The hardware implementation involved integrating a Raspberry Pi 3B+, a custom ADC board, and a PPG sensor, demonstrating the feasibility of real-time deployment of the CA system. Despite the challenges, this project lays the groundwork for future improvements in continuous authentication systems, emphasizing the importance of addressing time-varying signals for enhanced accuracy.


The ipynb file which was responsible for testing of the model on benchmark dataset is uploaded. The final .py code which is finally implemented in the RPI 3B+ is also uploaded. The .jpeg shows a snap of the model in action (Sahana was the name of my project partner)
