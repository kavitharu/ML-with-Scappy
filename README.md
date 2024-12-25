# ML-with-Scappy
An AI based real time Malware detection tool using Scappy.
AN AI-BASED MALWARE DETECTION SYSTEM USING PACKETS CAPTURED BY SCAPPY TOOL
M.A.K. Tharika


Abstract

This research presents a novel AI-based malware detection system utilizing Scapy for packet capture and analysis. The system leverages advanced machine learning algorithms to identify malicious patterns within network traffic, enabling proactive detection and prevention of sophisticated malware attacks. By employing Scapy's powerful capabilities for packet manipulation and analysis, the proposed system offers a robust and efficient solution for safeguarding networks against emerging cyber threats, utilizing Scapy to capture and extract relevant features from network traffic, such as protocol headers, payload data, and timing information. Employing state-of-the-art algorithms like deep neural networks, recurrent neural networks, or support vector machines for accurate malware classification. Identifying and selecting the most discriminative features to enhance detection accuracy and reduce computational overhead and implementing a system capable of processing network traffic in real-time to enable timely. Rigorously evaluating the system's performance using various datasets and comparing it to existing malware detection methods. The proposed AI-based malware detection system offers a promising approach to combating the ever-evolving landscape of cyber threats. By combining the power of Scapy with advanced machine learning techniques, this system can provide organizations with a valuable tool for protecting their networks and data.



1.	Introduction 

As cyber threats continue to evolve, organizations face increasingly sophisticated malware attacks targeting their networks and sensitive data. Traditional malware detection methods, often signature-based, struggle to keep pace with new and emerging threats that exploit network vulnerabilities. This paper presents a novel AI-based malware detection system leveraging Scapy for real-time packet capture and analysis, combined with advanced machine learning techniques for accurate classification of malicious traffic. By harnessing the power of deep learning algorithms and Scapy's packet manipulation capabilities, this system offers a proactive solution to detect and prevent malware within network traffic.  (Biondi, 2004)
The system captures and extracts key features from network packets, including protocol headers, payload data, and timing information, which are then processed by machine learning models such as deep neural networks (DNN), recurrent neural networks (RNN), or support vector machines (SVM). These models are trained to identify malicious patterns and anomalies within the traffic. The proposed solution not only enhances detection accuracy by identifying the most discriminative features but also minimises computational overhead, enabling real-time detection. Rigorous evaluation against benchmark datasets demonstrates the system’s effectiveness compared to existing detection methods, providing a promising tool for defending against the rapidly evolving landscape of cyber threats.  (Vinayakumar, 2017)
              
2.	Methodology

	The proposed methodology includes the following steps:
1.	Dataset Acquisition: Utilizing the Kaggle API, the malware detection dataset is downloaded and stored for preprocessing. For the dataset below, a dataset from Kaggle has been used,
Malware dataset.csv(18.01 MB)
 
Figure 1: The dataset.
To call the data set, use a line of Python code that utilizes the panda’s library. The code reads a CSV (Comma-Separated Values) file and loads its contents into a Pandas DataFrame. A data frame is a two-dimensional labelled data structure with columns that can hold different data types. It's a fundamental data structure used for data analysis and manipulation in Python.
2.	Data Preprocessing: The dataset is cleaned by filling in missing values and encoding labels. Features are selected based on their relevance to the detection task.
In the second step of methodology, data pre-processing was conducted using handling missing values, converting categorical labels to numerical values, removing irrelevant features, and separating the features and labels into X and y for further processing, such as splitting into training and testing sets and training a machine learning model. This Python script for the preprocessing has the following functions:
This starts by handling missing values in feature columns by filling them with 0. Then, it maps categorical labels in the 'classification' column to numerical values. Next, it removes rows with missing labels to ensure data integrity. Irrelevant features, like the 'hash' column, are dropped. Finally, the features and labels are separated into X and y, respectively, ready for model training and evaluation.  (Géron, 2019)
3.	Feature Scaling and Splitting: Data is split into training and testing sets, then normalization using StandardScaler to ensure consistent input for the machine learning models. In the Python code for feature extracting and splitting, the script splits the dataset into training and testing sets, with 20% of the data allocated for testing. It then standardizes the features using StandardScaler, which scales the features to have zero mean and unit variance. This is crucial for many machine learning algorithms as it ensures that features with larger scales don't dominate the learning process. The scaled training and testing sets, along with the original labels, are returned for further model training and evaluation. Additionally, the scaler object is returned to apply the same scaling transformation to new data in the future. 
4.	Model Development: A deep learning model is constructed using TensorFlow, consisting of multiple layers designed for binary classification of malware. The code defines a function to build and train a simple Deep Neural Network (DNN) model for binary classification. The model consists of three layers: an input layer, two hidden layers with ReLU activation, and an output layer with a sigmoid activation for binary classification. The model is compiled using the Adam optimizer to minimize binary cross-entropy loss, and accuracy is used as the evaluation metric. This model is ready to be trained in the provided training data. 
5.	Training and Evaluation: The model is trained on the training set, with its performance evaluated on the test set using metrics like accuracy and F1-score.
In the script training and evaluation have been conducted as below,
Training:
model.fit(...): Trains the model on the training data (X_train_scaled, y_train) for a specified number of epochs (epochs) and batch size (batch_size).
validation_data=(X_test_scaled, y_test): Uses the test set for validation during training, helping to prevent overfitting.
Evaluation:
model.evaluate(...): Evaluates the model's performance on the unseen test set (X_test_scaled, y_test), returning metrics like test loss and accuracy.
classification_report(...): Prints a detailed report on the model's performance for binary classification, including precision, recall, F1-score, and support for each class.
confusion_matrix(...): Calculates and prints the confusion matrix, visualizing how many data points were correctly and incorrectly classified.

6.	Real-Time Detection: The model is integrated with Scapy for live network packet analysis, allowing for the detection of potential threats as they occur.  
For real-time malware detection, a trained machine-learning model. It extracts relevant features from incoming network packets, such as packet length, protocol, and timestamp. These features are then standardized using the same scale used for training. The standardized packet data is fed into the trained model to obtain a prediction. If the prediction probability exceeds a certain threshold (0.5 here), the packet is flagged as malicious. Otherwise, it's considered benign. This real-time detection can be integrated into network security systems to proactively identify and mitigate potential threats.
 
Figure 2: The python script - pt. 1
 
Figure 3: The python script - pt. 2
 
Figure 4: The Python script - pt.3
 
Figure 5: The Python script main ().

3.	Results and Discussion

The model results can be shown below,

 
Figure 6: Training and validating the accuracy of a machine learning model over 10 epochs.
    Figure 06 declares these facts; The blue line represents the accuracy of the training data. It starts at a relatively low value and quickly increases to nearly 1.00 within the first few epochs. This indicates that the model is learning effectively from the training data. The orange line represents the accuracy of the validation data, which is a separate set of data not used for training. It also increases rapidly and plateaus at a high value, which is very close to training accuracy. Since the validation accuracy is very close to the training accuracy, it is safe to say there is no significant overfitting. The model achieves high accuracy on both training and validation data, suggesting that it has learned the underlying patterns effectively. 
The model performance can be shown using below Figure 7,
 
Figure 7: Performance of the model.
Figure 7 declares that, 
The Precision is 1.00, which means that for class 0 (likely benign), all predicted positive instances were actually positive, recall is 1.00, which indicates that the model correctly identified all actual positive instances of class 0, F1-score is 1.00 which is the harmonic mean of precision and recall, and a value of 1.00 suggests perfect performance, accuracy is 1.00 which means that the model correctly predicted the class for all 10000 instances, macro avg: is 1.00 which is the average of precision, recall, and F1-score across all classes. A value of 1.00 suggests perfect performance for all classes and the weighted average is 1.00 which is the weighted average of precision, recall, and F1-score, considering the class distribution. A value of 1.00 indicates perfect performance across all classes, even if they have different weights. Overall, these results suggest that the model is highly accurate and effective in classifying the given dataset. However, it's important to note that this is likely an ideal scenario and real-world datasets may present more challenges.
 
Figure 8: The real time capturing results.
Figure 8 shows the real-time capturing results from the model.

4.	Conclusions

This research proposed a novel AI-based malware detection system using Scapy for network traffic analysis and machine learning for classification. The system leverages deep learning models to identify malicious patterns in packets, enabling real-time threat detection. Experimental results demonstrate excellent performance on a benchmark dataset, achieving high accuracy with minimal overfitting. The system integrates with Scapy for real-time processing, making it a promising tool for organizations to combat evolving cyber threats. However, it's important to acknowledge that real-world scenarios might pose additional challenges.
References 
Abadi, M. B. P. C. J. e. a., 2016. TensorFlow: A System for Large-Scale Machine Learning.. 2th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)., p. 265–283.
Biondi, P. &. D. F., 2004. Scapy: a Python tool for packet manipulation and network exploration., CanSecWest.: Scappy.
Biondi, P., 2024. Scapy: A Python Library for Packet Manipulation and Network Analysis. [Online] 
Available at: https://scapy.readthedocs.io/en/latest/
García, S. L. J. &. H. F., 2015. Data Preprocessing in Data Mining., c: SpringerBriefs in Computer Science.
Géron, A., 2019. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, Germany: O'Reilly Media, Inc..
Kim, D. &. L. J., 2020. Using Deep Learning for Cybersecurity." IEEE Access,. IEEE , p. 153185–153198.
Vinayakumar, R. S. K. P. &. P. P. (., 2017. Applying deep learning approaches for network traffic analysis and intrusion detection. 2017 International Conference on Advances in Computing, C, CAnsas: IEEE.
Vinod, P. J. R. L. V. &. G. M. S., 2019. A Survey on Malware Detection Using Machine Learning Techniques.. Advances in Intelligent Systems and Computing, pp. 227-245.



