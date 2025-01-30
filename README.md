### **Overview of the Spam Text Classification Process Using RNN**

In this notebook, we will walk through the steps to classify text messages as either 'ham' (non-spam) or 'spam' using a Recurrent Neural Network (RNN). The dataset used is an SMS Spam Classification dataset, which contains labeled messages ('ham' and 'spam'). We will preprocess the data, build a model, train it, and finally use the trained model to predict whether new messages are spam or not.

### **Steps Involved:**

1. **Import Libraries and Dependencies**:
   - The necessary libraries such as `NumPy`, `Pandas`, `TensorFlow`, `Keras`, and `Seaborn` are imported. These will be used for data manipulation, model creation, evaluation, and visualization.

2. **Load and Preprocess the Dataset**:
   - The dataset is loaded from a URL using `Pandas` and the relevant columns are renamed. The dataset contains SMS messages labeled as either 'ham' (non-spam) or 'spam'.

3. **Data Exploration and Visualization**:
   - The distribution of 'ham' and 'spam' messages is visualized using a bar chart. Descriptive statistics of the dataset are displayed to get an overview of message types, lengths, and other attributes.

4. **Text Cleaning for Ham and Spam Messages**:
   - Text data from both 'ham' and 'spam' messages is cleaned by converting words to lowercase and removing unnecessary characters. This cleaned text is then used to generate WordClouds to visualize the most frequent terms in both message types.

5. **Balancing the Dataset**:
   - To avoid class imbalance, an equal number of 'ham' and 'spam' messages are sampled for training.

6. **Text Length and Label Encoding**:
   - The length of each message is computed, and the 'ham' and 'spam' labels are encoded as 0 and 1, respectively. The labels are mapped into a new column, `msg_type`.

7. **Splitting Data into Training and Testing Sets**:
   - The data is split into training and testing sets, where 80% of the data is used for training and 20% for testing.

8. **Tokenization and Padding**:
   - The text messages are tokenized into sequences, where each word is converted into a numeric value. Padding is applied to ensure all input sequences have the same length, making them suitable for model input.

9. **Model Architecture**:
   - A Sequential model is created using an Embedding layer, an LSTM (Long Short-Term Memory) layer for sequential learning, and a Dense output layer with a sigmoid activation function to predict whether the message is 'ham' or 'spam'.

10. **Model Training**:
    - The model is compiled using the `Adam` optimizer and binary cross-entropy loss function. It is then trained on the training data for 30 epochs, with validation on the test set to monitor the model's performance.

11. **Message Prediction**:
    - After training, the model is used to classify new messages. A function is defined to input new text, preprocess it, and predict whether each message is 'ham' or 'spam' based on the trained model.

12. **Additional Message Predictions**:
    - Additional messages are tested with the trained model to evaluate its performance on new unseen data.

### **Key Components:**

- **WordCloud**: Used to visualize frequent terms in both 'ham' and 'spam' messages.
- **LSTM Model**: Used to capture sequential dependencies in the text, making it ideal for spam detection in SMS messages.
- **Tokenization and Padding**: Essential preprocessing steps for converting raw text data into a format suitable for neural networks.
- **Model Evaluation**: The model's performance is monitored during training, and it is evaluated using accuracy on the test set.

### **Outcome**:
After completing these steps, you will have a model that is able to accurately classify whether a message is spam or not. You will also be able to predict new messages using the trained model.

### **Contact Information:**
- **Project Developed by**: Karan Bhosle
- **LinkedIn Profile**: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)
