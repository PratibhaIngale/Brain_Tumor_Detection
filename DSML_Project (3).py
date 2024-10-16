#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[2]:


# Set the path to the dataset
 dataset_path = r"C:/Users/prati/Downloads/archive (5)"  # Added raw string 'r' to avoid escape issues

# Define the training and testing directories
train_dir = os.path.join(dataset_path, "Training")  
test_dir = os.path.join(dataset_path, "Testing")    

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# "C:\Users\prati\Downloads\Dataset"


# In[3]:


# Load and preprocess the dataset
train_data = []
for category in categories:
    folder_path = os.path.join(train_dir, category)
    images = os.listdir(folder_path)
    count = len(images)
    train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))

train_df = pd.concat(train_data, ignore_index=True)

# Visualize the distribution of tumor types in the training dataset
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x="Category", y="Count")
plt.title("Distribution of Tumor Types")
plt.xlabel("Tumor Type")
plt.ylabel("Count")
plt.show()


# In[4]:


# Visualize the distribution of tumor types in the training dataset with a pie chart
plt.figure(figsize=(6,6))

# Count the number of images for each category
category_counts = train_df['Category'].value_counts()

# Create a pie chart
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))

# Set aspect ratio to be equal, so pie chart is a circle
plt.axis('equal')
plt.title("Distribution of Tumor Types")
plt.show()


# In[5]:


# Visualize sample images for each tumor type
plt.figure(figsize=(12, 8))
for i, category in enumerate(categories):
    folder_path = os.path.join(train_dir, category)
    image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    img = plt.imread(image_path)
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[6]:


# Set the image size
image_size = (150, 150)

# Set the batch size for training
batch_size = 32

# Set the number of epochs for training
epochs = 50


# In[7]:


# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)


# In[8]:


# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(len(categories), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[9]:


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)


# In[10]:


# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()


# In[11]:


# Plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()


# In[12]:


# Evaluate the model
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[13]:


# Make predictions on the test dataset
predictions = model.predict(test_generator)
predicted_categories = np.argmax(predictions, axis=1)
true_categories = test_generator.classes

# Create a confusion matrix
confusion_matrix = tf.math.confusion_matrix(true_categories, predicted_categories)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.yticks(ticks=np.arange(len(categories)), labels=categories)
plt.show()


# In[14]:


# Plot sample images with their predicted and true labels
test_images = test_generator.filenames
sample_indices = np.random.choice(range(len(test_images)), size=9, replace=False)
sample_images = [test_images[i] for i in sample_indices]
sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imread(os.path.join(test_dir, sample_images[i]))
    plt.imshow(img)
    plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[15]:


from sklearn.metrics import classification_report

# Print classification report
print("CNN Classification Report:")
print(classification_report(true_categories, predicted_categories, target_names=categories))

# Calculate precision, recall, and F1-score from the confusion matrix
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print precision, recall, and F1-score for each class
# for i, category in enumerate(categories):
#     print(f"Class: {category}")
#     print(f"Precision: {precision[i]:.4f}")
#     print(f"Recall: {recall[i]:.4f}")
#     print(f"F1-Score: {f1_score[i]:.4f}")
#     print()


# In[16]:


### CNN Model ###
print("\n--- CNN Model ---")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(train_generator, epochs=10, validation_data=test_generator)


# In[17]:


# CNN Performance Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.legend()
plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical

# Get the true labels from the test generator
test_labels = test_generator.classes  # These are the actual labels from the test data

# Binarize the true labels for the specific class you want to evaluate
test_labels_binarized = to_categorical(test_labels, num_classes=len(categories))

# Predict on the test data
cnn_predictions = model.predict(test_generator)
cnn_predictions_proba = cnn_predictions[:, 0]  # Get probabilities for the 'glioma' class

# Compute ROC curve
fpr_cnn, tpr_cnn, _ = roc_curve(test_labels_binarized[:, 0], cnn_predictions_proba, pos_label=1)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

# Plot the ROC curve
plt.figure()
plt.plot(fpr_cnn, tpr_cnn, label='CNN ROC curve (area = %0.2f)' % roc_auc_cnn)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[19]:


# Predict on the test data
cnn_predictions = np.argmax(model.predict(test_generator), axis=1)

# Get the true labels from the test generator
test_labels = test_generator.classes  # These are the actual labels from the test data

# Convert tensors to numpy arrays if necessary
if isinstance(test_labels, tf.Tensor):
    test_labels = test_labels.numpy()
if isinstance(cnn_predictions, tf.Tensor):
    cnn_predictions = cnn_predictions.numpy()

# Create the confusion matrix
cnn_conf_matrix = confusion_matrix(test_labels, cnn_predictions)

# Print the classification report
print("CNN Classification Report:")
print(classification_report(test_labels, cnn_predictions, target_names=categories))


# In[20]:


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cnn_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[21]:


import os
import cv2
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[22]:


# Preprocess the images and extract HOG features
def preprocess_images(image_dir):
    data = []
    labels = []
    for category in categories:
        folder_path = os.path.join(image_dir, category)
        images = os.listdir(folder_path)
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Resize the image
            features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
            data.append(features)
            labels.append(category)
    return np.array(data), np.array(labels)


# In[23]:


# Load training and testing data
train_data, train_labels = preprocess_images(train_dir)
test_data, test_labels = preprocess_images(test_dir)

# Encode the labels into numeric values
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)


# In[24]:


### EDA ###
print("\n--- Exploratory Data Analysis ---")
# Visualize the distribution of classes in the training dataset
sns.countplot(x=train_labels)
plt.title('Class Distribution in Training Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Visualize the distribution of classes in the testing dataset
sns.countplot(x=test_labels)
plt.title('Class Distribution in Testing Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[26]:


from sklearn.model_selection import GridSearchCV

### KNN Model with GridSearchCV ###
print("\n--- KNN Model with GridSearchCV ---")
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_model = KNeighborsClassifier()

knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(train_data, train_labels_encoded)

# Best KNN parameters
print(f"Best KNN Params: {knn_grid_search.best_params_}")

# Make predictions with KNN
knn_predictions = knn_grid_search.predict(test_data)


# In[27]:


# Evaluate the k-NN model
knn_accuracy = accuracy_score(test_labels_encoded, knn_predictions)
print(f"k-NN Accuracy: {knn_accuracy * 100:.2f}%")
print("k-NN Classification Report:")
print(classification_report(test_labels_encoded, knn_predictions, target_names=categories))


# In[28]:


# Confusion Matrix for k-NN
knn_conf_matrix = confusion_matrix(test_labels_encoded, knn_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("k-NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[29]:


from sklearn.preprocessing import label_binarize

# Binarize the labels for ROC curve (One-vs-Rest)
test_labels_binarized = label_binarize(test_labels_encoded, classes=range(len(categories)))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Use the best fitted KNN model from GridSearchCV
best_knn_model = knn_grid_search.best_estimator_

for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], best_knn_model.predict_proba(test_data)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[30]:


# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(len(categories)):
    plt.plot(fpr[i], tpr[i], label=f"ROC curve (area = {roc_auc[i]:.2f}) for class {categories[i]}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for KNN Model")
plt.legend(loc="lower right")
plt.show()


# In[31]:


from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# In[32]:


### EDA ###
print("\n--- Exploratory Data Analysis ---")
# Visualize the distribution of classes in the training dataset
sns.countplot(x=train_labels)
plt.title('Class Distribution in Training Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[33]:


# Visualize the distribution of classes in the testing dataset
sns.countplot(x=test_labels)
plt.title('Class Distribution in Testing Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[34]:


print("\n--- SVM Model with RandomizedSearchCV ---")

# Define the parameter distribution for SVM
svm_param_dist = {
    'C': uniform(0.1, 10),  # Randomly choose values between 0.1 and 10 for 'C'
    'kernel': ['linear', 'rbf', 'poly'],  # Try linear, rbf, and polynomial kernels
    'gamma': ['scale', 'auto'],  # Gamma options for rbf and poly kernels
    'degree': [2, 3, 4]  # Degree of the polynomial kernel function
}


# In[35]:


# Initialize SVM model
svm_model = SVC(probability=True)

# Randomized Search with cross-validation
svm_random_search = RandomizedSearchCV(
    svm_model,
    svm_param_dist,
    n_iter=20,  # Number of parameter settings to try
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all available processors
)


# In[36]:


### SVM Model ###
print("\n--- SVM Model ---")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(train_data, train_labels_encoded)

# Make predictions with SVM
svm_predictions = svm_model.predict(test_data)


# In[37]:


# Fit the model
# svm_random_search.fit(train_data, train_labels_encoded)

# Best SVM parameters
# print(f"Best SVM Params: {svm_random_search.best_params_}")

# Make predictions with SVM using the best found parameters
# svm_predictions = svm_random_search.predict(test_data)

# Probability predictions (useful if you want to calculate ROC-AUC)
# svm_probabilities = svm_random_search.predict_proba(test_data)


# In[38]:


# Evaluate the SVM model
svm_accuracy = accuracy_score(test_labels_encoded, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print("SVM Classification Report:")
print(classification_report(test_labels_encoded, svm_predictions, target_names=categories))


# In[39]:


# Confusion Matrix for SVM
svm_conf_matrix = confusion_matrix(test_labels_encoded, svm_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(svm_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[40]:


### ROC Curve ###
print("\n--- ROC Curve ---")
# Binarize the output labels for ROC analysis
test_labels_binarized = label_binarize(test_labels_encoded, classes=np.arange(len(categories)))


# In[41]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], svm_model.predict_proba(test_data)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[42]:


# Plot ROC curve
plt.figure(figsize=(10, 8))
for i in range(len(categories)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {categories[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for SVM')
plt.legend(loc='lower right')
plt.show()


# In[43]:


from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize


# In[44]:


# Initialize and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_data, train_labels_encoded)

# Make predictions on the test data
predictions = rf_model.predict(test_data)


# In[45]:


### EDA ###
print("\n--- Exploratory Data Analysis ---")
# Visualize the distribution of classes in the training dataset
sns.countplot(x=train_labels)
plt.title('Class Distribution in Training Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[46]:


# Visualize the distribution of classes in the testing dataset
sns.countplot(x=test_labels)
plt.title('Class Distribution in Testing Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

### Random Forest Model with RandomizedSearchCV ###
print("\n--- Random Forest Model with RandomizedSearchCV ---")
rf_param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': randint(3, 20)
}
rf_model = RandomForestClassifier(random_state=42)

rf_random_search = RandomizedSearchCV(rf_model, rf_param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_random_search.fit(train_data, train_labels_encoded)

# Best Random Forest parameters
print(f"Best Random Forest Params: {rf_random_search.best_params_}")

# Make predictions with Random Forest
rf_predictions = rf_random_search.predict(test_data)


# In[ ]:


# Evaluate the model
accuracy = accuracy_score(test_labels_encoded, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(test_labels_encoded, predictions, target_names=categories))


# In[ ]:


# Confusion Matrix
conf_matrix = confusion_matrix(test_labels_encoded, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[ ]:


### ROC Curve ###
print("\n--- ROC Curve ---")
# Binarize the output labels for ROC analysis
test_labels_binarized = label_binarize(test_labels_encoded, classes=np.arange(len(categories)))


# In[ ]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], rf_random_search.predict_proba(test_data)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves or do further analysis


# In[ ]:


# Plot ROC curve
plt.figure(figsize=(10, 8))
for i in range(len(categories)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {categories[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Performance metrics data
models = ['CNN', 'SVM', 'KNN', 'Random Forest']
accuracy = [94.37, 94.74, 89.24, 90.77]
precision = [0.93, 0.95, 0.88, 0.91]
recall = [0.94, 0.94, 0.89, 0.90]
f1_score = [0.94, 0.94, 0.88, 0.90]

# Create a DataFrame for better handling
metrics_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score
})

# Bar plot for accuracy
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pie chart for accuracy
plt.figure(figsize=(6, 6))
plt.pie(accuracy, labels=models, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Model Accuracy Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[ ]:


# ROC/AUC for All Models
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.title('ROC Curves for KNN, SVM, and Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# model.save('C:/Users/prati/Downloads/brain__tumor__detection_model.h5')

