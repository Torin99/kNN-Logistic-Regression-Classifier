#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import heapq
from math import sqrt
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import statistics



# In[2]:


def readFileLog(filename):
    data = []
    labels = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                row = line.split(",")
                if filename == "adult.data":
                    # Convert non-numerical features to float
                    age = float(row[0])
                    fnlwgt = float(row[2])
                    education_num = float(row[4])
                    capital_gain = float(row[10])
                    capital_loss = float(row[11])
                    hours_per_week = float(row[12])
                    # Combine the numerical features
                    numerical_features = [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week]
                    # Append the numerical features
                    data.append(numerical_features)
                    label = row[-1]
                    # Map the labels to binary values, e.g., '<=50K' to 0 and '>50K' to 1
                    labels.append(0 if label == ' <=50K' else 1)
                elif filename == "Rice_Cammeo_Osmancik.arff.txt":
                    data.append([float(val) for val in row[:-1]])
                    label = row[-1]
                    labels.append(0 if label == 'Cammeo' else 1)
                elif filename == "agaricus-lepiota.data":
                    label = 0 if row[0] == 'e' else 1
                    labels.append(label)

                    # Define mappings for categorical values
                    cap_shape_mapping = {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5}
                    cap_surface_mapping = {'f': 0, 'g': 1, 'y': 2, 's': 3}
                    cap_color_mapping = {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9}
                    bruises_mapping = {'t': 0, 'f': 1}
                    odor_mapping = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8}
                    gill_attachment_mapping = {'a': 0, 'd': 1, 'f': 2, 'n': 3}
                    gill_spacing_mapping = {'c': 0, 'w': 1, 'd': 2}
                    gill_size_mapping = {'b': 0, 'n': 1}
                    gill_color_mapping = {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11}
                    stalk_shape_mapping = {'e': 0, 't': 1}
                    stalk_root_mapping = {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6}
                    stalk_surface_above_ring_mapping = {'f': 0, 'y': 1, 'k': 2, 's': 3}
                    stalk_surface_below_ring_mapping = {'f': 0, 'y': 1, 'k': 2, 's': 3}
                    stalk_color_above_ring_mapping = {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8}
                    stalk_color_below_ring_mapping = {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8}
                    veil_type_mapping = {'p': 0, 'u': 1}
                    veil_color_mapping = {'n': 0, 'o': 1, 'w': 2, 'y': 3}
                    ring_number_mapping = {'n': 0, 'o': 1, 't': 2}
                    ring_type_mapping = {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7}
                    spore_print_color_mapping = {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8}
                    population_mapping = {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5}
                    habitat_mapping = {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6}

                    # Convert non-numerical features to float using the mappings
                    encoded_features = [
                        cap_shape_mapping[row[1]],
                        cap_surface_mapping[row[2]],
                        cap_color_mapping[row[3]],
                        bruises_mapping[row[4]],
                        odor_mapping[row[5]],
                        gill_attachment_mapping[row[6]],
                        gill_spacing_mapping[row[7]],
                        gill_size_mapping[row[8]],
                        gill_color_mapping[row[9]],
                        stalk_shape_mapping[row[10]],
                        stalk_root_mapping[row[11]],
                        stalk_surface_above_ring_mapping[row[12]],
                        stalk_surface_below_ring_mapping[row[13]],
                        stalk_color_above_ring_mapping[row[14]],
                        stalk_color_below_ring_mapping[row[15]],
                        veil_type_mapping[row[16]],
                        veil_color_mapping[row[17]],
                        ring_number_mapping[row[18]],
                        ring_type_mapping[row[19]],
                        spore_print_color_mapping[row[20]],
                        population_mapping[row[21]],
                        habitat_mapping[row[22]]
                    ]
                    data.append(encoded_features) 
                        
                else:
                    data.append([float(val) for val in row[:-1]])
                    label = row[-1]
                    labels.append(0 if label == 'b' else 1)


    return data, labels


# In[3]:


def readFileKNN(filename):
    file_data = []
    labels = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                row = line.split(",")
                if "?" not in row and " ?" not in row:
                    file_data.append(row)

    data = np.array(file_data)
    if filename == "data/adult.data" or filename == "data/Short_adult.data" :
        workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", '1st-4th', "10th", 'Doctorate', '5th-6th', "Preschool"]
        marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        occupation = ["Tech-support", "Craft-repair", "Other-service", 'Sales', "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        race = ['White', 'Asian-Pac-Islander', "Amer-Indian-Eskimo", 'Other', 'Black']
        sex = ['Female', 'Male']
        native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago','Peru', 'Hong', 'Holand-Netherlands']

        columns = [workclass, education, marital_status,occupation,relationship,race,sex,native_country]

        for c in columns:
            for i in range(len(c)):
                word = c[i]
                data[data==" " + word] = str(i)

        data = np.char.strip(data) 
    elif filename == "data/Short_agaricus-lepiota.data" or filename == "data/Short_agaricus-lepiota.data":
        def char_to_alphabet_index(char):
            if 'a' <= char <= 'z':
                return ord(char) - ord('a') + 1
            elif 'A' <= char <= 'Z':
                return ord(char) - ord('A') + 1
            else:
                return 0  # Return 0 for non-alphabet characters
    
        labels = data[:, 0]
        values = data[:,1:]

        # Vectorize the function to apply it to the entire array
        index_converter = np.vectorize(char_to_alphabet_index)

        # Apply the conversion function to the entire array
        result = index_converter(values)

        data = np.hstack((result, labels.reshape(-1, 1)))
    return data


def bestKValue(KNNmodel, dataSet, kRange=10):
    accuracy = []
    k_val =[]
    for k in range(kRange):
        k_val.append(k+1)
        KNNmodel.setK(k+1)
        accuracy.append(KNNmodel.kFoldCross(dataSet,5, False))
        
    plt.figure(figsize=(8, 6))
    plt.plot(k_val, accuracy, marker='o', linestyle='-')
    plt.title('KNN Model Accuracy vs. k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# In[ ]:
class dataAnalysis:
    def __init__(self, data, categories, classifications):
        self.data = data
        self.size = len(data)
        self.categories = categories
        self.classifications = classifications
    def printLabelStats(self):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Print statistics on every classification in data
        ===================================================================================
        '''
        print("=====================================================")
        print("Classification Analysis:")
        print("=====================================================")
        print(f"Total: {self.size} (%100)")
        
        for label in self.classifications:
            count = len(self.data[self.data[:, -1] == label])
            percentage = round(100*count/self.size,2)
            print(f"Value: {label}, Count: {count}, Percentage: %{percentage}")
            
    def printCategoryStats(self, max_ranges=10):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Print statistics on every attribute in data, and display broader count ranges if necessary
        ===================================================================================
        '''
        data = self.data
        categories = self.categories
        size = self.size
        labels = column = data[:, -1]
        print("=====================================================")
        print("Attribute Analysis:")
        print("=====================================================")
        for i in range(len(categories)):
            category = categories[i]
            column = data[:, i].astype(float)

            avg = np.mean(column)
            med = np.median(column)
            mode = float(stats.mode(column, keepdims=True)[0][0])
            std_dev = np.std(column)

            unique_values, counts = np.unique(column, return_counts=True)
            num_unique_values = len(unique_values)

            if num_unique_values > max_ranges:
                # Determine range width based on the number of unique values
                range_width = (np.max(unique_values) - np.min(unique_values)) / max_ranges
                range_counts = []
                range_start = unique_values[0]
                current_range_count = 0
                for value, count in zip(unique_values, counts):
                    if value - range_start <= range_width:
                        current_range_count += count
                    else:
                        range_counts.append((range_start, range_start + range_width, current_range_count))
                        range_start = value
                        current_range_count = count
                range_counts.append((range_start, range_start + range_width, current_range_count))

                print(f"-----------------------------------------------------")
                print(f"{category.upper()} Analysis (Count Ranges):")
                print(f"-----------------------------------------------------")

                for range_start, range_end, count in range_counts:
                    percentage = round(100 * count / size, 2)
                    print(f"Range: [{range_start} - {range_end}], Count: {count}, Percentage: %{percentage}")
            else:
                print(f"-----------------------------------------------------")
                print(f"{category.upper()} Analysis:")
                print(f"-----------------------------------------------------")
                for value, count in zip(unique_values, counts):
                    percentage = round(100 * count / size, 2)
                    print(f"Value: {value}, Count: {count}, Percentage: %{percentage}")

            print(f"\nMean: {avg}")
            print(f"Median: {med}")
            print(f"Mode: {mode}")
            print(f"Standard Deviation: {std_dev}")

            plt.figure(figsize=(4, 4))
            plt.bar(labels, column, edgecolor='black')
            plt.xlabel("classification")
            plt.ylabel(category)
            plt.title("Bar Chart of Numeric Data by Labels")
            plt.show()



