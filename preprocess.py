import numpy as np
# sys.path.insert(0, './')
import warnings
warnings.filterwarnings(action='ignore')
import time
number_of_vector_per_example = 100 # for nina, it is 52
number_of_canals = 8 # for nina, it is 16
size_non_overlap = 5


label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
file_name = ['subject_0']
######
data_name = 'raw_data/SD-Gesture' # change it to your data directory
i = 0

def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    t1 = time.time()
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (example == []):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                # print("time is : ", time.time()-t1)
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
                # break
    data_calculated =  dataset_example_formatted
    return np.array(data_calculated)



def read_data():
    final_dataset1 = []
    final_labels1 = []
    final_dataset2 = []
    final_labels2 = []
    final_dataset3 = []
    final_labels3 = []
    final_dataset4 = []
    final_labels4 = []
    for name in file_name:
        list_dataset1 = []
        list_labels1 = []
        list_dataset2 = []
        list_labels2 = []
        list_dataset3 = []
        list_labels3 = []
        list_dataset4 = []
        list_labels4 = []
        for candidate in range(18): # for nina, it is 53
            label_index = candidate
            labels1 = []
            examples1 = []
            labels2 = []
            examples2 = []
            labels3 = []
            examples3 = []
            labels4 = []
            examples4 = []
            count = 0
            t1= time.time()
            for file in open('./data/raw_data/SD-Gesture/' + name + '/' +label_name[candidate]+'.txt', 'r'): # +str(label_index)+'-'
                tmp = file.strip()
                inter = [int(i)-2000 for i in tmp.split(',')]
                inter = inter[:8]
                inter = np.array(inter)
                inter = inter.reshape(1, 8)
                if count == 0:
                    data = inter
                else:
                    data = np.concatenate((data, inter), axis=1)
                count += 1

            dataset_example1 = format_data_to_train(data[0][0:len(data[0])//4])
            dataset_example2 = format_data_to_train(data[0][len(data[0]) // 4:2 * (len(data[0]) // 4)])
            dataset_example3 = format_data_to_train(data[0][2 * (len(data[0]) // 4):3 * (len(data[0]) // 4)])
            dataset_example4 = format_data_to_train(data[0][3 * (len(data[0]) // 4):4 * (len(data[0]) // 4)])


            examples1.append(dataset_example1)
            labels1.append(label_index + np.zeros(dataset_example1.shape[0]))
            list_dataset1.append(np.array(examples1[0]))
            list_labels1.append(np.array(labels1[0]))


            examples2.append(dataset_example2)
            labels2.append(label_index + np.zeros(dataset_example2.shape[0]))
            list_dataset2.append(np.array(examples2[0]))
            list_labels2.append(np.array(labels2[0]))


            examples3.append(dataset_example3)
            labels3.append(label_index + np.zeros(dataset_example3.shape[0]))
            list_dataset3.append(np.array(examples3[0]))
            list_labels3.append(np.array(labels3[0]))

            examples4.append(dataset_example4)
            labels4.append(label_index + np.zeros(dataset_example4.shape[0]))
            list_dataset4.append(np.array(examples4[0]))
            list_labels4.append(np.array(labels4[0]))

        final_dataset1.append(list_dataset1)
        final_labels1.append(list_labels1)
        final_dataset2.append(list_dataset2)
        final_labels2.append(list_labels2)
        final_dataset3.append(list_dataset3)
        final_labels3.append(list_labels3)
        final_dataset4.append(list_dataset4)
        final_labels4.append(list_labels4)

    ##5초
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep1_data', np.array(final_dataset1), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep1_label', np.array(final_labels1), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep2_data', np.array(final_dataset2), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep2_label', np.array(final_labels2), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep3_data', np.array(final_dataset3), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep3_label', np.array(final_labels3), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep4_data', np.array(final_dataset4), allow_pickle=True)
        np.save(f'./data/processed_data/SD-Gesture/{file_name[0]}/rep4_label', np.array(final_labels4), allow_pickle=True)
        print(name)
    return np.array(list_dataset1), np.array(list_labels1)                 #np.array(list_dataset), np.array(list_labels)



if __name__ == "__main__":
    read_data()
