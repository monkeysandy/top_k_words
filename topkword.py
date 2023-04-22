import pandas as pd
import numpy as np
import re
import os
import time
import multiprocessing as mp
import psutil

file_path = 'dataset/data_300MB.txt'
stopword_path = 'stopword.txt'

def count_words(chunk, stop_words):
    # Count word frequencies in a chunk

    """
        Input: chunk - list of words
            stop_words - set of stop words
        Output: Dictionary with word frequencies
    """

    word_count = {}
    for word in chunk:
        # Only count words that are not stop words
        # and are alphabetic
        if word not in stop_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count

def top_k_words(file_path, stopword_path, k, shard, num_processes):
    # Count word frequencies in a file
    # and return top k words and their frequencies

    """
        Input: file_path - path to file
               stopword_path - path to stopword file
               k - number of top words to return
               shard - number of shards to split file into
               num_processes - number of processes to use
        Output: DataFrame with top k words and their frequencies
                sorted in descending order
    """

    # Reads in a list of stopwords from "stopword_path"
    # and stores them in a set
    # strip() removes trailing whitespace
    stop_words = set(line.strip() for line in open(stopword_path))
    freq = {}
    
    # Determine chunk size and number of chunks
    # the input file will be split into
    # file_size = os.path.getsize(file_path)
    # file_size = int(os.popen('wc -l < {}'.format(file_path)).read())
    file_size = int(os.popen('wc -c < {}'.format(file_path)).read())
    chunk_size = int(np.ceil(file_size / shard))
    # num_chunks = file_size // chunk_size + 1
    
    # Set up multiprocessing pool
    pool = mp.Pool(num_processes)
    
    # Process each chunk in a separate process
    # and store the results in a list
    results = []
    with open(file_path) as f:
        for i in range(shard):
            # count time of average sharding time
            chunk = f.read(chunk_size).lower()
            # apply_async() schedules the function to be executed asynchronously
            # in a separate process and returns a 'asyncresult' object
            # which is added to the results list
            results.append(pool.apply_async(count_words, args=(re.findall(r'\w+', chunk), stop_words)))
    
    # Combine word counts from all processes
    for result in results:
        word_count = result.get()
        for word in word_count:
            if word in freq:
                freq[word] += word_count[word]
            else:
                freq[word] = word_count[word]
    
    # Convert word frequencies to Pandas DataFrame and sort
    freq_df = pd.DataFrame({'word': list(freq.keys()), 'count': list(freq.values())})
    # Only keep alphabetic words
    freq_df = freq_df[freq_df['word'].apply(lambda x: x.isalpha())]
    freq_df = freq_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    # Print top k words and their frequencies
    print(freq_df.head(k))

def MainUserInterface():
    print("------------ Welcome to the Word Frequency Counter ------------")
    print("Please enter the path to the file you would like to analyze")
    file_path = input()
    if not os.path.exists(file_path):
        print("Please enter a valid path")
    
    print("Please enter the path to the stopword file")
    stopword_path = input()
    if not os.path.exists(stopword_path):
        print("Please enter a valid path")
    
    print("Please enter the number of top words you would like to see")
    k = int(input())
    if k < 1:
        print("Please enter a positive integer")
    
    print("Please enter the shard number you would like to split the file into")
    shard = int(input())
    if shard < 1:
        print("Please enter a positive integer")
    file_size = int(os.popen('wc -c < {}'.format(file_path)).read())
    # num_chunks = file_size // chunk_size + 1
    # num_chunks = os.path.getsize(file_path) // chunk_size + 1
    chunk_size = int(np.ceil(file_size / shard))

    print("Please enter the number of processes you would like to use [default: 4]")
    num_processes = int(input())
    if num_processes < 1:
        print("Please enter a positive integer")
    print('---------Starting Word Frequency Counter---------')
    print('Starting on {} using {} processes'.format(file_path, num_processes))
    print('File size: {} bytes'.format(file_size))
    print('shard number: {}'.format(shard))
    print('chunk size: {} bytes'.format(chunk_size))
    print('------------TOP {} WORDS ------------'.format(k))

    # print('Current total memory: {} GB'.format(psutil.virtual_memory()[0] / 1e9))
    # print('Current available memory: {} GB'.format(psutil.virtual_memory()[1] / 1e9))
    # print('Current CPU usage: {} %'.format(psutil.cpu_percent()))
    
    start_time = time.time()
    top_k_words(file_path, stopword_path, k, shard, num_processes)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

def test():
    # helper function to test the performance of the program
    K_list = [10, 15]
    shard_list = [10, 20, 50, 100, 200, 400, 500, 750, 1000]
    print("Generating test results...")
    resulttime = {}
    for k in K_list:
        print("test top {} words on 300MB file".format(k))
        testtime = []
        for shard in shard_list:
            file_size = int(os.popen('wc -c < {}'.format(file_path)).read())
            chunk_size = int(np.ceil(file_size / shard))
            print("test chunk size: {}".format(chunk_size))
            print("test shard number: {}".format(shard))
            start_time = time.time()
            top_k_words(file_path, stopword_path, k, shard, 4)
            end_time = time.time()
            print("Average time taken: ", end_time - start_time)
            print("------------------------------------------------")
            testtime.append(end_time - start_time)
        resulttime[k] = testtime
    print("Complete test results:")
    return shard_list, resulttime

if __name__ == '__main__':
    MainUserInterface()