import pandas as pd
import numpy as np
import re
import os
import time
import multiprocessing as mp

file_path = '/Users/sandy/Desktop/COEN242/PA1/dataset/data_300MB.txt'
stopword_path = '/Users/sandy/Desktop/COEN242/PA1/stopword.txt'

def shard_file(file_path, chunk_size):
    # split large file into smaller pieces
    with open(file_path, 'r') as file:
        shard_paths = []
        index = 0
        while True:
            chunk = file.readlines(chunk_size)
            if not chunk:
                break
            shard_path = f'shard_{index}.txt'
            with open(shard_path, 'w') as shard_file:
                shard_file.writelines(chunk)
            shard_paths.append(shard_path)
            index += 1
    return shard_paths

def process_shard(shard_path, stopwords):
    # read shard file and count word frequencies
    with open(shard_path, 'r') as file:
        text = file.read().lower()
        words = re.findall(r'\w+', text)
        words = [w for w in words if w.isalpha() and w not in stopwords]
        freqs = pd.Series(words).value_counts()
        freq_df = freqs.to_frame().reset_index()
        freq_df.columns = ['word', 'count']
    return freq_df

def count_words(chunk, stop_words):
    word_count = {}
    for word in chunk:
        if word not in stop_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count

def top_k_words(file_path, stopword_path, k, chunk_size=10_000_000, num_processes=4):
    stop_words = set(line.strip() for line in open(stopword_path))
    freq = {}
    
    # Determine chunk size and number of chunks
    file_size = os.path.getsize(file_path)
    num_chunks = file_size // chunk_size + 1
    
    # Set up multiprocessing pool
    pool = mp.Pool(num_processes)
    
    # Process each chunk in a separate process
    results = []
    with open(file_path) as f:
        for i in range(num_chunks):
            chunk = f.read(chunk_size)
            results.append(pool.apply_async(count_words, args=(re.findall(r'\w+', chunk.lower()), stop_words)))
    
    # Combine word counts from all processes
    for result in results:
        word_count = result.get()
        for word in word_count:
            if word in freq:
                freq[word] += word_count[word]
            else:
                freq[word] = word_count[word]
    
    # Convert word frequencies to DataFrame and sort
    freq_df = pd.DataFrame({'word': list(freq.keys()), 'count': list(freq.values())})
    freq_df = freq_df[freq_df['word'].apply(lambda x: x.isalpha())]
    freq_df = freq_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    # Print top k words and their frequencies
    print(freq_df.head(10))

if __name__ == '__main__':
    start_time = time.time()
    top_k_words(file_path, stopword_path,10)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds')