import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
nltk.download('punkt')


''' Set the seed for reproducibility'''
np.random.seed(5)

''' This function is to avoid the CParserError due to body column containing commas'''
def manual_separation(bad_line):
    right_split = bad_line[:-2] + [",".join(bad_line[-2:])] # All the "bad lines" where all coming from the same last column that was containing ","
    return right_split


column_names = ["anime_id", "anime_url", "title", "synopsis","main_pic","type","source_type", "num_episodes","status","start_date" ]
animes_df = pd.read_csv('anime.csv', sep='\t')
#print("Number of animes loaded: " + str(len(animes_df)))
#print(animes_df.head())
#print(animes_df["synopsis"].isna().sum())

''' Dropping missing value rows. Next updated version of this system must have the
    latest dataset because the one used is 2 years old. This reduces 12k rows to 6k rows of data.'''
animes_df = animes_df.dropna(subset=["synopsis"])
#print("Number of animes loaded after cleaend: " + str(len(animes_df)))

# This variable is specifically to view the dendrogram
# 500 rows of data only to avoid recursionError of max 999 calls
#animes_df = animes_df.head(900)
#print("locating problem: " + animes_df["title"].head())
#animes_df.to_csv('commaproblem.csv', index=False)

''' Tokenization Process'''

def tokenize_and_stem(text):

    '''Tokenize sentences first (Outer loop), then word (Inner loop)'''
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]

    '''Filter out raw tokens to remove noise'''
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    ''' Stem the filtered tokens'''
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in filtered_tokens]

    return stems

''' Tfidf Vectorization'''

tfidf_vectorizer = TfidfVectorizer(
    max_df = 0.98,
    max_features = 200000,
    min_df = 0.01,
    stop_words='english',
    use_idf=True,
    tokenizer=tokenize_and_stem,
    ngram_range = (1,3)
)

''' Fitting and transforming the tfidf_vectorizer with the "synopsis" of each animes 
    to create a vector representation of the anime summaries'''

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in animes_df["synopsis"]])
#print(tfidf_matrix.shape) # (6328, 124)
#print(tfidf_matrix[:5]) # test
#print(animes_df["synopsis"].head())

''' Clustering data '''

km = KMeans(n_clusters=5, random_state=42, init='k-means++')
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print(clusters[:5])

cluster_df = pd.DataFrame(
    {'Cluster': clusters }
)

animes_df["Cluster"] = clusters

#print(cluster_df)

''' Display number of animes per cluster (clusters from 0 to 4) '''
#print("number of animes per cluster: " + str(cluster_df['Cluster'].value_counts()))
#print("number of animes per cluster: " + str(animes_df['Cluster'].value_counts()))

''' Visualising the distribution'''
plt.hist(clusters, bins=range(6), align='left', rwidth=0.8)
plt.xlabel('Clusters')
plt.ylabel('Number of Animes')
plt.title('Anime Distribution Across Clusters')
plt.show() # A rather unbalanced distribution is produced (w/out randomstate and init)

''' Calculate similarity distance '''
similarity_distance = 1 - cosine_similarity(tfidf_matrix)
#print(similarity_distance)

''' Create mergings matrix '''
mergings = linkage(similarity_distance, method='complete')

#print(animes_df["title"])

''' Plot the dendrogram, using title as label column'''
dendrogram_ = dendrogram(mergings,
               labels=[x for x in animes_df["title"]],
               leaf_rotation=90,
               leaf_font_size=10
)

# Adjust the plot

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

#plt.show() # matplotlib does not allow scrolling of the plot
plt.savefig('dendrogram_all.png')
#print(animes_df["title"].tolist())
print(animes_df.head())


# further analysis using cluster plot

pca = PCA(n_components=2)
pca_result = pca.fit_transform(similarity_distance)

# Add PCA results to the DataFrame
animes_df['PCA1'] = pca_result[:, 0]
animes_df['PCA2'] = pca_result[:, 1]

# Plot the scatter plot with seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', style='Cluster', data=animes_df, palette='viridis', legend='full', markers=True, s=100, edgecolor='black')
plt.title('Cluster Plot of Animes')
plt.show()

def similar_animes(anime_title, num_similar=5):
    # Get the index of the given anime title
    anime_indices = animes_df.index[animes_df['title'] == anime_title].tolist()

    # Check if the list is not empty
    if not anime_indices:
        print(f"Anime title '{anime_title}' not found.")
        return []

    anime_index = anime_indices[0]

    # Get cluster assignments
    clusters = fcluster(mergings, t=5, criterion='maxclust')  # Adjust the threshold 't' based on your clustering result

    # Find the cluster of the given anime
    target_cluster = clusters[anime_index]

    # Find animes in the same cluster
    similar_animes_indices = [i for i, cluster in enumerate(clusters) if cluster == target_cluster]

    # Exclude the input anime from the list
    similar_animes_indices.remove(anime_index)

    # Filter out indices that are not present in the DataFrame
    valid_indices = [index for index in similar_animes_indices if index in animes_df.index]

    # Print the titles of similar animes one by one
    print(f"Animes similar to '{anime_title}':")
    for index in valid_indices[:num_similar]:
        similar_anime_title = animes_df.loc[index, 'title']
        print(f"- {similar_anime_title}")

    # Return a list of similar anime titles
    return [similar_anime_title for index in valid_indices[:num_similar]]

similar_animes("Bocchi the Rock!")










