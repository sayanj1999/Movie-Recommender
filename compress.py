import bz2
import pickle
import os

print("Loading similarity.pkl...")
similarity = pickle.load(open('similarity.pkl', 'rb'))

print("Compressing and saving to similarity.pkl.bz2...")
with bz2.BZ2File('similarity.pkl.bz2', 'wb') as f:
    pickle.dump(similarity, f)

print("Done! Removing uncompressed file...")
os.remove('similarity.pkl')
print("similarity.pkl removed.")
