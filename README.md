# PLSR-words-translation
PLSR word translation

This work is based on the code provided by Aurelie Herbelot at https://github.com/ml-for-nlp/word-translation, featuring some implementations on the code and the data analysis. 

<br>
<br>

## **FOLDER word-translation ENG_CAT:**
1. Original Dataset
2. Newly created PNGs of English and Catalan subspaces
3. Original code with some implementations:
   - Autocheck for different ncomps and nns values;
   - Creating a Table with different precision scores;
   - Modification of the PCA and PNG creation.


to run plsr_regression_ENG_CAT: 
```bash
python plsr_regression_ENG_CAT.py 
```

<br> 
<br>

## **FOLDER word-translation ENG_ITA:**
1. Original Dataset
2. Newly created pairs in English-Italian
3. Newly created PNGs of English and Italian subspaces
4. Newly created PNGs of English and Italian sub-subspaces of words present in pairs
5. Original code with some implementations:
   - Autocheck for different ncomps and nns values;
   - Creating a Table with different precision scores;
   - Modification of the PCA and PNG creation;
   - Subspaces both for Italian and English sets and for Italian and English words present in pairs.

to run plsr_regression_ENG_ITA: 
```bash
python plsr_regression_ENG_ITA.py 
```
_The command to create the space based on words present in pairs is commented and needs to be uncommented to work._ 
