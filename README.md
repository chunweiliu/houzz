# houzz
A demo program for the data from [houzz.com](https://www.houzz.com/)

##Data
The data is assembled by [Sirion Vittayakorn](https://www.cs.unc.edu/~sirionv/).

    Houzz
        - images
        - keys
        - lib
        - metadata

The content is categorized by the type of the room, e.g. bedroom, home-office, or kitchen.

##Dependencies
[SciPy + NumPy](http://www.scipy.org/)
[NLTK](http://www.nltk.org/)
[gensim](https://radimrehurek.com/gensim/index.html)
[Caffe](http://caffe.berkeleyvision.org/)
[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

SciPy, NumPy, and NLTK are all included in the [Anaconda] 
distribution of Python.


##Installation
In __init__.py, set DATASET_ROOT to where your dataset is located.
