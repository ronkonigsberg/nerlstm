FROM python:2

# Install python stuff
RUN pip install ipython
RUN pip install ipdb

# Install cmake && boost
RUN apt-get update -y
RUN apt-get install vim -y
RUN apt-get install cmake -y
RUN apt-get install libboost-all-dev -y

# Create workspace
RUN  mkdir -p /workspace

# Install CNN & checkout commit 0057fbb343c6aa341d41fbbea391698c4247a26f
RUN mkdir -p /workspace/clab && cd /workspace/clab && git clone https://github.com/clab/cnn.git && cd cnn && git submodule init && git submodule update
ENV PATH_TO_CNN /workspace/clab/cnn
RUN cd $PATH_TO_CNN && git checkout 0057fbb343c6aa341d41fbbea391698c4247a26f
# Install Eigen
RUN cd $PATH_TO_CNN && hg clone https://bitbucket.org/eigen/eigen/ -r 47fa289dda2dc13e0eea70adfc8671e93627d466
ENV PATH_TO_EIGEN /workspace/clab/cnn/eigen
RUN cd $PATH_TO_CNN && mkdir build && cd build && cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DBOOST_ROOT=$HOME/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON && make -j 2
# Install pycnn
RUN pip install cython
RUN pip install numpy
RUN cd $PATH_TO_CNN/pycnn && sed -i  "s|..\/..\/cnn\/|$PATH_TO_CNN|g" setup.py && sed -i  "s|..\/..\/eigen\/|$PATH_TO_EIGEN|g" setup.py && make && make install
ENV LD_LIBRARY_PATH $PATH_TO_CNN/pycnn

# Install nerlstm
RUN cd /workspace && git clone https://github.com/ronkonigsberg/nerlstm.git 
ENV LSTM_BASE_DIR /workspace/nerlstm
COPY conll /workspace/nerlstm/conll
COPY glove /workspace/nerlstm/glove
RUN cd $LSTM_BASE_DIR/src && python setup.py develop
