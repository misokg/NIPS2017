# The misoKG algorithm for Multi-Information Source Optimization

## What is Multi-Information Source Optimization?
In multi-information source optimization (MISO) the task is to optimize an expensive-to-evaluate black-box objective function while optionally accessing cheaper biased noisy approximations, often referred to as "information sources (IS)".
This setting has multiple applications (see [the spotlight video][7] and [the NIPS paper][3] for details):
* hyperparameter tuning of machine learning algorithms: we can evaluate hyperparameters on a small related data set
or a subset of the validation set.
* robotics: we may evaluate a parameterized robot control policy in simulation, in a laboratory, or in a field test.
* aerospace engineering: we may test a design in a simulation or in a wind-tunnel experiment.

While cheap approximations offer a route to drastically reduce optimization costs, they are often subject to unknown bias and noise. Specifically, information sources typically exhibit "model discrepancy", originating from inherent limitations of their underlying computational models to model real-world phenomena.



## The misoKG algorithm
The misoKG algorithm proposed by [Matthias Poloczek][4], [Jialei Wang][6], and [Peter Frazier][5] is tolerant to both noise and bias. Therefore, it is able to efficiently leverage cheap information sources, maximizing the expected incremental gain per unit cost.

An illustration of the MISO-specific generative model is given [here][8].


---


## Installation
This implementation of the misoKG algorithm is built upon the [Metric Optimization Engine][1] (MOE), a global, black box optimization engine for real world metric optimization.

We provide a step-by-step guide to install misoKG and MOE. We have tested this procedure on Ubuntu 14.04.3 LTS.
First we install MOE following the steps described [here][2].
Then we will install misoKG.


#### 1. Install software required by MOE: 
MOE requires python 2.6.7+, gcc 4.7.3+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+.

```bash
$ sudo apt-get update
$ sudo apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git python-numpy python-scipy
```


#### 2. Setup a virtual environment:
It is recommended to install MOE into a virtual environment. Substitute VIRT_ENV, e.g., by 'env_misoKG' in

```bash
$ pip install virtualenv
$ virtualenv --no-site-packages VIRT_ENV
```


#### 3. Set environment variables for compiling the C++ part of MOE. We suggest creating a textfile "moe_setup.sh" with the following content:
```bash
export MOE_CXX_PATH=/path/to/g++
export MOE_CC_PATH=/path/to/gcc
export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/path/to/include/python2.7 -D MOE_PYTHON_LIBRARY=/path/to/libpython2.7.so"
```
Then you may source this file via `$ source ./moe_setup.sh`.

Note that the file "libpython2.7.so" can be located via
```bash
$ sudo updatedb
$ locate libpython2.7.so
```


#### 4. Install MOE in the virtualenv:
```bash
$ source VIRT_ENV/bin/activate
(VIRT_ENV) $ git clone -b jialei_mixed_square_exponential https://github.com/Yelp/MOE.git
(VIRT_ENV) $ cd MOE
(VIRT_ENV) $ pip install -r requirements.txt
(VIRT_ENV) $ python setup.py install
```


#### 5. Install misoKG:
```bash
(VIRT_ENV) $ cd ..
(VIRT_ENV) $ git clone https://github.com/misokg/NIPS2017.git
(VIRT_ENV) $ cd misoKG
(VIRT_ENV) $ pip install -r requirements.txt
```

---


## Example running misoKG on a benchmark
Run
```bash
(VIRT_ENV) $ python run_misoKG.py miso_rb_benchmark_mkg 0 0
```
for the Rosenbrock function proposed by Lam, Allaire, and Willcox (2015), or 
```bash
(VIRT_ENV) $ python run_misoKG.py miso_rb_benchmark_mkg 1 0
```
for the noisy variant proposed in the MISO paper.

The results are stored in a pickled dictionary in the current working directory. The filename is output when the program starts.
Note that the last parameter is a nonnegative integer that is used for the filename, e.g., when running multiple replications.



## Citation
```bash
@inproceedings{pwf17,
  title={Multi-Information Source Optimization},
  author={Poloczek, Matthias and Wang, Jialei and Frazier, Peter I.},
  booktitle={Advances in Neural Information Processing Systems},
  note={Accepted for publication. Code is available at \url{http://github.com/misokg}},
  year={2017}
}
```


## Supported Platforms
Python 2.7 and higher. 

Currently, there is no time line on Python 3+ support.


## The Spotlight Presentation
[![Spotlight video on Youtube](http://img.youtube.com/vi/edgbDQJKzTo/0.jpg)](http://youtu.be/edgbDQJKzTo)

[1]: https://github.com/Yelp/MOE
[2]: http://yelp.github.io/MOE/install.html#install-from-source
[3]: https://papers.nips.cc/paper/7016-multi-information-source-optimization
[4]: http://www.sie.arizona.edu/poloczek
[5]: http://people.orie.cornell.edu/pfrazier/
[6]: http://www.linkedin.com/in/jialeiwang/
[7]: http://youtu.be/edgbDQJKzTo
[8]: https://github.com/misokg/NIPS2017/blob/master/illustration_model.md
