# The misoKG algorithm for Multi-Information Source Optimization

# Coming soon!

## What is Multi-Information Source Optimization?
In multi-information source optimization (MISO) the taks is to optimize an expensive-to-evaluate black-box objective function while optionally accessing cheaper biased noisy approximations, often referred to as "information sources (IS)".
This setting has multiple applications (see [the NIPS paper][3] for details):
* hyperparameter tuning of machine learning algorithms: we can evaluate hyperparameters on a small related data set
or a subset of the validation set.
* robotics: we may evaluate a parameterized robot control policy in simulation, in a laboratory, or in a field test.
* reinforcement learning: we may run the policy of a simulation of nature.
While cheap approximations offer a route to drastically reduce optimization costs, they are often subject to unknown bias and noise. Specifically, information sources typically exhibit "model discrepancy", originating from inherent limitations of their underlying computational models to model real-world phenomena.

## The misoKG algorithm
The misoKG algorithm proposed by [Matthias Poloczek][4], [Jialei Wang][6], and [Peter Frazier][5] is tolerant to both noise and bias and efficiently leverages cheap information sources, maximizing the expected incremental gain per unit cost.

An example illustrating the statistical model of misoKG:
<center><img src="https://github.com/misoKG/NIPS2017/example/stat-model.gif" height="300" width="600"></center>


## Installation
This implementation of the misoKG algorithm is built upon the [Metric Optimization Engine][1] (MOE), a global, black box optimization engine for real world metric optimization.

We provide a step-by-step guide to install misoKG and MOE. We have tested this procedure on Ubuntu 14.04.3 LTS.
First we install MOE following the steps described [here][2].
Then we will install misoKG.

#### 1. Install software required by MOE: 
MOE requires python 2.6.7+, gcc 4.7.3+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+.

```bash
$ apt-get update
$ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git python-numpy python-scipy
```

#### 2. Setup a virtual environment:
It is recommended to install MOE into a virtual environment. Substitute VIRT_ENV, e.g., by 'env_misoKG' in

```bash
$ pip install virtualenv
$ virtualenv --no-site-packages VIRT_ENV
```

#### 3. Set environment variables for compiling the C++ part of MOE. We suggest creating a textfile "moe_setup.sh" with the following content:
```bash
export MOE_CC_PATH=/path/to/your/gcc
export MOE_CXX_PATH=/path/to/your/g++
export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/path/to/where/Python.h/is/found -D MOE_PYTHON_LIBRARY=/path/to/python/shared/library/object"
```
Then source this file via
```bash
source ./moe_setup.sh
```

#### 4. Install MOE in the virtualenv:
```bash
$ source VIRT_ENV/bin/activate
$ git clone -b jialei_mixed_square_exponential https://github.com/Yelp/MOE.git
$ cd MOE
$ pip install -r requirements.txt
$ python setup.py install
```

#### 5. Install misoKG:
```bash
$ source VIRT_ENV/bin/activate
$ git clone https://github.com/misokg/NIPS2017.git
$ cd misoKG
$ pip install -r requirements.txt
```

## Example running misoKG on a benchmark
```bash
$ coming soon
```

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
Python 2.7 and higher

[1]: https://github.com/Yelp/MOE
[2]: http://yelp.github.io/MOE/install.html#install-from-source
[3]: https://papers.nips.cc/paper/7016-multi-information-source-optimization
[4]: http://www.sie.arizona.edu/poloczek
[5]: http://people.orie.cornell.edu/pfrazier/
[6]: http://www.linkedin.com/in/jialeiwang/
