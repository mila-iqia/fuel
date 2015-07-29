.. image:: https://img.shields.io/coveralls/mila-udem/fuel.svg
   :target: https://coveralls.io/r/mila-udem/fuel

.. image:: https://travis-ci.org/mila-udem/fuel.svg?branch=master
   :target: https://travis-ci.org/mila-udem/fuel

.. image:: https://readthedocs.org/projects/fuel/badge/?version=latest
   :target: https://fuel.readthedocs.org/

.. image:: https://img.shields.io/scrutinizer/g/mila-udem/fuel.svg
   :target: https://scrutinizer-ci.com/g/mila-udem/fuel/

.. image:: https://requires.io/github/mila-udem/fuel/requirements.svg?branch=master
   :target: https://requires.io/github/mila-udem/fuel/requirements/?branch=master

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/mila-udem/fuel/blob/master/LICENSE

Fuel
====

Fuel provides your machine learning models with the data they need to learn.

* Interfaces to common datasets such as MNIST, CIFAR-10 (image datasets), Google's One Billion Words (text), and many more
* The ability to iterate over your data in a variety of ways, such as in minibatches with shuffled/sequential examples
* A pipeline of preprocessors that allow you to edit your data on-the-fly, for example by adding noise, extracting n-grams from sentences, extracting patches from images, etc.
* Ensure that the entire pipeline is serializable with pickle; this is a requirement for being able to checkpoint and resume long-running experiments. For this, we rely heavily on the picklable_itertools_ library.

Fuel is developed primarily for use by Blocks_, a Theano toolkit that helps you train neural networks.

If you have questions, don't hesitate to write to the `mailing list`_.

Citing Fuel
   If you use Blocks or Fuel in your work, we'd really appreciate it if you could cite the following paper:
   
   Bart van Merriënboer, Dzmitry Bahdanau, Vincent Dumoulin, Dmitriy Serdyuk, David Warde-Farley, Jan Chorowski, and Yoshua Bengio, "`Blocks and Fuel: Frameworks for deep learning`_," *arXiv preprint arXiv:1506.00619 [cs.LG]*, 2015.
    
Documentation
   Please see the documentation_ for more information.


.. _picklable_itertools: http://github.com/dwf/picklable_itertools
.. _Blocks: http://github.com/mila-udem/blocks
.. _mailing list: https://groups.google.com/d/forum/fuel-users
.. _documentation: http://fuel.readthedocs.org/en/latest/
.. _Blocks and Fuel\: Frameworks for deep learning: http://arxiv.org/abs/1506.00619
