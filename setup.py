from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='optimistic_actor_critic',
      version='0.1',
      description='Tensorflow implementation of the Optimistic Actor Critic algorithm.',
      long_description=long_description,
      url='http://github.com/ac-93/optimistic_actor_critic',
      author='Alex Church',
      author_email='alexchurch1993@gmail.com',
      license='MIT',
      packages=['optimistic_actor_critic'],
      install_requires=[
      ],
      zip_safe=False)
