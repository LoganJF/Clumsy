from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='CML',
   version='0.0',
   description='CML Scripts',
   license="Upenn",
   long_description=long_description,
   author='Logan Fickling',
   author_email='loganfickling@gmail.com',
   url="http://www.foopackage.com/",
   packages=['CML'],  #same as name
   install_requires=['mne', 'ptsa'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)