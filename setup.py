from setuptools import setup, find_packages

setup(name='BottleCapDetector', 
    version='1.0', 
    author='Umer Khan',
    author_email='umer.khan@smail.inf.h-brs.de',
    install_requires=['numpy', 'cv2', 'sklearn']
    packages=find_packages()
    )