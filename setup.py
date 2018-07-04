from setuptools import setup

setup(
    name='mplplot',
    version='0.1',
    packages=['mplplot'],
    author='Keji Li',
    author_email='mail@keji.li',
    install_requires=['numpy', 'matplotlib', 'seaborn'],
    extras_require={'network': ['networkx']},
    description='matplotlib customizations and customized ploting functions'
)
