from setuptools import setup

setup(
    name="diffpepbuilder",
    packages=[
        'data',
        'analysis',
        'model',
        'experiments',
        'openfold',
        'SSbuilder'
    ],
    package_dir={
        'data': './data',
        'analysis': './analysis',
        'model': './model',
        'experiments': './experiments',
        'openfold': './openfold',
        'SSbuilder': './SSbuilder'
    },
)
