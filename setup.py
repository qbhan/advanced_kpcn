from distutils.core import setup

setup(
    name='kpcn',
    version='1.0',
    install_requires=[
        'torch', 'torchvision', 'scikit-image', 'scipy', 'pyexr'
    ]
)