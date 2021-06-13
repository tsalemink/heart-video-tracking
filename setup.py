"""

"""
from setuptools import setup

classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE
Programming Language :: Python
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Topic :: Scientific/Engineering :: Medical Science Apps.
Topic :: Scientific/Engineering :: Visualization
Topic :: Software Development :: Libraries :: Python Modules
"""
with open('README.rst') as f:
    doc_lines = f.read().split("\n")

requires = ['opencv_python', 'opencv_contrib_python', 'numpy', 'scipy', 'scikit_image', 'pillow']

setup(
    name='sparc.videotracking',
    author='M. Osanlouy',
    author_email='mahyar@auckland.ac.nz',
    packages=['sparc', 'sparc.videotracking'],
    package_dir={'': 'src'},
    platforms=['any'],
    url='https://github.com/mahyar/heart_video_tracking',
    license='GNU GENERAL PUBLIC LICENSE',
    description=doc_lines[0],
    classifiers=classifiers.split("\n"),
    requires=requires,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
)
