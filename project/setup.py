from setuptools import setup

setup(name='market_predict',
      version='0.0.0',
      packages=['market_predict','market_predict.mp'],
      install_requires=[],
      entry_points={
          'console_scripts': [
              'market_predict = market_predict.__main__:main'
          ]
      },
)
