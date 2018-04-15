from setuptools import setup

setup(name='market_predictor',
      version='0.0.0',
      packages=['market_predictor','market_predictor.mp'],
      install_requires=[
          'quandl',
          'click',
          'coloredlogs'    
      ],
      entry_points={
          'console_scripts': [
              'market_predictor = market_predictor.__main__:main'
          ]
      },
)
