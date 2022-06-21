from setuptools import setup, find_packages
setup(name='gym_A1',
      packages=find_packages(),
      include_package_data=True,
      package_data={
          '': ['*.png', '*.urdf', '*.pth'],
      },
      version='0.0.1',
      install_requires=['gym', 'numpy']
      )
