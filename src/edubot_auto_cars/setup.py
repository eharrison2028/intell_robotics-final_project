from setuptools import find_packages, setup

package_name = 'edubot_auto_cars'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eharriso1',
    maintainer_email='eharriso1@ltu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_detector_threshold = edubot_auto_cars.lane_detector_threshold:main',
            'mapping = edubot_auto_cars.mapping:main'
        ],
    },
)
