from setuptools import find_packages, setup

package_name = 'perception_tools_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/data', ['data/dummy_cloud.npy']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='britton',
    maintainer_email='brittonty@gmail.com',
    description='Perception tools for the dVRK, including a SAM 2 segmented point cloud publisher',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'segmented_pc_publisher = perception_tools_pkg.segmented_pc_publisher:main'
        ],
    },
)
