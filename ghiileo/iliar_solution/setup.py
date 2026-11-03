from setuptools import find_packages, setup

package_name = 'iliar_solution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/rviz_gui.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/simu_gui.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/teleop.launch.xml']),
        ('share/' + package_name + '/config', ['config/gui.rviz']),
        ('share/' + package_name + '/launch', ['launch/iliar_record.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/autopilot.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/autopilot_teleop.launch.xml']),
        ('share/' + package_name + '/models', ['models/model.onnx']),
        ('share/' + package_name + '/launch', ['launch/eval.launch.xml']),

        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ghiles',
    maintainer_email='ghiles.besseghir6@etu.univ-lorraine.fr',
    description='Solution ILiAR – fichiers launch pour la simulation et l\'interface RViz',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'teleop_command = iliar_solution.teleop_command:main',
            'record_dataset = iliar_solution.record_dataset:main',
            'path_publisher = iliar_solution.path_publisher:main',
            'path_follower = iliar_solution.path_follower:main',
            'dist_to_path = iliar_solution.dist_to_path:main',
            'autopilot = iliar_solution.autopilot:main',
            'command_mux = iliar_solution.command_mux:main',
            'path_dilater = iliar_solution.path_dilater:main',
            'scorer = iliar_solution.scorer:main',
            'traveled_distance = iliar_solution.traveled_distance:main'
        ],
    },
)
