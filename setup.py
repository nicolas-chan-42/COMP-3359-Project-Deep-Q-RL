from setuptools import setup, find_packages

setup(
    name="losing_connect_four",
    version="0.1",
    description="HKU COMP 3359 Project: "
                "A RL agent that cannot win in playing connect four",
    licence="MIT",
    url="https://github.com/nicolas-chan-42/COMP-3359-Project-Deep-Q-RL",
    author="Howard Chan, Nicolas Chan",
    author_email="u3537635@connect.hku.hk, ncsy@connect.hku.hk",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=[
        "gym>=0.17.1",
        "tensorflow>=2.1.0",
        'tensorflow-gpu>=2.1.0',
        "keras>=2.3.1",
        "numpy>=1.18.1",
        "matplotlib>=3.1.0",
        "pandas>=1.0.3",
        "pygame~=1.9.6",
        "setuptools~=46.1.3",
        "h5py",
        "tqdm",
    ],
    project_urls={
        "Source": "https://github.com/nicolas-chan-42/COMP-3359-Project-Deep-Q-RL",
        "Gym Reference Source": "https://github.com/IASIAI/gym-connect-four"
    }
)
