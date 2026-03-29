"""Setup script for Forex AI Bot."""
from setuptools import setup, find_packages

setup(
    name="forex-ai-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
    ],
    extras_require={
        "lstm": ["tensorflow>=2.14.0"],
        "mt5": ["MetaTrader5>=5.0.45"],
    },
    entry_points={
        "console_scripts": [
            "forex-bot=forex_ai_bot.run:main",
        ],
    },
    python_requires=">=3.9",
)
