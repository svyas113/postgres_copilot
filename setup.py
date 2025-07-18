from setuptools import setup, find_packages

setup(
    name="postgres-copilot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "appdirs",
        "boto3",
        "google-generativeai",
        "litellm",
        "mcp",
        "psycopg2-binary",
        "python-dotenv",
        "lancedb",
        "colorama",
        "sentence-transformers",
        "pandas",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "postgres-copilot=postgres_copilot.postgres_copilot_chat:entry_point_main",
        ],
    },
)
